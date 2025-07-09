import torch
import torch.nn as nn
import numpy as np
from functools import partial

from models.ebt.ebt_adaln import EBTAdaLN
from models.ebt.model_utils import EBTModelArgs


class DEBT(nn.Module):
    """Diffusion Energy‑Based Transformer (DEBT) –  implementation.
    
    Training: 
    real_embedding: [<start> <gt_token_0> <gt_token_1> <gt_token_2>]
    pred_embedding: [<pred_token_0> <pred_token_1> <pred_token_2> <end>]
    """

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        embed_dim=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        class_num=1000,
        dropout_prob=0.1,
        mcmc_num_steps=2,
        mcmc_step_size=0.01,
        langevin_dynamics_noise=0.0,
        denoising_initial_condition="random_noise",
        double_condition=False,
        training=True,
    ):
        super().__init__()

        # -------- image / patch ----------
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.vae_embed_dim = 16
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w  # S
        self.token_embed_dim = 16 * patch_size ** 2
        self.embed_dim = embed_dim

        self.mcmc_num_steps = mcmc_num_steps
        self.langevin_dynamics_noise = langevin_dynamics_noise
        self.denoising_initial_condition = denoising_initial_condition
        self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)), requires_grad=True)
        self.finished_warming_up = False

        self.y_embedder = nn.Embedding(class_num + 1, embed_dim)
        # Learnable positional embedding shared by real & pred sequences.
        # Length = S+2  (positions 0..S for real seq, 1..S+1 for pred seq)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 2, embed_dim))
        self.start_token = nn.Parameter(torch.randn(1, embed_dim))
        self.end_token = nn.Parameter(torch.randn(1, embed_dim))
        self.double_condition = double_condition

        self.input_proj = nn.Linear(self.token_embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, self.token_embed_dim)

        # Configure EBT for DEBT
        t_args = EBTModelArgs(
            dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            ffn_dim_multiplier=mlp_ratio,
            adaln_zero_init=False,
            max_seq_len=2 * (self.seq_len + 1),  # real + pred sequences (+1 for <start>/<end>)
            weight_initialization="xavier",
            weight_initialization_gain=1.0,
            final_output_dim=1,  # for energy output
        )
        self.transformer = EBTAdaLN(params=t_args, max_mcmc_steps=mcmc_num_steps)
        self.training = training

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.y_embedder.weight, 0, 0.02)
        nn.init.normal_(self.pos_embed, 0, 0.02)
        nn.init.normal_(self.start_token, 0, 0.02)
        nn.init.normal_(self.end_token, 0, 0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def patchify(self, x):
        """Convert image to patches: (B, C, H, W) -> (B, seq_len, token_embed_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        return x.reshape(B, h * w, -1)

    def unpatchify(self, x):
        """Convert patches back to image: (B, seq_len, token_embed_dim) -> (B, C, H, W)"""
        B = x.size(0)
        p = self.patch_size
        c = self.vae_embed_dim
        h = w = self.seq_h
        x = x.reshape(B, h, w, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        return x.reshape(B, c, h * p, w * p)

    def corrupt_embeddings(self, real_embeddings):
        """Generate initial corrupted embeddings for MCMC refinement."""
        if self.denoising_initial_condition == "zeros":
            return torch.zeros_like(real_embeddings)
        elif self.denoising_initial_condition == "random_noise":
            return torch.randn_like(real_embeddings)
        elif self.denoising_initial_condition == "most_recent_embedding":
            return real_embeddings.clone()
        else:
            raise ValueError(f"Unknown denoising_initial_condition: {self.denoising_initial_condition}")

    def refine_embeddings_training(self, real_embeddings, cond_embeddings):
        """
        Refine embeddings during training using MCMC.
        
        real_embeddings: (B, S, D) - ground truth tokens
        cond_embeddings: (B, D) - conditioning embeddings (e.g., class label)
        
        Returns: refined predicted embeddings (B, S, D)
        """
        B, S, D = real_embeddings.shape
        
        # Initialize predicted embeddings
        predicted_embeddings = self.corrupt_embeddings(real_embeddings)
        
        alpha = torch.clamp(self.alpha, min=1e-4)
        langevin_std = torch.clamp(torch.tensor(self.langevin_dynamics_noise), min=1e-6)
        
        start_tokens = self.start_token.expand(B, 1, D)
        end_tok_exp = self.end_token.expand(B, 1, D)

        
        with torch.set_grad_enabled(True):
            for mcmc_step in range(self.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                
                if self.finished_warming_up and self.langevin_dynamics_noise > 0:
                    ld_noise = torch.randn_like(predicted_embeddings) * langevin_std
                    predicted_embeddings = predicted_embeddings + ld_noise
                
                # real_seq: [<start>] + all S ground-truth tokens
                real_seq = torch.cat([start_tokens, real_embeddings], dim=1)  # (B, S+1, D)
                
                # pred_seq: [predicted_tokens] + <end>
                pred_seq = torch.cat([predicted_embeddings, end_tok_exp], dim=1)  # (B, S+1, D)
                
                # Add positional embeddings: real uses indices 0..S, pred uses 1..S+1 (shift by +1)
                real_seq_pe = self.pos_embed[:, :real_seq.size(1), :]
                pred_seq_pe = self.pos_embed[:, 1:1 + pred_seq.size(1), :]
                real_seq = real_seq + real_seq_pe
                pred_seq = pred_seq + pred_seq_pe

                all_embeddings = torch.cat([real_seq, pred_seq], dim=1)  # (B, 2*(S+1), D)
                
                # EBT forward pass
                energy_preds = self.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step, c=cond_embeddings)
                energy_preds = energy_preds.reshape(-1, 1)
                
                # MCMC update
                if self.training:
                    predicted_embeds_grad = torch.autograd.grad(
                        [energy_preds.sum()], [predicted_embeddings], 
                        create_graph=True, retain_graph=True
                    )[0]
                else:
                    predicted_embeds_grad = torch.autograd.grad(
                        [energy_preds.sum()], [predicted_embeddings]
                    )[0]
                
                predicted_embeddings = predicted_embeddings - alpha * predicted_embeds_grad
        
        return predicted_embeddings

    def refine_embeddings_inference(self, real_prefix, predicted_token, cond_embeddings, mcmc_steps=None):
        """
        Refine a single predicted token during inference.
        
        real_prefix: (B, prefix_len, D) - previously generated tokens with start token
        predicted_token: (B, 1, D) - current token to refine
        cond_embeddings: (B, D) - conditioning embeddings (e.g., class label)
        
        Returns: refined predicted token (B, 1, D)
        """
        if mcmc_steps is None:
            mcmc_steps = self.mcmc_num_steps
            
        alpha = torch.clamp(self.alpha, min=1e-4)
        B = predicted_token.size(0)
        
        with torch.set_grad_enabled(True):
            for mcmc_step in range(mcmc_steps):
                predicted_token = predicted_token.detach().requires_grad_()
                
                # real_seq contain start + generated tokens so far (up to current position)
                # pred_seq contain current token being refined + padding
                
                current_pos = real_prefix.size(1) - 1  # start token is at position 0
                
                # if we've generated all tokens, just refine the last one
                if current_pos >= self.seq_len:
                    current_pos = self.seq_len - 1
                
                extended_len = self.seq_len + 1

                # real sequence: <start> + generated tokens + <end> padding
                real_seq = torch.zeros(B, extended_len, self.embed_dim, device=predicted_token.device)
                real_seq[:, 0] = self.start_token.squeeze(0)  # <start>

                # fill in the already generated tokens
                if current_pos > 0 and real_prefix.size(1) > 1:
                    copy_len = min(current_pos, real_prefix.size(1) - 1)
                    real_seq[:, 1:1 + copy_len] = real_prefix[:, 1:1 + copy_len]

                # fill remaining slots with <end>
                if current_pos + 1 < extended_len:
                    real_seq[:, current_pos + 1:] = self.end_token.expand(B, extended_len - current_pos - 1, -1)

                # Predicted sequence: place current predicted token, append fixed <end>
                pred_seq = torch.zeros(B, extended_len, self.embed_dim, device=predicted_token.device)
                pred_seq[:, current_pos:current_pos + 1] = predicted_token  # current token being refined
                pred_seq[:, -1] = self.end_token.squeeze(0)
                
                # Add positional embeddings with shift (same logic as training)
                real_seq_pe = self.pos_embed[:, :real_seq.size(1), :]
                pred_seq_pe = self.pos_embed[:, 1:1 + pred_seq.size(1), :]
                real_seq = real_seq + real_seq_pe
                pred_seq = pred_seq + pred_seq_pe

                all_embeddings = torch.cat([real_seq, pred_seq], dim=1)  # (B, 2*(seq_len+1), D)
                
                # Get energy prediction
                energy_preds = self.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step, c=cond_embeddings)
                
                # Extract energy for current predicted token
                energy_sum = energy_preds[:, current_pos, 0].sum()
                
                grad = torch.autograd.grad(energy_sum, predicted_token)[0]
                predicted_token = predicted_token - alpha * grad
        
        return predicted_token.detach()

    def forward(self, x_start, labels):
        """Training forward pass."""
        # Convert image to tokens and project to embedding space
        gt_tokens = self.patchify(x_start)  # (B, S, token_embed_dim)
        real_embeddings = self.input_proj(gt_tokens)  # (B, S, embed_dim)
        
        # Get class conditioning
        cond_embeddings = self.y_embedder(labels)  # (B, embed_dim)
        
        # Refine embeddings using MCMC
        refined_embeddings = self.refine_embeddings_training(real_embeddings, cond_embeddings)
        
        # Project back to token space and reconstruct image
        pred_tokens = self.output_proj(refined_embeddings)  # (B, S, token_embed_dim)
        recon = self.unpatchify(pred_tokens)  # (B, C, H, W)
        
        # Compute reconstruction loss
        return ((recon - x_start) ** 2).mean()

    def sample_tokens(self, bsz, num_iter=None, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, gt_prefix_tokens=None):       
        """
        Autoregressive sampling during inference.
        
                
        Generates tokens one by one:
        Step 1: real: [<start>], pred: [<pred_token_0>]
        Step 2: real: [<start> <pred_token_0>], pred: [<pred_token_1>]
        ...

        If gt_prefix_tokens is provided (B, prefix_len, token_embed_dim), the first
        half (or arbitrary prefix length) of the sequence will be taken directly
        from these ground-truth tokens and only the remaining tokens will be
        generated autoregressively.
        """  
        print("DEBT sampling started")
        
        # total tokens to generate in a full image (flattened patch grid)
        tokens_to_generate = self.seq_len

        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.y_embedder.num_embeddings - 1, (bsz,), device=device)

        was_training = self.training
        self.eval()

        # Get class conditioning
        with torch.no_grad():
            cond_embeddings = self.y_embedder(labels)

        # Initialize with start token
        start_tokens = self.start_token.expand(bsz, 1, -1)
        real_prefix = start_tokens.clone()
        generated_embeddings = []  # list[(B,1,D)]

        # ------------------------------------------------------------
        # Optional: prepend ground-truth prefix tokens
        # ------------------------------------------------------------
        if gt_prefix_tokens is not None:
            # gt_prefix_tokens shape: (B, prefix_len, token_embed_dim)

            # Project to embedding space
            prefix_embeddings = self.input_proj(gt_prefix_tokens.to(device))  # (B, L, D)
            prefix_len = prefix_embeddings.size(1)
            assert prefix_len <= self.seq_len, "Prefix length exceeds total sequence length"

            # Update bookkeeping
            tokens_to_generate = self.seq_len - prefix_len

            # Update prefix context for subsequent generation
            real_prefix = torch.cat([real_prefix, prefix_embeddings.detach()], dim=1)  # (B, 1+L, D)

            # Store prefix tokens for later reconstruction (avoid passing through
            # output_proj again, which would degrade them).
            tokens_prefix = gt_prefix_tokens.to(device)
        else:
            tokens_prefix = None

        # Generate exactly seq_len tokens autoregressively
        for step in range(tokens_to_generate):
            if progress and step % 64 == 0:
                print(f"Generating token {step}/{tokens_to_generate}")
            
            if self.denoising_initial_condition == "zeros":
                next_token = torch.zeros(bsz, 1, self.embed_dim, device=device)
            else:
                next_token = torch.randn(bsz, 1, self.embed_dim, device=device)
            
            refined_token = self.refine_embeddings_inference(
                real_prefix, next_token, cond_embeddings
            )
            
            generated_embeddings.append(refined_token.detach())
            
            real_prefix = torch.cat([real_prefix, refined_token.detach()], dim=1)
        
        # Convert generated embeddings to tokens and reconstruct image
        with torch.no_grad():
            if tokens_to_generate > 0:
                all_generated = torch.cat(generated_embeddings, dim=1)  # (B, seq_len - L, embed_dim)
                generated_tokens = self.output_proj(all_generated)  # (B, seq_len - L, token_embed_dim)
            else:
                generated_tokens = torch.empty(bsz, 0, self.token_embed_dim, device=device)

            # Combine ground-truth prefix tokens (if any) with generated tokens
            if gt_prefix_tokens is not None:
                pred_tokens = torch.cat([tokens_prefix, generated_tokens], dim=1)
            else:
                pred_tokens = generated_tokens  # no prefix case = full generation

            # Safety check
            assert pred_tokens.size(1) == self.seq_len, "Pred tokens length mismatch"

            result = self.unpatchify(pred_tokens)  # (B, C, H, W)
        
        # Restore training mode
        if was_training:
            self.train()
            
        return result


if __name__ == "__main__":
    model = DEBT(
        img_size=64,
        vae_stride=8,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mcmc_num_steps=2,
        class_num=10
    )
    
    # Create test data
    seq_len = model.seq_len  # This will be 8*8 = 64
    x = torch.randn(2, 16, 8, 8)  # (B, C, H, W)
    labels = torch.tensor([0, 1])
    
    print(f"Model sequence length: {seq_len}")
    print(f"Input shape: {x.shape}")
    
    # Test training
    print("\n=== Testing Training ===")
    model.training = True
    model.train()
    try:
        loss = model(x, labels)
        print("Training loss:", loss.item())
        print("✓ Training forward pass successful")
    except Exception as e:
        print("✗ Training failed:", str(e))
        import traceback
        traceback.print_exc()
    
    # Test inference
    print("\n=== Testing Inference ===")
    model.training = False
    model.eval()
    try:
        # Generate fewer tokens for faster testing
        sampled = model.sample_tokens(2, num_iter=4, labels=labels, progress=True)
        print("Sampled shape:", sampled.shape)
        print("✓ Inference successful")
    except Exception as e:
        print("✗ Inference failed:", str(e))
        import traceback
        traceback.print_exc()


def debt_2xs(**kwargs):
    """DEBT 2xs: 6 layers (3+3), 384 dim, 6 heads"""
    return DEBT(
        embed_dim=384,
        depth=6,  # 3 + 3
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )

def debt_base(**kwargs):
    """DEBT base: 24 layers (12+12), 768 dim, 12 heads"""
    return DEBT(
        embed_dim=768,
        depth=24,  # 12 + 12
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )

def debt_large(**kwargs):
    """DEBT large: 32 layers (16+16), 1024 dim, 16 heads"""
    return DEBT(
        embed_dim=1024,
        depth=32,  # 16 + 16
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )

def debt_huge(**kwargs):
    """DEBT huge: 40 layers (20+20), 1280 dim, 16 heads"""
    return DEBT(
        embed_dim=1280,
        depth=40,  # 20 + 20
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )
