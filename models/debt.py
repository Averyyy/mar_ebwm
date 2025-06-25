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
    
    Uses causal mask where pred tokens only attend to previous ground truth tokens.
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
        # Position embedding for 2*seq_len (real + pred sequences of length seq_len each)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.seq_len, embed_dim))
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
            max_seq_len=2 * self.seq_len,  # real + pred sequences
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

    def refine_embeddings_training(self, real_embeddings, time_embeddings):
        """
        Refine embeddings during training using MCMC.
        
        real_embeddings: (B, S, D) - ground truth tokens
        time_embeddings: (B, D) - class/time conditioning
        
        Returns: refined predicted embeddings (B, S, D)
        """
        B, S, D = real_embeddings.shape
        
        # Initialize predicted embeddings
        predicted_embeddings = self.corrupt_embeddings(real_embeddings)
        
        alpha = torch.clamp(self.alpha, min=1e-4)
        langevin_std = torch.clamp(torch.tensor(self.langevin_dynamics_noise), min=1e-6)
        
        # Construct sequences according to DEBT specification for EBTAdaLN compatibility:
        # For sequence length S, we create:
        # real_seq: [<start> <gt_token_0> <gt_token_1> ... <gt_token_{S-2}>]  # length S-1+1 = S
        # pred_seq: [<pred_token_0> <pred_token_1> ... <pred_token_{S-1}>]     # length S
        # Total length: 2*S (matches EBTAdaLN expectation of 2*(S-1) when S maps to S-1)
        
        start_tokens = self.start_token.expand(B, 1, -1)  # (B, 1, D)
        
        with torch.set_grad_enabled(True):
            for mcmc_step in range(self.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                
                # Add Langevin dynamics noise after warming up
                if self.finished_warming_up and self.langevin_dynamics_noise > 0:
                    ld_noise = torch.randn_like(predicted_embeddings) * langevin_std
                    predicted_embeddings = predicted_embeddings + ld_noise
                
                # real_seq: [<start> + first S-1 ground truth tokens]
                real_seq = torch.cat([start_tokens, real_embeddings[:, :-1]], dim=1)  # (B, S, D)
                # pred_seq: [all S predicted tokens] (corresponds to all S ground truth tokens)
                pred_seq = predicted_embeddings  # (B, S, D)
                
                # Concatenate real and predicted sequences for EBT: total length 2*S
                all_embeddings = torch.cat([real_seq, pred_seq], dim=1)  # (B, 2*S, D)
                
                # Add positional embeddings
                pos_embed_slice = self.pos_embed[:, :all_embeddings.size(1), :]
                all_embeddings = all_embeddings + pos_embed_slice
                
                # Get energy predictions from transformer
                energy_preds = self.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step, c=time_embeddings)
                energy_preds = energy_preds.reshape(-1, 1)
                
                # Compute gradients for MCMC update
                if self.training:
                    predicted_embeds_grad = torch.autograd.grad(
                        [energy_preds.sum()], [predicted_embeddings], 
                        create_graph=True, retain_graph=True
                    )[0]
                else:
                    predicted_embeds_grad = torch.autograd.grad(
                        [energy_preds.sum()], [predicted_embeddings]
                    )[0]
                
                # Update predicted embeddings
                predicted_embeddings = predicted_embeddings - alpha * predicted_embeds_grad
        
        return predicted_embeddings

    def refine_embeddings_inference(self, real_prefix, predicted_token, time_embeddings, mcmc_steps=None):
        """
        Refine a single predicted token during inference.
        
        real_prefix: (B, prefix_len, D) - previously generated tokens with start token
        predicted_token: (B, 1, D) - current token to refine
        time_embeddings: (B, D) - class conditioning
        
        Returns: refined predicted token (B, 1, D)
        """
        if mcmc_steps is None:
            mcmc_steps = self.mcmc_num_steps
            
        alpha = torch.clamp(self.alpha, min=1e-4)
        B = predicted_token.size(0)
        
        with torch.set_grad_enabled(True):
            for mcmc_step in range(mcmc_steps):
                predicted_token = predicted_token.detach().requires_grad_()
                
                # For inference, construct minimal sequence that matches EBT expectations
                # real_seq should contain start + generated tokens so far (up to current position)
                # pred_seq should contain current token being refined + padding
                
                current_pos = real_prefix.size(1) - 1  # subtract 1 for start token to get generation position
                
                if current_pos >= self.seq_len:
                    # If we've generated all tokens, just refine the last one
                    current_pos = self.seq_len - 1
                
                # Create fixed-length sequences for EBT
                # Real sequence: start + generated tokens + padding
                real_seq = torch.zeros(B, self.seq_len, self.embed_dim, device=predicted_token.device)
                real_seq[:, 0] = self.start_token.squeeze(0)  # start token
                
                # Fill in generated tokens up to current position
                if current_pos > 0 and real_prefix.size(1) > 1:
                    copy_len = min(current_pos, real_prefix.size(1) - 1)
                    real_seq[:, 1:1+copy_len] = real_prefix[:, 1:1+copy_len]
                
                # Fill remaining positions with end tokens
                if current_pos + 1 < self.seq_len:
                    real_seq[:, current_pos+1:] = self.end_token.expand(B, self.seq_len - current_pos - 1, -1)
                
                # Predicted sequence: current token at appropriate position + zeros elsewhere
                pred_seq = torch.zeros(B, self.seq_len, self.embed_dim, device=predicted_token.device)
                pred_seq[:, current_pos:current_pos+1] = predicted_token
                
                # Debug information for first MCMC step
                # if mcmc_step == 0:
                #     print(f"Inference refinement - current_pos: {current_pos}, real_prefix_len: {real_prefix.size(1)}")
                #     print(f"Real seq shape: {real_seq.shape}, Pred seq shape: {pred_seq.shape}")
                
                # Concatenate for EBT
                all_embeddings = torch.cat([real_seq, pred_seq], dim=1)  # (B, 2*seq_len, D)
                
                # Add positional embeddings
                pos_embed_slice = self.pos_embed[:, :all_embeddings.size(1), :]
                all_embeddings = all_embeddings + pos_embed_slice
                
                # if mcmc_step == 0:
                #     print(f"All embeddings shape: {all_embeddings.shape}")
                #     print(f"Pos embed slice shape: {pos_embed_slice.shape}")
                
                # Get energy prediction
                energy_preds = self.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step, c=time_embeddings)
                
                # if mcmc_step == 0:
                #     print(f"Energy preds shape: {energy_preds.shape}")
                
                # Extract energy for current predicted token
                if energy_preds.dim() == 3:
                    energy_sum = energy_preds[:, current_pos, 0].sum()
                else:
                    energy_sum = energy_preds.flatten()[current_pos]
                
                # Compute gradients and update
                grad = torch.autograd.grad(energy_sum, predicted_token)[0]
                predicted_token = predicted_token - alpha * grad
        
        return predicted_token.detach()

    def forward(self, x_start, labels):
        """Training forward pass."""
        # Convert image to tokens and project to embedding space
        gt_tokens = self.patchify(x_start)  # (B, S, token_embed_dim)
        real_embeddings = self.input_proj(gt_tokens)  # (B, S, embed_dim)
        
        # Get class conditioning
        time_embeddings = self.y_embedder(labels)  # (B, embed_dim)
        
        # Refine embeddings using MCMC
        refined_embeddings = self.refine_embeddings_training(real_embeddings, time_embeddings)
        
        # Project back to token space and reconstruct image
        pred_tokens = self.output_proj(refined_embeddings)  # (B, S, token_embed_dim)
        recon = self.unpatchify(pred_tokens)  # (B, C, H, W)
        
        # Compute reconstruction loss
        return ((recon - x_start) ** 2).mean()

    def sample_tokens(self, bsz, num_iter=None, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        """
        Autoregressive sampling during inference.
        
        Generates tokens one by one:
        Step 1: real: [<start>], pred: [<pred_token_0>]
        Step 2: real: [<start> <pred_token_0>], pred: [<pred_token_1>]
        ...
        
        Args:
            bsz: batch size
            num_iter: number of iterations/tokens to generate (will be clamped to seq_len)
            cfg: classifier-free guidance scale (not used in DEBT but kept for compatibility)
            cfg_schedule: CFG schedule (not used in DEBT but kept for compatibility) 
            labels: class labels
            temperature: sampling temperature (not used in DEBT but kept for compatibility)
            progress: whether to show progress
        """
        print("DEBT sampling started")
        
        # For DEBT, we always generate exactly seq_len tokens (no more, no less)
        # This is because unpatchify expects exactly seq_len tokens to reconstruct the image
        tokens_to_generate = self.seq_len
        
        # print(f"DEBT generating {tokens_to_generate} tokens for {self.seq_h}x{self.seq_w} image patches")
        # print(f"Requested num_iter: {num_iter}, but using seq_len: {tokens_to_generate}")
            
        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.y_embedder.num_embeddings - 1, (bsz,), device=device)
        
        # Disable gradient computation for model parameters during inference
        was_training = self.training
        self.eval()
        
        # Get class conditioning
        with torch.no_grad():
            time_embeddings = self.y_embedder(labels)
        
        # Initialize with start token
        start_tokens = self.start_token.expand(bsz, 1, -1)
        real_prefix = start_tokens.clone()
        generated_embeddings = []
        
        # Generate exactly seq_len tokens autoregressively
        for step in range(tokens_to_generate):
            if progress and step % 64 == 0:
                print(f"Generating token {step}/{tokens_to_generate}")
            
            # Initialize next token with noise
            if self.denoising_initial_condition == "zeros":
                next_token = torch.zeros(bsz, 1, self.embed_dim, device=device)
            else:
                next_token = torch.randn(bsz, 1, self.embed_dim, device=device)
            
            # Refine the token using MCMC (needs gradients)
            refined_token = self.refine_embeddings_inference(
                real_prefix, next_token, time_embeddings
            )
            
            # Add refined token to generated sequence
            generated_embeddings.append(refined_token.detach())
            
            # Update real prefix for next iteration
            real_prefix = torch.cat([real_prefix, refined_token.detach()], dim=1)
        
        # Convert generated embeddings to tokens and reconstruct image
        with torch.no_grad():
            all_generated = torch.cat(generated_embeddings, dim=1)  # (B, seq_len, embed_dim)
            
            # print(f"Generated embeddings shape: {all_generated.shape}")
            # print(f"Expected shape: ({bsz}, {self.seq_len}, {self.embed_dim})")
            
            pred_tokens = self.output_proj(all_generated)  # (B, seq_len, token_embed_dim)
            # print(f"Pred tokens shape: {pred_tokens.shape}")
            # print(f"Expected shape: ({bsz}, {self.seq_len}, {self.token_embed_dim})")
            
            result = self.unpatchify(pred_tokens)  # (B, C, H, W)
            # print(f"Final image shape: {result.shape}")
        
        # Restore training mode
        if was_training:
            self.train()
            
        return result


if __name__ == "__main__":
    # Test with smaller sequence length for faster testing
    model = DEBT(
        img_size=64,  # smaller image
        vae_stride=8,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mcmc_num_steps=2,
        class_num=10
    )
    
    # Create test data
    seq_len = model.seq_len  # This will be 8*8 = 64 for the above config
    x = torch.randn(2, 16, 8, 8)  # (B, C, H, W) - matching vae_stride=8
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
