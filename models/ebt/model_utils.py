import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import traceback
from torchvision.transforms import functional as TF
import torchvision.models as models
from diffusers import AutoencoderKL
import math
import random
import numpy as np
from functools import partial
from PIL import Image
from torchvision.transforms import ToPILImage
from datetime import datetime
import torch.distributed as dist
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class EBTModelArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 64
    max_seq_len: int = 16
    weight_initialization: str = "xavier"
    adaln_zero_init: bool = True
    ebt_norm: str = "rms"
    ebt_act_func: str = "silu"
    weight_initialization_gain: float = 1.0

model_sizes = { # small -> xl same as mamba https://arxiv.org/pdf/2312.00752
    "4xs": {
        "num_transformer_blocks": 2,
        "multiheaded_attention_heads": 2,
        "embedding_dim": 128,
    },
    "3xs": {
        "num_transformer_blocks": 4,
        "multiheaded_attention_heads": 4,
        "embedding_dim": 256,
    },
    "xxs": {
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "2xs": { # same as xxs
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "xs": {
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "small": {
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "medium": {
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "large": {
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1536,
    },
    "xl": {
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 32,
        "embedding_dim": 2048,
    },
}




class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out  # Add the residual connection

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        if layer_norm:
            self.layers.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.layers.append(ResidualBlock(hidden_size, dropout_rate))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.layers.append(ResidualBlock(hidden_size, dropout_rate))
        else:
            self.layers.append(nn.Linear(hidden_size, final_size, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def calc_out_of_bounds_loss(energy): # gives loss for < 0 or > 1
    lower_bound_loss = torch.abs(energy)
    upper_bound_loss = torch.abs(energy - 1)
    loss = torch.where(energy < 0, lower_bound_loss, 
                    torch.where(energy > 1, upper_bound_loss, torch.zeros_like(energy)))
    loss = torch.mean(loss)
    
    return loss

def log_pred_futures(futures, device, dataset_name, i, denormalize):
    denormalized_futures = denormalize(futures.clone(), dataset_name, device = device)

    to_pil = ToPILImage()
    for b in range(denormalized_futures.shape[0]):  # Loop over the batch size
        if b % 16 == 0:
            for s in range(denormalized_futures.shape[1]):  # Loop over the sequence length
                frame_to_save = to_pil(denormalized_futures[b, s].cpu())  # Extract a frame (C x W x H)
                
                # Save the image
                current_time = datetime.now().strftime("%H_%M_%S")
                frame_to_save.save(f"./logs/debug/mcmc_futures/{current_time}_batch_{b}_seq_{s}_dev_{device}_iter_{i}.png")

def denormalize(tensor, dataset_name, device, custom_normalization):
    tensor = tensor.clone().detach()

    # Define default normalization values
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    default_mean = torch.tensor(default_mean, device=device).view(1, 1, 3, 1, 1)
    default_std = torch.tensor(default_std, device=device).view(1, 1, 3, 1, 1)
    # Dataset-specific normalization lookup
    if custom_normalization:
        normal_lookup = {
            "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
            "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
            "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet": ([1, 1, 1], [0, 0, 0]),
            "something": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet1k": ([1, 1, 1], [0, 0, 0])
        }
        dataset_std, dataset_mean = normal_lookup.get(dataset_name, ([1, 1, 1], [0, 0, 0]))

        # Convert means and stds to tensors and reshape for broadcast compatibility
        dataset_mean = torch.tensor(dataset_mean, device=device).view(1, 1, 3, 1, 1)
        dataset_std = torch.tensor(dataset_std, device=device).view(1, 1, 3, 1, 1)
        

        # Perform denormalization
        # First reverse the dataset-specific normalization
        tensor = tensor * dataset_std + dataset_mean
    # Then reverse the default normalization
    return tensor * default_std + default_mean

# def scale_clamp(tensor, min_value, max_value): #this is made to be a differentiable version of torch's clamp
#     scale_down_factor = torch.where(tensor > max_value, tensor / max_value, torch.ones_like(tensor))
#     scale_up_factor = torch.where(tensor < min_value, tensor / min_value, torch.ones_like(tensor))
    
#     combined_scale_factor = torch.where(tensor > max_value, scale_down_factor, 
#                                         torch.where(tensor < min_value, scale_up_factor, torch.ones_like(tensor)))
    
#     scaled_tensor = tensor / combined_scale_factor
    
#     return scaled_tensor

def scale_clamp(tensor, min_value, max_value):
    scale_factor = torch.ones_like(tensor)
    scale_factor = torch.where(tensor > max_value, tensor / max_value, scale_factor)
    scale_factor = torch.where(tensor < min_value, tensor / min_value, scale_factor)
    
    scaled_tensor = tensor / scale_factor
    return scaled_tensor

def load_trained_pl_model(ckpt_path, new_hparams, for_inference = False):
    from base_model_trainer import ModelTrainer
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model = ModelTrainer(new_hparams)
    model.load_state_dict(checkpoint['state_dict'])
    if for_inference:
        model.cuda().eval()
        model.model.eval()
    return model.model

def print_model_layers_and_status(model):
    for name, module in model.named_modules():
        print(f'Layer: {name}, Type: {type(module).__name__}, Training Mode: {module.training}')

def init_whole_model_weights(model, weight_initialization_method, nonlinearity='linear', weight_initialization_gain=1.0):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                valid_nonlinearities = ['linear', 'relu', 'leaky_relu', 'selu', 'tanh']
                if nonlinearity not in valid_nonlinearities:
                    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}. Must be one of {valid_nonlinearities}")
                
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            else:
                raise ValueError(f"Unknown weight init method: {weight_initialization_method}")
    
    model.apply(init_weights)


def load_image_encoder(backbone_type, backbone_size):
    vit_backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
        
    if backbone_type == 'dinov2':
        backbone_name = vit_backbone_archs[backbone_size]
        backbone = torch.hub.load('facebookresearch/dinov2', model=f"dinov2_{backbone_name}")
        del backbone._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues
        return backbone
    elif backbone_type == "vae":
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        return vae
    else:
        raise NotImplementedError(f"Unspported backbone type: {backbone_type}")
    
def get_encoded_images(batch, backbone_type, image_encoder):
    if backbone_type == 'dinov2':
        return image_encoder(batch)
    elif backbone_type == "vae":
        return image_encoder.encode(batch).latent_dist.mean

    
def hinged_mse_loss(predictions, targets, margin=0.1):
    """
    Compute the Hinged MSE loss between predictions and targets.
    :param predictions: Predicted values.
    :param targets: Ground truth values.
    :param margin: The threshold below which errors are ignored.
    :return: Hinged MSE loss.
    """
    errors = torch.abs(predictions - targets)
    hinged_errors = torch.where(errors > margin, errors, torch.zeros_like(errors))
    loss = torch.mean(hinged_errors ** 2)
    return loss

def find_subsequences(input_tensor, sub_seq):
    sub_seq_len = len(sub_seq)
    batch_size, seq_len = input_tensor.shape
    sub_seq_tensor = torch.tensor(sub_seq, device=input_tensor.device)
    sub_seq_tensor = sub_seq_tensor.view(1, -1)
    windows = input_tensor.unfold(1, sub_seq_len, 1)
    matches = (windows == sub_seq_tensor).all(dim=2).long()
    
    if not matches.any(dim=1).all():
        raise ValueError("Sub-sequence not found in one or more sequences.")
    
    start_positions = matches.argmax(dim=1)
    return start_positions

def mask_q_tokens(input_tensor, tokenizer):
    '''
    input_tensor = [batch size, seq len]
    '''
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    answer_tag = tokenizer.encode("[[Answer]]:", add_special_tokens=True)
    
    answer_start_pos = find_subsequences(input_tensor, answer_tag)
    answer_start_pos += len(answer_tag)
    mask = torch.arange(seq_length, device=input_tensor.device).expand(batch_size, seq_length)
    mask = mask < answer_start_pos.unsqueeze(1)
    input_tensor = torch.where(mask, tokenizer.pad_token_id, input_tensor)
    
    return input_tensor

def analyse_tokens(input_tensor, tokenizer):
    '''for debugging only'''
    decode = tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    for i in range(input_tensor.shape[0]):
        print(input_tensor[i].tolist())
        print(decode[i])
        print('-'*60)

def calculate_synthetic_embedding_accuracy(predicted_embeddings, gt_embeddings, synthetic_threshold, synthetic_percent): 
    # Compute the absolute difference between predicted and ground truth embeddings
    diff = torch.abs(predicted_embeddings - gt_embeddings)
    
    # Check which elements are within the threshold
    within_threshold = (diff <= synthetic_threshold).float()
    
    # Compute the proportion of elements within the threshold for each embedding
    proportion_within_threshold = within_threshold.mean(dim=2)  # Average over embedding dimension (dim=2)
    
    # Check if the proportion satisfies the required percentage
    satisfied_embeddings = (proportion_within_threshold >= synthetic_percent)
    
    # Calculate accuracy: proportion of embeddings satisfying the required percentage
    accuracy = satisfied_embeddings.float().mean().item()
    
    return accuracy

def setup_ebt(hparams): # specifically for EBT not for baseline transformer
    # to prevent circular import
    from model.ebt import EBTDefault
    from model.ebt_time_embed import EBTTimeConcat
    from model.ebt_adaln import EBTAdaLN
    max_seq_len = hparams.context_length+1 # for next pred in context 
    max_seq_len = max_seq_len + 1 if hparams.ebt_type == "time_embed" else max_seq_len # need +1 since cat time embed on sequence dim

    adaln_zero_init = True if hparams.ebt_type == "adaln_zero" else False
    transformer_args = EBTModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=max_seq_len, weight_initialization = hparams.weight_initialization_method, adaln_zero_init=adaln_zero_init, ebt_norm=hparams.ebt_norm, ffn_dim_multiplier=hparams.ffn_dim_multiplier, ebt_act_func=hparams.ebt_act_func, weight_initialization_gain=hparams.weight_initialization_gain)
    
    if hparams.ebt_type == "default": # causal decoder trans for ebm https://arxiv.org/abs/2406.08862
        ebt = EBTDefault(params=transformer_args)
    elif hparams.ebt_type == "time_embed": # time embed
        ebt = EBTTimeConcat(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)
    else: # adaln or adaln_zero
        ebt = EBTAdaLN(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)

    return ebt

def setup_transformer(hparams): # specifically for baseline transformer
    from model.transformer import Transformer, TransformerModelArgs
    transformer_args = TransformerModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=hparams.context_length, weight_initialization = hparams.weight_initialization_method, ffn_dim_multiplier=hparams.ffn_dim_multiplier, weight_initialization_gain=hparams.weight_initialization_gain)
    transformer = Transformer(params=transformer_args)
    return transformer

def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq):
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log="all", log_freq = wandb_watch_log_freq)
    
    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm) 
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        # print("\nNon-LayerNorm modules:")
        # for name, _ in non_layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        # print("\nLayerNorm modules:")
        # for name, _ in layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        wandb_logger.watch(non_layernorm_container, log="all", log_freq=wandb_watch_log_freq)
        wandb_logger.watch(layernorm_container, log="parameters", log_freq=wandb_watch_log_freq)