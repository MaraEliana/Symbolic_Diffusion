import numpy as np
import math
import time
import random
import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from IPython import display

#import hyperparameters
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

def linear_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class DiscreteDiffusion:
    def __init__(self, num_timesteps=NUM_TIMESTEPS, vocab_size=VOCAB_SIZE, device=DEVICE):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.device = device

        if SCHEDULE_TYPE == 'linear':
            self.betas = linear_beta_schedule(num_timesteps).to(device)
        elif SCHEDULE_TYPE == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule type: {SCHEDULE_TYPE}")

        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        self.log_q_t_x_t_minus_1 = self._compute_log_q_t_x_t_minus_1()
        self.log_q_t_x_0 = self._compute_log_q_t_x_0()
        self.log_q_t_minus_1_x_t_x_0 = self._compute_log_q_t_minus_1_x_t_x_0()

    def _compute_log_q_t_x_t_minus_1(self):
        log_q = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size, device=self.device, dtype=torch.float64)
        eye = torch.eye(self.vocab_size, device=self.device)
        for t in range(self.num_timesteps):
            beta_t = self.betas[t]
            diag_indices = torch.arange(self.vocab_size, device=self.device)
            log_q[t, diag_indices, diag_indices] = torch.log(1.0 - beta_t + beta_t / self.vocab_size)
            off_diag_val = torch.log(beta_t / self.vocab_size)
            log_q[t] = log_q[t] + off_diag_val * (1.0 - eye)
        return log_q.float()

    def _compute_log_q_t_x_0(self):
        log_q = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size, device=self.device, dtype=torch.float64)
        eye = torch.eye(self.vocab_size, device=self.device)
        epsilon = 1e-40
        for t in range(self.num_timesteps):
            alpha_bar_t = self.alphas_cumprod[t]
            diag_indices = torch.arange(self.vocab_size, device=self.device)
            diag_term = alpha_bar_t + (1.0 - alpha_bar_t) / self.vocab_size
            off_diag_term = (1.0 - alpha_bar_t) / self.vocab_size
            log_diag_term_clamped = torch.log(diag_term.clamp(min=epsilon))
            off_diag_val_clamped = torch.log(off_diag_term.clamp(min=epsilon))
            log_q[t, diag_indices, diag_indices] = log_diag_term_clamped
            log_q[t] = log_q[t] + off_diag_val_clamped * (1.0 - eye)
        return log_q.float()

    def _compute_log_q_t_minus_1_x_t_x_0(self):
        log_q_posterior = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size, self.vocab_size, device=self.device, dtype=torch.float64)
        log_q_t_x_t_minus_1_64 = self.log_q_t_x_t_minus_1.double()
        log_q_t_x_0_64 = self.log_q_t_x_0.double()
        for t in range(1, self.num_timesteps):
            log_q_t_given_t_minus_1 = log_q_t_x_t_minus_1_64[t]
            log_q_t_minus_1_given_0 = log_q_t_x_0_64[t-1]
            log_q_posterior[t] = log_q_t_given_t_minus_1.unsqueeze(1) + log_q_t_minus_1_given_0.unsqueeze(0)
        log_denominator = torch.logsumexp(log_q_posterior, dim=-1, keepdim=True)
        log_denominator = torch.where(torch.isinf(log_denominator), torch.zeros_like(log_denominator), log_denominator)
        log_q_posterior = log_q_posterior - log_denominator
        log_q_posterior = torch.clamp(log_q_posterior, -100.0, 0.0)
        return log_q_posterior.float()

    def q_sample(self, x_start, t):
        batch_size, seq_len = x_start.shape
        log_q_t_x_0_for_batch_t = self.log_q_t_x_0[t]
        x_start_expanded = x_start.unsqueeze(-1)
        log_q_t_x_0_expanded = log_q_t_x_0_for_batch_t.unsqueeze(1).expand(-1, seq_len, -1, -1)
        x_start_indices = x_start_expanded.unsqueeze(-1).expand(-1, -1, self.vocab_size, -1)
        x_start_indices = torch.clamp(x_start_indices, 0, self.vocab_size - 1)
        log_probs = torch.gather(log_q_t_x_0_expanded, dim=3, index=x_start_indices).squeeze(-1)
        gumbel_noise = torch.rand_like(log_probs)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
        x_t = torch.argmax(log_probs + gumbel_noise, dim=-1)
        return x_t.long()

    def q_posterior_log_probs(self, x_0, x_t, t):
        batch_size, seq_len = x_0.shape
        log_q_posterior_t = self.log_q_t_minus_1_x_t_x_0[t]
        log_q_posterior_t = log_q_posterior_t.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        x_t_idx = x_t.view(batch_size, seq_len, 1, 1, 1).expand(-1, -1, -1, self.vocab_size, self.vocab_size)
        x_t_idx = torch.clamp(x_t_idx, 0, self.vocab_size - 1)
        log_q_posterior_t_i = torch.gather(log_q_posterior_t, dim=2, index=x_t_idx).squeeze(2)
        x_0_idx = x_0.view(batch_size, seq_len, 1, 1).expand(-1, -1, -1, self.vocab_size)
        x_0_idx = torch.clamp(x_0_idx, 0, self.vocab_size - 1)
        log_q_posterior_t_i_j = torch.gather(log_q_posterior_t_i, dim=2, index=x_0_idx).squeeze(2)
        return log_q_posterior_t_i_j

    def p_log_probs(self, model, x_t, t, condition):
        log_pred_x0 = model(x_t, t, condition)
        return F.log_softmax(log_pred_x0, dim=-1)

    def p_sample(self, model, x_t, t, condition):
        batch_size, seq_len = x_t.shape
        device = x_t.device
        log_pred_x0 = self.p_log_probs(model, x_t, t, condition)
        log_q_posterior_t = self.log_q_t_minus_1_x_t_x_0[t]
        log_q_posterior_t = log_q_posterior_t.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        x_t_idx = x_t.view(batch_size, seq_len, 1, 1, 1).expand(-1, -1, -1, self.vocab_size, self.vocab_size)
        x_t_idx = torch.clamp(x_t_idx, 0, self.vocab_size - 1)
        log_q_posterior_t_i = torch.gather(log_q_posterior_t, dim=2, index=x_t_idx).squeeze(2)
        log_pred_x0_expanded = log_pred_x0.unsqueeze(-1)
        log_sum_terms = log_q_posterior_t_i + log_pred_x0_expanded
        log_p_t_minus_1_given_t = torch.logsumexp(log_sum_terms, dim=2)
        log_p_t_minus_1_given_t = F.log_softmax(log_p_t_minus_1_given_t, dim=-1)
        gumbel_noise = torch.rand_like(log_p_t_minus_1_given_t)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
        x_t_minus_1 = torch.argmax(log_p_t_minus_1_given_t + gumbel_noise, dim=-1)
        return x_t_minus_1.long()

# AI Modifed loss function
    def compute_loss(
            self,
            model,
            x_start,
            condition,
            *,
            pad_token_id: int = PAD_TOKEN_ID,
            eos_token_id: int = EOS_TOKEN_ID,
            eos_weight: float = EOS_LOSS_WEIGHT,
            pad_weight: float = 0.5,      
        ):
        """
        Cross-entropy on every position **including** PAD, with:
        • extra reward for predicting the first <eos>
        • mild penalty for any non-PAD emitted *after* the first <eos>
        """
        B, S = x_start.shape
        device = x_start.device

        # 1 pick diffusion timestep and corrupt the input
        t   = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        x_t = self.q_sample(x_start, t)                     # (B,S)

        # 2 model prediction (log-probs over vocab)
        log_pred_x0 = self.p_log_probs(model, x_t, t, condition)  # (B,S,V)

        # 3 plain NLL for every token (no ignore_index!)
        loss_tok = F.nll_loss(
            log_pred_x0.permute(0, 2, 1),   # (B,V,S)
            x_start,                        # (B,S)
            reduction='none'                # keep per-token loss
        )                                   # (B,S)

        # ------------------------------------------------------------------ #
        # 4 build a per-token weight matrix
        weights = torch.ones_like(x_start, dtype=torch.float, device=device)

        # (a) first <eos> in each sequence
        eos_mask = x_start == eos_token_id
        weights  += eos_mask * (eos_weight - 1.0)     # boost eos

        # (b) tokens AFTER the first <eos> are expected to be PAD
        eos_cum   = torch.cumsum(eos_mask.long(), dim=1)          # (B,S)
        should_be_pad = (eos_cum > 0) & ~eos_mask                # after-eos positions
        weights = torch.where(should_be_pad, pad_weight * weights, weights)

        # (c) “real” padding on the right of sequences *before* eos — ignore
        true_pad = (x_start == pad_token_id) & (eos_cum == 0)
        weights  = torch.where(true_pad, torch.zeros_like(weights), weights)

        # ------------------------------------------------------------------ #
        # 5 aggregate
        weighted_loss = loss_tok * weights
        denom = weights.sum().clamp(min=1)       # avoid div-by-zero
        loss  = weighted_loss.sum() / denom

        return loss

    @torch.no_grad()
    def sample(self, model, condition, shape):
        batch_size, seq_len = shape
        device = self.device
        model.eval()
        x_t = torch.randint(1, self.vocab_size, size=shape, device=device).long()

        for t in reversed(range(0, self.num_timesteps)):
            print(f"\rSampling timestep {t+1}/{self.num_timesteps}   ", end="")
            sys.stdout.flush()
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if t > 0:
                 x_t = self.p_sample(model, x_t, t_tensor, condition)
            else:
                 log_pred_x0 = self.p_log_probs(model, x_t, t_tensor, condition)
                 gumbel_noise = torch.rand_like(log_pred_x0)
                 gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
                 x_t = torch.argmax(log_pred_x0 + gumbel_noise, dim=-1).long()

        print("\nSampling complete.")
        model.train()
        return x_t


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe: torch.Tensor
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim=CONDITION_FEATURE_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )


        self.mlp2 = nn.Sequential(
            nn.Linear(256, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, point_cloud):
        x = point_cloud.permute(0, 2, 1) # (B, CONDITION_FEATURE_DIM, N_POINTS)
        point_features = self.mlp1(x) # (B, 256, N_POINTS)
        global_feature, _ = torch.max(point_features, dim=2) # (B, 256)
        condition_embedding = self.mlp2(global_feature) # (B, embed_dim)
        return condition_embedding


class ConditionalD3PMTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dim_feedforward,
                 seq_len,
                 condition_feature_dim,
                 num_timesteps, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=seq_len + 1)
        self.timestep_embedding = nn.Sequential(
            TimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.condition_encoder = PointCloudEncoder(input_dim=condition_feature_dim, embed_dim=embed_dim)

        self.condition_dropout_prob = 0.05 # Set to 0 to disable

        #Transformer Block Components
        self.encoder_self_attn_layers = nn.ModuleList()
        self.encoder_cross_attn_layers = nn.ModuleList()
        self.encoder_ffn_layers = nn.ModuleList()
        self.encoder_norm1_layers = nn.ModuleList()
        self.encoder_norm2_layers = nn.ModuleList()
        self.encoder_norm3_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.encoder_self_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.encoder_cross_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.encoder_ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim_feedforward, embed_dim)
            ))
            self.encoder_norm1_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm2_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm3_layers.append(nn.LayerNorm(embed_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.token_embedding.padding_idx is not None:
             self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        for layer in self.condition_encoder.modules():
             if isinstance(layer, (nn.Conv1d, nn.Linear)):
                 layer.weight.data.normal_(mean=0.0, std=0.02)
                 if layer.bias is not None:
                     layer.bias.data.zero_()
             elif isinstance(layer, nn.BatchNorm1d):
                 layer.weight.data.fill_(1.0)
                 layer.bias.data.zero_()

    def forward(self, x, t, condition):
        # CONDITION INPUT SHAPE: Expects (B, N_POINTS, CONDITION_FEATURE_DIM)
        batch_size, seq_len = x.shape
        device = x.device

        token_emb = self.token_embedding(x) * math.sqrt(self.embed_dim)
        token_emb_permuted = token_emb.transpose(0, 1)
        pos_emb_permuted = self.positional_encoding(token_emb_permuted)
        pos_emb = pos_emb_permuted.transpose(0, 1)
        time_emb = self.timestep_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        cond_emb_proj = self.condition_encoder(condition)
        if self.training and self.condition_dropout_prob > 0:
            mask = (torch.rand(cond_emb_proj.shape[0], 1, device=cond_emb_proj.device) > self.condition_dropout_prob).float()
            cond_emb_proj = cond_emb_proj * mask
        cond_kv = cond_emb_proj.unsqueeze(1)

        current_input = pos_emb + time_emb
        padding_mask = (x == PAD_TOKEN_ID)

        for i in range(self.num_layers):
            sa_norm_input = self.encoder_norm1_layers[i](current_input)
            sa_output, _ = self.encoder_self_attn_layers[i](query=sa_norm_input, key=sa_norm_input, value=sa_norm_input, key_padding_mask=padding_mask)
            x = current_input + self.dropout_layers[i](sa_output)
            ca_norm_input = self.encoder_norm3_layers[i](x)
            ca_output, _ = self.encoder_cross_attn_layers[i](query=ca_norm_input, key=cond_kv, value=cond_kv)
            x = x + self.dropout_layers[i](ca_output)
            ffn_norm_input = self.encoder_norm2_layers[i](x)
            ffn_output = self.encoder_ffn_layers[i](ffn_norm_input)
            x = x + ffn_output
            current_input = x

        transformer_output = current_input
        output_logits = self.output_layer(transformer_output)
        return output_logits


class SymbolicRegressionDataset(Dataset):
    def __init__(self, data, x_means, x_stds, y_mean=0.0, y_std=1.0):
        self.data = data
        self.x_means, self.x_stds = x_means, x_stds
        self.y_mean, self.y_std = y_mean, y_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids = np.array(item['token_ids'], dtype=np.int64)
        if token_ids.size == 0:
            token_ids = np.array([PAD_TOKEN_ID], dtype=np.int64)
        if np.any(token_ids >= VOCAB_SIZE):
            token_ids = np.clip(token_ids, 0, VOCAB_SIZE - 1)

        xy_coords = np.array(item['X_Y_combined'], dtype=np.float32)
        xy_coords[:, :-1] = (xy_coords[:, :-1] - self.x_means) / (self.x_stds + 1e-8)
        xy_coords[:, -1] = (xy_coords[:, -1] - self.y_mean) / (self.y_std + 1e-8)

        condition_tensor = torch.from_numpy(xy_coords)

        return {
            'token_ids': torch.from_numpy(token_ids),
            'condition': condition_tensor
        }


def dynamic_pad_collate(batch):
    if not batch:
        raise ValueError("Received empty batch in collate_fn")

    token_tensors = []
    condition_tensors = []
    max_seq_len = 0

    for item in batch:
        token_ids = item['token_ids']
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)

        if token_ids.numel() == 0:
            token_ids = torch.tensor([PAD_TOKEN_ID], dtype=torch.long)

        token_tensors.append(token_ids)
        max_seq_len = max(max_seq_len, token_ids.shape[0])

        condition = item['condition']
        if not isinstance(condition, torch.Tensor):
            condition = torch.as_tensor(condition, dtype=torch.float32)
        condition_tensors.append(condition)

    padded_tokens = torch.full((len(batch), max_seq_len), PAD_TOKEN_ID, dtype=torch.long)
    for i, token_ids in enumerate(token_tensors):
        seq_len = token_ids.shape[0]
        padded_tokens[i, :seq_len] = token_ids

    conditions = torch.stack(condition_tensors, dim=0)

    return {
        'token_ids': padded_tokens,
        'condition': conditions,
    }


@torch.no_grad()
def evaluate(model, diffusion, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        x_start = batch['token_ids'].to(device)
        condition = batch['condition'].to(device) # Shape (B, N_POINTS, CONDITION_FEATURE_DIM)
        if x_start.max() >= VOCAB_SIZE or x_start.min() < 0:
             print(f"\nWarning: Invalid token ID in validation batch. Skipping.")
             continue
        loss = diffusion.compute_loss(model, x_start, condition, pad_token_id=PAD_TOKEN_ID)
        if not torch.isnan(loss):
             total_val_loss += loss.item()
             num_batches += 1
        else:
            print("\nWarning: NaN loss encountered during validation. Skipping batch.")
    model.train()
    if num_batches == 0:
        print("\nWarning: No valid batches processed during evaluation.")
        return float('inf')
    return total_val_loss / num_batches



def train(train_data, test_data, x_means=None, x_stds=None, y_mean=None, y_std=None):
    print(f"Using device: {DEVICE}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation (test) data size: {len(test_data)}")

    if not train_data:
        raise ValueError("train_data list is empty.")
    perform_validation = bool(test_data)
    if not perform_validation:
        print("Warning: test_data is empty. Skipping validation.")

    # If normalization stats not provided, compute from a sample
    if x_means is None or x_stds is None or y_mean is None or y_std is None:
        print("Calculating per-dimension normalization statistics from train_data sample...")
        all_coords_list = []
        sample_size = min(1000, len(train_data))
        indices = np.random.choice(len(train_data), size=sample_size, replace=False)
        
        for idx in indices:
            item = train_data[idx]
            if 'X_Y_combined' in item:
                all_coords_list.append(item['X_Y_combined'])
        
        if not all_coords_list:
            raise ValueError("train_data is empty, cannot calculate normalization stats.")
        all_coords_np = np.array(all_coords_list, dtype=np.float32)
        all_coords_np = np.nan_to_num(all_coords_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Separate X features (all columns except last) and Y feature (last column)
        all_x_features = all_coords_np[:, :, :-1] # Shape: (N_records, N_POINTS, D)
        all_y_features = all_coords_np[:, :, -1]  # Shape: (N_records, N_POINTS)

        # Calculate mean/std PER X DIMENSION (axis=(0, 1) averages over records and points)
        x_means = np.mean(all_x_features, axis=(0, 1)) # Shape: (D,)
        x_stds = np.std(all_x_features, axis=(0, 1))   # Shape: (D,)

        # Calculate mean/std for Y dimension (axis=(0, 1) averages over records and points)
        y_mean = float(np.mean(all_y_features)) # Scalar
        y_std = float(np.std(all_y_features))   # Scalar

        # Prevent division by zero if std is too small
        x_stds = np.where(x_stds > 1e-6, x_stds, 1.0)
        y_std = y_std if y_std > 1e-6 else 1.0

    print(f"Normalization Stats:")
    for d in range(len(x_means)):
         print(f"  X dim {d+1}: Mean={x_means[d]:.3f}, Std={x_stds[d]:.3f}")
    print(f"  Y feature: Mean={y_mean:.3f}, Std={y_std:.3f}")

    # Create Datasets (Pass the calculated stats - now x_means/x_stds are arrays)
    train_dataset = SymbolicRegressionDataset(train_data, x_means, x_stds, y_mean, y_std)
    val_loader = None
    if perform_validation:
        val_dataset = SymbolicRegressionDataset(test_data, x_means, x_stds, y_mean, y_std)
        val_loader = DataLoader(
            val_dataset,
            batch_size=VALIDATION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if DEVICE == "cuda" else False,
            collate_fn=dynamic_pad_collate,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False,
        collate_fn=dynamic_pad_collate,
    )

    model = ConditionalD3PMTransformer(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, seq_len=SEQ_LEN,
        condition_feature_dim=CONDITION_FEATURE_DIM, # <<< Pass the correct dimension
        num_timesteps=NUM_TIMESTEPS, dropout=DROPOUT
    ).to(DEVICE)

    diffusion = DiscreteDiffusion(num_timesteps=NUM_TIMESTEPS, vocab_size=VOCAB_SIZE, device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if perform_validation:
        print(f"Early stopping patience: {PATIENCE}")
        print(f"Best model will be saved to: {BEST_MODEL_PATH}")

    best_val_loss = float('inf'); epochs_no_improve = 0
    epochs_plotted = []; train_losses = []; val_losses = []

    # --- Training Loop ---
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()
        processed_batches = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x_start = batch['token_ids'].to(DEVICE)
            condition = batch['condition'].to(DEVICE) # Shape (B, N_POINTS, CONDITION_FEATURE_DIM)

            if x_start.max() >= VOCAB_SIZE or x_start.min() < 0:
                 print(f"\nWarning: Invalid token ID in train batch. Skipping.")
                 continue

            loss = diffusion.compute_loss(model, x_start, condition, pad_token_id=PAD_TOKEN_ID)

            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected during training. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            processed_batches += 1

            print(f"\rEpoch [{epoch+1}/{EPOCHS}], Current LR: {optimizer.param_groups[0]['lr']:.6e}, Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}   ", end="")
            sys.stdout.flush()

        # --- End of Epoch ---
        avg_train_loss = total_train_loss / processed_batches if processed_batches > 0 else 0
        epoch_time = time.time() - start_time

        # Validation Step
        avg_val_loss = float('inf')
        if perform_validation and val_loader:
            print(f"\nEpoch [{epoch+1}/{EPOCHS}] completed in {epoch_time:.2f}s. Avg Train Loss: {avg_train_loss:.4f}. Current LR: {optimizer.param_groups[0]['lr']:.6e}. Evaluating...", end="")
            sys.stdout.flush()
            avg_val_loss = evaluate(model, diffusion, val_loader, DEVICE)
            print(f" Avg Val Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)
        else:
             print(f"\nEpoch [{epoch+1}/{EPOCHS}] completed in {epoch_time:.2f}s. Avg Train Loss: {avg_train_loss:.4f}. (Validation Skipped)")

        # Store losses for plotting
        epochs_plotted.append(epoch + 1)
        train_losses.append(avg_train_loss)
        if perform_validation:
             val_losses.append(avg_val_loss)

        # Live Plotting
        try:
            display.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_plotted, train_losses, 'bo-', label='Training Loss')
            if val_losses:
                ax.plot(epochs_plotted, val_losses, 'ro-', label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss Over Epochs')
            ax.grid(True)
            ax.legend()
            # Adjust y-scale dynamically
            min_loss = min(min(train_losses, default=1.0), min(val_losses, default=1.0))
            max_loss = max(max(train_losses, default=0.0), max(val_losses, default=0.0))
            if max_loss > 5 * min_loss and min_loss > 0: # Use log scale if large range
                 ax.set_yscale('log')
            display.display(fig)
            plt.close(fig)
        except Exception as e:
            print(f"\nError during plotting: {e}")

        # Early Stopping Check
        if perform_validation:
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve from {best_val_loss:.4f}. Patience: {epochs_no_improve}/{PATIENCE}")
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    # End of Training Loop
    print("Training finished.")
    if perform_validation and os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model weights from {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    elif perform_validation:
        print("Warning: Best model path not found, but validation was performed. Returning model from last epoch.")
    else:
        print("Validation was not performed. Returning model from last epoch.")
    return model