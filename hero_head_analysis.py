import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer

# --- 1. CONFIGURATION & ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Hero Head Analysis: Attention & SAE Features")
    
    # Input/Output
    parser.add_argument("--ablation_csv", type=str, default="ablation_results.csv", help="Results from ablation test to find Hero Head")
    parser.add_argument("--save_dir", type=str, default="./hero_analysis", help="Where to save plots")
    
    # Model Paths
    parser.add_argument("--control_model", type=str, default="./checkpoints/janus_control_SAE.pt")
    parser.add_argument("--adaptive_model", type=str, default="./checkpoints/janus_adaptive_v1/janus_adaptive_final.pt")
    parser.add_argument("--sae_dir", type=str, default="./data/sae_models", help="Directory containing trained SAEs")
    
    # Analysis Settings
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog because it relies on complex physics.", help="Input text to analyze")
    parser.add_argument("--force_layer", type=int, default=None, help="Override Hero detection: Force Layer X")
    parser.add_argument("--force_head", type=int, default=None, help="Override Hero detection: Force Head Y")
    
    # Model Params (Must match training)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=50257)
    
    # SAE Params
    parser.add_argument("--expansion_factor", type=int, default=32, help="Must match training config (default: 32)")
    
    return parser.parse_args()

# --- 2. MODEL ARCHITECTURE (With Inspection Hooks) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = int(d_model * 8 / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class NewAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.register_buffer("freqs_cis", self.precompute_freqs_cis(config.d_head, config.max_seq_len))

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def apply_rope(self, x, freqs_cis):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = freqs_cis[:x.shape[1]].view(1, x.shape[1], 1, -1)
        x_out = torch.view_as_real(x_c * freqs).flatten(3)
        return x_out.type_as(x)

    def forward(self, x, return_internals=False):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head)

        q = self.apply_rope(q, self.freqs_cis)
        k = self.apply_rope(k, self.freqs_cis)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn, dim=-1)

        # Output of attention heads before mixing (Project Out)
        # Shape: [B, Heads, Seq, Head_Dim]
        head_out = attn_probs @ v
        
        # Combine
        out = head_out.transpose(1, 2).reshape(B, S, D)
        
        if return_internals:
            return self.o_proj(out), attn_probs, head_out
            
        return self.o_proj(out)

class NewBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = NewAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model)
        
    def forward(self, x, return_internals=False):
        if return_internals:
            out_attn, probs, head_out = self.attn(self.ln1(x), return_internals=True)
            x = x + out_attn
            x = x + self.mlp(self.ln2(x))
            return x, probs, head_out
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

class NewGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([NewBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight

    def forward(self, idx, target_layer=None):
        # We allow capturing internals of a specific layer
        x = self.token_emb(idx)
        captured_probs = None
        captured_acts = None
        
        for i, block in enumerate(self.blocks):
            if target_layer is not None and i == target_layer:
                x, probs, acts = block(x, return_internals=True)
                captured_probs = probs
                captured_acts = acts
            else:
                x = block(x)
                
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, captured_probs, captured_acts

class ModelConfig:
    def __init__(self, args):
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.d_head = args.d_head
        self.max_seq_len = 512
        self.dropout = 0.0

# --- 3. SAE ARCHITECTURE ---
class SAEConfig:
    def __init__(self, input_dim, expansion_factor):
        self.input_dim = input_dim
        self.d_sae = input_dim * expansion_factor

class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        # Encoder
        self.W_enc = nn.Parameter(torch.empty(cfg.input_dim, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        # Decoder
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.input_dim))
        self.b_dec = nn.Parameter(torch.zeros(cfg.input_dim))

    def forward(self, x):
        # Pre-bias
        x_cent = x - self.b_dec
        # Encode
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        # Decode
        x_rec = acts @ self.W_dec + self.b_dec
        return x_rec, acts

# --- 4. ANALYSIS LOGIC ---

def find_hero_head(csv_path, force_layer, force_head):
    if force_layer is not None and force_head is not None:
        return force_layer, force_head
        
    if not os.path.exists(csv_path):
        print(f"⚠️ Ablation CSV not found at {csv_path}. Defaulting to L9 H0.")
        return 9, 0
        
    df = pd.read_csv(csv_path)
    # Filter for adaptive run if available
    if "run" in df.columns:
        df = df[df["run"] == "adaptive"]
        
    if df.empty:
        print("⚠️ CSV empty or format mismatch. Defaulting to L9 H0.")
        return 9, 0
        
    # Find row with max 'pct_impact'
    hero = df.loc[df["pct_impact"].idxmax()]
    layer = int(hero["layer"])
    head = int(hero["head"])
    print(f"🏆 Detected Hero Head: Layer {layer}, Head {head} (Impact: {hero['pct_impact']:.2f}%)")
    return layer, head

def plot_attention_heatmap(run_name, attn_matrix, tokens, layer, head, save_dir):
    """
    Plots [Seq, Seq] attention matrix.
    """
    # Detach and move to CPU
    data = attn_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap="viridis", 
                xticklabels=tokens, yticklabels=tokens, 
                square=True, cbar=True)
    
    title = f"{run_name.upper()} Attention (L{layer} H{head})"
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    
    fname = os.path.join(save_dir, f"attn_{run_name}_L{layer}_H{head}.png")
    plt.savefig(fname)
    print(f"   📊 Heatmap saved to {fname}")
    plt.close()

def interpret_sae_features(run_name, sae_path, head_acts, tokens, expansion_factor):
    """
    Runs activations through SAE and prints top features.
    head_acts shape: [1, Seq, d_head]
    """
    if not os.path.exists(sae_path):
        print(f"   ⚠️ SAE checkpoint not found: {sae_path}")
        return

    device = head_acts.device
    d_head = head_acts.shape[-1]
    
    # Init SAE
    sae_cfg = SAEConfig(d_head, expansion_factor)
    sae = SparseAutoencoder(sae_cfg).to(device)
    
    # Load Weights
    # Handle potentially raw state dict or wrapped
    sd = torch.load(sae_path, map_location=device)
    sae.load_state_dict(sd)
    sae.eval()
    
    # Forward Pass
    # Flatten: [Seq, 64]
    flat_acts = head_acts.squeeze(0) 
    _, feature_acts = sae(flat_acts) # [Seq, d_sae]
    
    # Find most active features across the whole sequence
    # Max activation per feature
    max_per_feat, _ = feature_acts.max(dim=0)
    top_values, top_indices = max_per_feat.topk(5)
    
    print(f"\n   🧠 {run_name.upper()} SAE - Top 5 Active Features:")
    
    for i, feat_idx in enumerate(top_indices):
        feat_id = feat_idx.item()
        score = top_values[i].item()
        
        if score < 0.1: continue # Ignore dead/noise features
        
        # Which tokens triggered this feature?
        # Get column for this feature: [Seq]
        feat_col = feature_acts[:, feat_id]
        
        # Get top 3 tokens for this specific feature
        tok_vals, tok_inds = feat_col.topk(3)
        
        triggers = []
        for val, idx in zip(tok_vals, tok_inds):
            if val.item() > 0.1:
                t_str = tokens[idx.item()].replace('\n', '\\n')
                triggers.append(f"'{t_str}'({val:.1f})")
                
        print(f"      Feature #{feat_id:04d} (Max: {score:.1f}) triggered by: {', '.join(triggers)}")

# --- 5. MAIN ---
def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"🚀 Hero Head Analysis")
    print(f"   Input: \"{args.text}\"")
    
    # 1. Determine Hero Head
    LAYER, HEAD = find_hero_head(args.ablation_csv, args.force_layer, args.force_head)
    
    # 2. Tokenize
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer(args.text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]
    
    # 3. Analyze Both Models
    runs = [
        ("control", args.control_model),
        ("adaptive", args.adaptive_model)
    ]
    
    cfg = ModelConfig(args)
    
    for run_name, model_path in runs:
        print(f"\n==========================================")
        print(f"🔍 Analyzing {run_name.upper()} Model")
        
        if not os.path.exists(model_path):
            print(f"   ❌ Model not found: {model_path}")
            continue
            
        # Load Model
        model = NewGPT(cfg).to(DEVICE)
        sd = torch.load(model_path, map_location=DEVICE)
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        
        # Forward Pass with capture
        with torch.no_grad():
            _, attn_probs, head_acts = model(input_ids, target_layer=LAYER)
            
        # A. Attention Heatmap
        # attn_probs: [Batch, Heads, Seq, Seq] -> Slice [Seq, Seq]
        my_attn = attn_probs[0, HEAD, :, :]
        plot_attention_heatmap(run_name, my_attn, tokens, LAYER, HEAD, args.save_dir)
        
        # B. SAE Interpretation
        # head_acts: [Batch, Heads, Seq, Dim] -> Slice [Batch, Seq, Dim]
        # Note: NewAttention returns [B, Heads, S, D], so we slice dim 1
        my_acts = head_acts[:, HEAD, :, :] 
        
        sae_file = f"sae_{run_name}_L{LAYER}_H{HEAD}.pt"
        sae_path = os.path.join(args.sae_dir, sae_file)
        
        interpret_sae_features(run_name, sae_path, my_acts, tokens, args.expansion_factor)

    print(f"\n✅ Analysis Complete. Check {args.save_dir} for plots.")

if __name__ == "__main__":
    main()