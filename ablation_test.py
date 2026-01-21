import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset

# --- 1. CONFIGURATION & ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Head Ablation Test")
    
    # Paths
    parser.add_argument("--control_model", type=str, default="./checkpoints/janus_control_SAE.pt", help="Path to Control model")
    parser.add_argument("--adaptive_model", type=str, default="./checkpoints/janus_adaptive_v1/janus_adaptive_final.pt", help="Path to Adaptive model")
    parser.add_argument("--save_file", type=str, default="ablation_results.csv", help="Where to save the CSV results")
    
    # Test Settings
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 6, 9], help="Layers to ablate")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads per layer")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_batches", type=int, default=50, help="Number of batches to avg loss over")
    parser.add_argument("--ctx_len", type=int, default=512)
    
    return parser.parse_args()

# --- 2. DATA LOADER (Hugging Face) ---
class HFWikiTextLoader:
    def __init__(self, seq_len, split="validation"):
        print(f"📥 Loading {split} data for ablation baseline...")
        self.seq_len = seq_len
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        import logging
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        text_blob = "\n\n".join(dataset["text"])
        tokens = self.tokenizer.encode(text_blob)
        self.data = np.array(tokens, dtype=np.uint16)
        print(f"✅ Data loaded. {len(self.data):,} tokens.")

    def get_eval_batch(self, batch_size, device, batch_idx):
        # Deterministic slicing for fair comparison
        stride = batch_size * self.seq_len
        start = (batch_idx * stride) % (len(self.data) - stride - 1)
        end = start + stride
        
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk).view(batch_size, self.seq_len).to(device)
        y = torch.from_numpy(self.data[start+1:end+1].astype(np.int64)).view(batch_size, self.seq_len).to(device)
        return x, y

# --- 3. MODEL ARCHITECTURE (With Masking Support) ---
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
        
        # ABLATION MASK: [1, n_heads, 1, 1]
        self.head_mask = None

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

    def forward(self, x):
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

        # --- ABLATION APPLIED HERE ---
        if self.head_mask is not None:
            # Broadcast mask [1, H, 1, 1] over [B, H, S, S]
            attn_probs = attn_probs * self.head_mask

        head_out = attn_probs @ v
        out = head_out.transpose(1, 2).reshape(B, S, D)
        return self.o_proj(out)

class NewBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = NewAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model)
    def forward(self, x):
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

    def forward(self, idx, targets=None):
        B, S = idx.shape
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

class ModelConfig:
    def __init__(self):
        self.vocab_size = 50257; self.d_model = 512; self.n_layers = 12
        self.n_heads = 8; self.d_head = 64; self.max_seq_len = 512; self.dropout = 0.0

# --- 4. ABLATION LOGIC ---
def get_validation_loss(model, loader, n_batches, device, batch_size):
    """Calculates mean loss over fixed N batches."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i in range(n_batches):
            x, y = loader.get_eval_batch(batch_size, device, i)
            loss = model(x, y)
            losses.append(loss.item())
            
    return np.mean(losses)

def run_ablation_suite(run_name, model_path, loader, args, device):
    print(f"\n🔪 Starting Ablation for: {run_name.upper()}")
    
    # 1. Load Model
    cfg = ModelConfig()
    model = NewGPT(cfg).to(device)
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}, skipping.")
        return []

    try:
        sd = torch.load(model_path, map_location=device)
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"❌ Error loading {run_name}: {e}")
        return []

    # 2. Baseline Loss
    base_loss = get_validation_loss(model, loader, args.n_batches, device, args.batch_size)
    print(f"   Baseline Loss: {base_loss:.4f}")
    
    results = []
    
    # 3. Iterate Layers & Heads
    for layer_idx in args.layers:
        print(f"   Testing Layer {layer_idx}...", end=" ")
        
        for head_idx in range(args.n_heads):
            # A. Create Mask: [1, n_heads, 1, 1]
            mask = torch.ones(1, args.n_heads, 1, 1).to(device)
            mask[0, head_idx, 0, 0] = 0.0 # Kill this head
            
            # B. Apply Mask
            model.blocks[layer_idx].attn.head_mask = mask
            
            # C. Measure Loss
            loss = get_validation_loss(model, loader, args.n_batches, device, args.batch_size)
            
            # D. Record
            delta = loss - base_loss
            results.append({
                "run": run_name,
                "layer": layer_idx,
                "head": head_idx,
                "base_loss": base_loss,
                "ablated_loss": loss,
                "impact": delta, 
                "pct_impact": (delta / base_loss) * 100
            })
            
            # E. Reset
            model.blocks[layer_idx].attn.head_mask = None
            
        print("Done.")

    return results

# --- 5. MAIN ---
def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    loader = HFWikiTextLoader(args.ctx_len)
    
    all_results = []
    
    # Run Control
    res_ctl = run_ablation_suite("control", args.control_model, loader, args, DEVICE)
    all_results.extend(res_ctl)
    
    # Run Adaptive
    res_adt = run_ablation_suite("adaptive", args.adaptive_model, loader, args, DEVICE)
    all_results.extend(res_adt)
    
    if not all_results:
        print("❌ No results collected.")
        return

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(args.save_file, index=False)
    print(f"\n✅ Ablation Complete. Saved to {args.save_file}")
    
    # Summary
    print("\n--- Impact Summary (Mean % Increase in Loss) ---")
    print(df.groupby(["run", "layer"])["pct_impact"].mean())
    
    print("\n--- Top 3 Most Critical Heads ---")
    print(df.sort_values("pct_impact", ascending=False).head(3)[["run", "layer", "head", "pct_impact"]])

if __name__ == "__main__":
    main()