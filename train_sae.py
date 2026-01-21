import os
import sys
import glob
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION & ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train SAEs on Harvested Activations")
    
    # Paths
    parser.add_argument("--act_dir", type=str, default="./data/sae_activations", help="Directory containing .pt activation files")
    parser.add_argument("--save_dir", type=str, default="./data/sae_models", help="Directory to save trained SAE models")
    
    # Run Settings
    parser.add_argument("--runs", nargs='+', default=["control", "adaptive"], help="List of run names to process")
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 6, 9], help="List of layers to train on")
    
    # Model Structure
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    
    # SAE Hyperparameters
    parser.add_argument("--expansion_factor", type=int, default=32, help="Ratio of hidden features to input dim")
    parser.add_argument("--l1_coeff", type=float, default=3e-4, help="Sparsity penalty coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5000, help="Training steps per head")
    
    return parser.parse_args()

# --- 2. SAE ARCHITECTURE ---
class SAEConfig:
    def __init__(self, input_dim, expansion_factor=32, l1_coeff=3e-4):
        self.input_dim = input_dim
        self.d_sae = input_dim * expansion_factor
        self.l1_coeff = l1_coeff

class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Encoder: W_enc * (x - b_dec) + b_enc
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.input_dim, cfg.d_sae)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        
        # Decoder: W_dec * f + b_dec
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.input_dim)))
        self.b_dec = nn.Parameter(torch.zeros(cfg.input_dim))
        
        # Normalize decoder columns to unit norm (standard SAE practice)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

    def forward(self, x):
        # 1. Pre-bias subtraction
        x_cent = x - self.b_dec
        
        # 2. Encode
        # ReLU( (x-b_dec) @ W_enc + b_enc )
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        # 3. Decode
        # acts @ W_dec + b_dec
        x_reconstruct = acts @ self.W_dec + self.b_dec
        
        return x_reconstruct, acts

    @torch.no_grad()
    def make_decoder_weights_normalized(self):
        # Enforce unit norm on decoder weights periodically
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

# --- 3. TRAINING LOGIC ---
def get_batch(data, batch_size, device):
    """Random sampling from loaded tensor."""
    ix = torch.randint(0, data.shape[0], (batch_size,))
    return data[ix].to(device)

def train_sae_on_head(data_tensor, cfg, args, device="cuda"):
    """
    Trains a single SAE on the provided data tensor [N_samples, d_head].
    """
    input_dim = data_tensor.shape[1]
    sae_cfg = SAEConfig(input_dim, args.expansion_factor, args.l1_coeff)
    sae = SparseAutoencoder(sae_cfg).to(device)
    
    optimizer = optim.Adam(sae.parameters(), lr=args.lr)
    
    # Simple linear warmup, cosine decay schedule could be added here
    # For SAEs, constant LR or simple decay is often sufficient for small runs
    
    pbar = tqdm(range(args.steps), leave=False, desc="Training SAE")
    
    final_mse = 0.0
    final_l0 = 0.0
    
    for step in pbar:
        sae.train()
        batch = get_batch(data_tensor, args.batch_size, device)
        
        # Forward
        x_hat, acts = sae(batch)
        
        # Losses
        # 1. Reconstruction (MSE)
        loss_mse = (x_hat - batch).pow(2).sum(dim=-1).mean()
        
        # 2. Sparsity (L1)
        loss_l1 = acts.sum(dim=-1).mean() * args.l1_coeff
        
        total_loss = loss_mse + loss_l1
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Normalize decoder gradients (prevent blowing up) - optional but recommended
        # sae.make_decoder_weights_normalized() # Can be done post-step or via hook
        
        optimizer.step()
        
        # Enforce unit norm constraint
        sae.make_decoder_weights_normalized()
        
        # Metrics
        with torch.no_grad():
            l0 = (acts > 0).float().sum(dim=-1).mean().item()
            final_mse = loss_mse.item()
            final_l0 = l0
            
        if step % 100 == 0:
            pbar.set_postfix({"MSE": f"{final_mse:.4f}", "L0": f"{final_l0:.1f}"})

    return sae, final_mse, final_l0

# --- 4. DATA LOADING ---
def load_layer_activations(run_name, layer_idx, act_dir, n_heads, d_head):
    """
    Loads activation chunks, concatenates them, and reshapes for heads.
    Output: [Total_Tokens, n_heads, d_head]
    """
    # Matches format: "run_name_L3_0001.pt"
    pattern = os.path.join(act_dir, f"{run_name}_L{layer_idx}_*.pt")
    files = sorted(glob.glob(pattern))

    if not files:
        # Fallback to verify if maybe named differently (e.g. from original script)
        pattern = os.path.join(act_dir, f"{run_name}_*_L{layer_idx}_*.pt")
        files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"⚠️ No activation files found for {run_name} Layer {layer_idx}")
        return None

    print(f"   📂 Loading {len(files)} files for {run_name} Layer {layer_idx}...")
    
    tensors = []
    for f in files:
        # Load to CPU
        t = torch.load(f, map_location="cpu")
        tensors.append(t)
        
    full_data = torch.cat(tensors, dim=0) # [Tokens, 512]
    
    # Reshape to heads [Tokens, 8, 64]
    # NOTE: This assumes the activations were saved as flattened d_model=512
    # and that the model layout is contiguous.
    try:
        full_data = full_data.view(-1, n_heads, d_head)
    except Exception as e:
        print(f"❌ Error reshaping data: {e}. Expected shape [N, {n_heads*d_head}]")
        return None
        
    return full_data

# --- 5. MAIN EXECUTION ---
def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = []

    print(f"🚀 SAE TRAINER")
    print(f"   Runs: {args.runs}")
    print(f"   Layers: {args.layers}")
    print(f"   Device: {DEVICE}")

    for run_name in args.runs:
        print(f"\n==========================================")
        print(f"🧠 RUN: {run_name.upper()}")
        print(f"==========================================")

        for layer_idx in args.layers:
            # 1. Load Data
            full_data = load_layer_activations(
                run_name, layer_idx, args.act_dir, args.n_heads, args.d_head
            )
            
            if full_data is None: continue
            
            print(f"   ✅ Data Loaded: {full_data.shape} (Tokens, Heads, Dim)")

            # 2. Train per Head
            for head_idx in range(args.n_heads):
                # Slice specific head [N, 64]
                # .clone() is important to ensure memory is contiguous for rapid GPU transfer
                head_acts = full_data[:, head_idx, :].clone()
                
                # Normalize Input (Optional but often helpful for SAE stability)
                # Centers the sphere of activations
                # head_acts = head_acts - head_acts.mean(dim=0, keepdim=True)

                print(f"      ▶️ Training Head {head_idx}...", end="")
                
                sae, mse, l0 = train_sae_on_head(head_acts, None, args, device=DEVICE)
                
                # 3. Save Model
                fname = f"sae_{run_name}_L{layer_idx}_H{head_idx}.pt"
                save_path = os.path.join(args.save_dir, fname)
                torch.save(sae.state_dict(), save_path)
                
                results.append({
                    "run": run_name,
                    "layer": layer_idx,
                    "head": head_idx,
                    "mse": mse,
                    "l0": l0,
                    "path": save_path
                })
                
                print(f" Done. MSE={mse:.4f}, L0={l0:.1f}")
                
            # Free memory
            del full_data
            del head_acts
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            import gc; gc.collect()

    # Save Registry
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.save_dir, "sae_training_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Training Complete. Registry saved to {csv_path}")
        print("\nSummary (Mean L0 per Layer):")
        print(df.groupby(["run", "layer"])["l0"].mean())
    else:
        print("\n❌ No models trained. Check data paths.")

if __name__ == "__main__":
    main()