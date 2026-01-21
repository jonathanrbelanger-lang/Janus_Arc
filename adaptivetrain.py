import os
import sys
import math
import gc
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset

# --- 1. CONFIGURATION & ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Janus Adaptive Scheduler SAE")
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints/janus_adaptive_v1", help="Directory to save checkpoints and logs")
    
    # Model Config
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    
    # Training Config
    parser.add_argument("--max_steps", type=int, default=4005)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6.29e-4)
    
    # Adaptive Controller Config
    parser.add_argument("--target_sigma", type=float, default=0.0035)
    parser.add_argument("--k_p", type=float, default=0.5)
    parser.add_argument("--control_interval", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=1500)
    
    # IO Config
    parser.add_argument("--ckpt_interval", type=int, default=2000)
    parser.add_argument("--lite_interval", type=int, default=100)
    
    return parser.parse_args()

# --- 2. DATA LOADER (Hugging Face) ---
class HFWikiTextLoader:
    """
    Replaces the original ChunkLoader. 
    Downloads WikiText-103, tokenizes it into a single flat tensor in RAM for fast random sampling.
    """
    def __init__(self, config, split="train"):
        print(f"📥 Loading {split} data from Hugging Face (wikitext-103-raw-v1)...")
        self.config = config
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Suppress HF logging for cleaner output
        import logging
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        # Tokenize entire dataset
        # We join with EOS token to utilize all text, similar to standard LM training
        text_blob = "\n\n".join(dataset["text"])
        tokens = self.tokenizer.encode(text_blob)
        
        # Convert to numpy uint16 to save RAM (GPT2 vocab < 65535)
        self.data = np.array(tokens, dtype=np.uint16)
        print(f"✅ {split.capitalize()} data loaded. {len(self.data):,} tokens.")
        
    def get_batch(self, batch_size, device):
        # Random sampling (same behavior as original ChunkLoader)
        high = len(self.data) - self.config.max_seq_len
        ix = np.random.randint(0, high, size=batch_size)
        
        x_batch = []
        y_batch = []
        
        for i in ix:
            # Grab slice
            chunk = self.data[i : i + self.config.max_seq_len + 1].astype(np.int64)
            x_batch.append(torch.tensor(chunk[:-1], dtype=torch.long))
            y_batch.append(torch.tensor(chunk[1:], dtype=torch.long))
            
        x = torch.stack(x_batch).to(device)
        y = torch.stack(y_batch).to(device)
        return x, y

# --- 3. HOMEOSTATIC CONTROLLER ---
class Homeostat:
    def __init__(self, config):
        self.target = config.target_sigma
        self.kp = config.k_p
        self.warmup = config.warmup_steps
        self.current_lambda = 0.0
        
    def update(self, step, current_sigma):
        # Phase 1: Ignition (Open Loop Ramp)
        if step < self.warmup:
            # Ramp from 0.00 to 0.05
            self.current_lambda = 0.05 * (step / self.warmup)
            return self.current_lambda

        # Phase 2: Homeostasis (Closed Loop P-Control)
        if current_sigma is None: return self.current_lambda

        error = current_sigma - self.target
        # CORRECTION: Positive Error (Too Redundant) -> INCREASE Pressure
        delta = self.kp * error

        # Apply & Clamp
        self.current_lambda += delta
        # Hard limits to prevent explosion or negative pressure
        self.current_lambda = max(0.0, min(0.25, self.current_lambda))

        return self.current_lambda

# --- 4. MODEL ARCHITECTURE ---
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

        self.dropout = nn.Dropout(config.dropout)
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

    def forward(self, x, lambdas, return_metrics=False):
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
        attn_probs = self.dropout(attn_probs)
        head_out = attn_probs @ v

        steer_loss = 0.0
        l_coh, l_div = lambdas

        if (l_div > 0.0 and self.training) or return_metrics:
            flat = head_out.transpose(0, 1).reshape(self.n_heads, -1)
            norm = F.normalize(flat, p=2, dim=1)
            gram = torch.mm(norm, norm.t())
            identity = torch.eye(self.n_heads, device=x.device)

            if l_div > 0.0:
                steer_loss += torch.norm(gram - identity, p='fro') * l_div

        metrics = {}
        if return_metrics:
            with torch.no_grad():
                flat = head_out.transpose(0, 1).reshape(self.n_heads, -1)
                norm = F.normalize(flat, p=2, dim=1)
                sim = torch.mm(norm, norm.t())
                mask_diag = ~torch.eye(self.n_heads, dtype=torch.bool, device=x.device)
                metrics['sigma_a'] = (sim.abs() * mask_diag.float()).sum(dim=1) / (self.n_heads - 1)

        out = head_out.transpose(1, 2).reshape(B, S, D)
        return self.o_proj(out), steer_loss, metrics

class NewBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = NewAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model)
    def forward(self, x, lambdas, return_metrics=False):
        a, s, m = self.attn(self.ln1(x), lambdas, return_metrics)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, s, m

class NewGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([NewBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for name, p in module.named_parameters():
            if "o_proj.weight" in name or "w3.weight" in name:
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(self, idx, lambdas_list, targets=None, return_metrics=False):
        B, S = idx.shape
        x = self.token_emb(idx)
        total_steer = 0.0
        all_metrics = []
        for i, block in enumerate(self.blocks):
            x, s, m = block(x, lambdas_list[i], return_metrics)
            total_steer += s
            if return_metrics: all_metrics.append(m)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, total_steer, all_metrics

# --- 5. LOGGING ---
class BlackBox:
    def __init__(self, save_dir):
        self.buffer = []
        self.save_dir = save_dir
        self.start_step = 0
        
    def set_start_step(self, step):
        self.start_step = step
        
    def log(self, step, loss, val_loss, pressure, metrics):
        if step < self.start_step: return

        # Calculate mean sigma_a across layers
        avg_sigma = 0.0
        if metrics:
            sigmas = [m['sigma_a'].mean().item() for m in metrics if 'sigma_a' in m]
            if sigmas: avg_sigma = sum(sigmas) / len(sigmas)

        row = {
            "step": step, "loss": loss, "val_loss": val_loss,
            "perplexity": math.exp(val_loss) if val_loss < 20 else 0.0,
            "pressure": pressure,
            "sigma_a_avg": avg_sigma
        }
        self.buffer.append(row)
        
    def flush(self):
        if not self.buffer: return
        df = pd.DataFrame(self.buffer)
        fpath = os.path.join(self.save_dir, "telemetry_adaptive.parquet")
        if os.path.exists(fpath):
            try:
                existing = pd.read_parquet(fpath)
                df = pd.concat([existing, df])
            except: pass
        df.to_parquet(fpath)
        self.buffer = []
        print("💾 Telemetry Flushed.")

# --- 6. MAIN RUN LOOP ---
def main():
    # 1. Setup
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Memory Nuke
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print(f"🧠 JANUS ADAPTIVE (Homeostatic Control)")
    print(f"⚙️ Hardware: {DEVICE}")
    print(f"📂 Save Directory: {args.save_dir}")

    # 3. Initialize Components
    model = NewGPT(args).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.34e-4)
    controller = Homeostat(args)
    recorder = BlackBox(args.save_dir)
    
    # 4. Data Loaders
    train_loader = HFWikiTextLoader(args, split="train")
    val_loader = HFWikiTextLoader(args, split="validation")

    # 5. Resume Logic
    start_step = 0
    ckpts = [f for f in os.listdir(args.save_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest = ckpts[-1]
        ckpt_path = os.path.join(args.save_dir, latest)
        print(f"🔄 Resuming Adaptive Run from {latest}...")
        c = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optim'])
        controller.current_lambda = c.get('current_lambda', 0.0)
        start_step = c['step'] + 1
        recorder.set_start_step(start_step)

    print(f"\n🧠 STARTING ADAPTIVE RUN: {start_step} -> {args.max_steps}")
    print(f"   Target Sigma: {args.target_sigma}")
    print(f"   Warmup Steps: {args.warmup_steps}")

    # 6. Training Loop
    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps)
    current_sigma_avg = None

    for step in pbar:
        # A. Control Update
        if step % args.control_interval == 0 or step < args.warmup_steps:
             new_lambda = controller.update(step, current_sigma_avg)

        # Apply lambda to layers
        p = controller.current_lambda
        base_coh = p * 0.2
        lambdas_list = []
        for i in range(args.n_layers):
            ratio = (i + 1) / args.n_layers
            s_mult = ratio ** 3
            lambdas_list.append((base_coh * s_mult, p * s_mult))

        # B. Training Step
        model.train()
        batch_loss = 0.0
        optimizer.zero_grad()
        accum_sigmas = []

        for _ in range(args.grad_accum):
            x, y = train_loader.get_batch(args.batch_size, DEVICE)

            is_control_step = (step % args.control_interval == 0)
            is_log_step = (step % args.lite_interval == 0)
            return_metrics = is_control_step or is_log_step

            # Mixed Precision Context
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                loss, steer, metrics = model(x, lambdas_list, y, return_metrics=return_metrics)
                total = (loss + steer) / args.grad_accum

            total.backward()
            batch_loss += loss.item()

            if return_metrics and metrics:
                s = [m['sigma_a'].mean().item() for m in metrics]
                if s: accum_sigmas.append(sum(s)/len(s))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if accum_sigmas:
            current_sigma_avg = sum(accum_sigmas) / len(accum_sigmas)

        # C. Telemetry
        if step > 0 and (step % args.lite_interval == 0 or step == args.max_steps - 1):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                v_losses = []
                # Validate on 10 random batches
                for _ in range(10): 
                    vx, vy = val_loader.get_batch(args.batch_size, DEVICE)
                    vl, _, _ = model(vx, [(0.0,0.0)]*args.n_layers, vy, return_metrics=False)
                    v_losses.append(vl.item())
                val_loss = np.mean(v_losses)

            recorder.log(step, batch_loss/args.grad_accum, val_loss, controller.current_lambda, metrics)
            recorder.flush()

            pbar.set_description(f"L:{batch_loss/args.grad_accum:.2f}|V:{val_loss:.2f}|P:{controller.current_lambda:.3f}|Sig:{current_sigma_avg:.4f}")
        else:
            pbar.set_description(f"L:{batch_loss/args.grad_accum:.2f}|P:{controller.current_lambda:.3f}|Sig:{current_sigma_avg if current_sigma_avg else 0:.4f}")

        # D. Checkpoints
        if step > 0 and step % args.ckpt_interval == 0:
            ckpt_name = f"ckpt_step_{step}.pt"
            ckpt_path = os.path.join(args.save_dir, ckpt_name)
            save_dict = {
                'step': step,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'current_lambda': controller.current_lambda,
                'rng_state': torch.get_rng_state()
            }
            torch.save(save_dict, ckpt_path)
            
            # Keep only last 3 checkpoints
            all_ckpts = [f for f in os.listdir(args.save_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
            all_ckpts.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            while len(all_ckpts) > 3:
                os.remove(os.path.join(args.save_dir, all_ckpts.pop(0)))

    # Final Save
    final_path = os.path.join(args.save_dir, "janus_adaptive_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n🏆 ADAPTIVE RUN COMPLETE. Saved to {final_path}")

if __name__ == "__main__":
    main()
