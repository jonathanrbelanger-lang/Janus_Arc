import os
import sys
import glob
import math
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset

# --- 1. CONFIGURATION & ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Janus Activation Harvester")
    
    # I/O
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--save_dir", type=str, default="./data/sae_activations", help="Where to save .pt activation files")
    parser.add_argument("--run_name", type=str, default="run", help="Prefix for saved files (e.g., 'control', 'adaptive')")
    
    # Harvesting Params
    parser.add_argument("--tokens_to_harvest", type=int, default=10_000_000, help="Total tokens to process")
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 6, 9], help="List of layer indices to hook (e.g. 3 6 9)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ctx_len", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=100, help="Number of batches to buffer in RAM before flushing to disk")

    # Model Params (Must match training config)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    
    return parser.parse_args()

# --- 2. DATA LOADER (Hugging Face) ---
class HFWikiTextLoader:
    """
    Downloads WikiText-103, tokenizes it, and serves batches.
    """
    def __init__(self, seq_len, split="train"):
        print(f"📥 Loading {split} data from Hugging Face...")
        self.seq_len = seq_len
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Suppress HF logging
        import logging
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        # Tokenize entire dataset into flat array
        text_blob = "\n\n".join(dataset["text"])
        tokens = self.tokenizer.encode(text_blob)
        self.data = np.array(tokens, dtype=np.uint16)
        print(f"✅ Data loaded. {len(self.data):,} tokens.")

    def get_stream_generator(self, batch_size, device):
        """Yields batches sequentially to harvest continuously."""
        total_len = len(self.data)
        chunk_size = self.seq_len * batch_size
        num_batches = total_len // chunk_size
        
        # Reshape data into (num_batches, batch_size, seq_len)
        # We cut off the remainder
        cutoff = num_batches * chunk_size
        stream_data = self.data[:cutoff].reshape(num_batches, batch_size, self.seq_len)
        
        for i in range(num_batches):
            batch_np = stream_data[i].astype(np.int64)
            yield torch.from_numpy(batch_np).to(device)

# --- 3. HARVESTER CLASS ---
class ActivationsHarvester:
    """
    Hooks into model layers, buffers activations, and saves them to disk.
    """
    def __init__(self, model, layers, save_dir, run_name, buffer_limit=100):
        self.model = model
        self.layers = layers
        self.save_dir = save_dir
        self.run_name = run_name
        self.buffer_limit = buffer_limit
        
        self.hooks = []
        self.buffers = {layer: [] for layer in layers}
        self.file_counters = {layer: 0 for layer in layers}
        
        os.makedirs(save_dir, exist_ok=True)
        self._register_hooks()
        
    def _register_hooks(self):
        print(f"🪝 Hooking layers: {self.layers}")
        for layer_idx in self.layers:
            # We hook the Attention Module specifically
            # In NewGPT, layers are in model.blocks
            target_module = self.model.blocks[layer_idx].attn
            
            # Register forward hook
            handle = target_module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output shape: [Batch, Seq, D_Model]
            # We detach and move to CPU immediately to save VRAM
            # We flatten Batch and Seq to [Batch*Seq, D_Model] for SAE training
            activations = output.detach().flatten(0, 1).cpu()
            self.buffers[layer_idx].append(activations)
        return hook

    def check_flush(self, force=False):
        """Checks if buffers are full and flushes to disk."""
        # Check first layer length (assuming all layers fill equally)
        if len(self.buffers[self.layers[0]]) >= self.buffer_limit or force:
            self._flush()

    def _flush(self):
        for layer_idx in self.layers:
            if not self.buffers[layer_idx]:
                continue
                
            # Concatenate all buffered batches
            data = torch.cat(self.buffers[layer_idx], dim=0) # [Total_Tokens, D_Model]
            
            # Save
            count = self.file_counters[layer_idx]
            fname = f"{self.run_name}_L{layer_idx}_{count:04d}.pt"
            fpath = os.path.join(self.save_dir, fname)
            
            torch.save(data, fpath)
            
            # Reset
            self.buffers[layer_idx] = []
            self.file_counters[layer_idx] += 1
            
        if any(len(b) > 0 for b in self.buffers.values()):
            print(f"💾 Flushed activations to {self.save_dir}")

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.check_flush(force=True)

# --- 4. MODEL DEFINITIONS ---
# (Must match the training script exactly)

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

    def forward(self, idx):
        B, S = idx.shape
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# --- 5. MAIN EXECUTION ---
def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 STARTING HARVEST: {args.run_name}")
    print(f"   Model: {args.model_path}")
    print(f"   Saving to: {args.save_dir}")
    print(f"   Target: {args.tokens_to_harvest:,} tokens")

    # 1. Load Model
    config = args 
    model = NewGPT(config).to(DEVICE)
    
    try:
        # Load state dict
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # We use strict=False because the saved model might contain 'lambdas' buffers 
        # or optimizer states that our inference definition doesn't need.
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    model.eval()

    # 2. Attach Harvester
    harvester = ActivationsHarvester(
        model, 
        layers=args.layers, 
        save_dir=args.save_dir, 
        run_name=args.run_name,
        buffer_limit=args.buffer_size
    )

    # 3. Data Stream
    data_loader = HFWikiTextLoader(args.ctx_len)
    stream = data_loader.get_stream_generator(args.batch_size, DEVICE)

    # 4. Harvest Loop
    tokens_processed = 0
    pbar = tqdm(total=args.tokens_to_harvest, desc="Harvesting")

    with torch.no_grad():
        for batch in stream:
            # Forward pass triggers hooks automatically
            model(batch)

            # Update accounting
            batch_tokens = batch.numel()
            tokens_processed += batch_tokens
            pbar.update(batch_tokens)
            
            # Check for save
            harvester.check_flush()

            if tokens_processed >= args.tokens_to_harvest:
                break

    # 5. Finalize
    harvester.detach() # Flushes remaining buffer
    pbar.close()
    print(f"✅ Harvest Complete. Data saved to {args.save_dir}")

if __name__ == "__main__":
    main()
