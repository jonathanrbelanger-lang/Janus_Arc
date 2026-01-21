Janus Arc: Adaptive Interpretability Research

Janus Arc is a PyTorch implementation of "Physics-Based Steering" for Language Models. It explores how homeostatic pressure (forcing attention heads to differ) and adaptive scheduling can spontaneously induce "Hero Heads"—highly specialized components that emerge to handle complex token dependencies.

This repository provides a complete, end-to-end pipeline: from training the base model with adaptive controllers, to harvesting activations, training Sparse Autoencoders (SAEs), and finally visualizing the resulting interpretability features.

⚡ Quick Start

This codebase is self-contained. It relies on Hugging Face for data (wikitext-103) and tokenizers.

1. Installation
git clone https://github.com/yourusername/janus-arc.git
cd janus-arc
pip install torch numpy pandas tqdm transformers datasets matplotlib seaborn

2. The Research Pipeline

The scripts are designed to be run in a specific order to replicate the full experiment.

Phase 1: Training the Model
Train the adaptive model (Janus) which uses a PID controller to regulate attention head diversity during training.

# Trains a ~85M parameter model on WikiText-103
python janus_adaptive.py --save_dir ./checkpoints/v1 --target_sigma 0.0035

Phase 2: Harvesting Activations
Once trained, we stream data through the model to collect raw neural activations from specific layers.
python harvest_activations.py \
    --model_path ./checkpoints/v1/janus_adaptive_final.pt \
    --run_name adaptive \
    --layers 3 6 9

Phase 3: Training SAEs (Sparse Autoencoders)

Train dictionary learning models on the harvested activations to decompose dense layers into interpretable features.
python train_sae.py --runs adaptive --layers 3 6 9

Phase 4: Finding the "Hero Head"
Run an ablation study to mathematically identify which attention head bears the most load (impacts loss the most when removed).
python ablation_test.py \
    --adaptive_model ./checkpoints/v1/janus_adaptive_final.pt \
    --save_file results_ablation.csv

Phase 5: Hero Analysis
Visualize the specific features learned by the Hero Head. This generates attention heatmaps and decodes the neural circuitry using the SAEs trained in Step 3.
python hero_head_analysis.py \
    --ablation_csv results_ablation.csv \
    --text "The quick brown fox jumps over the lazy dog because of complex physics."

📂 Architecture Overview
The "Homeostatic" Controller
Located in janus_adaptive.py, the Homeostat class acts as a PID controller during the training loop.
 * Input: The "Sigma" (\sigma) of the attention heads (a measure of redundancy/similarity).
 * Target: A user-defined orthogonality setpoint (e.g., 0.0035).
 * Output: A penalty coefficient (\lambda) applied to the loss function.
 * Result: If heads become too similar, pressure increases, mechanically forcing them apart. If they diverge too much, pressure relaxes.
The Model (NewGPT)
A modernized GPT-2 style architecture featuring:
 * RMSNorm (Pre-normalization)
 * SwiGLU Activations
 * RoPE (Rotary Positional Embeddings)
 * Project-Out Attention: Attention outputs are calculated explicitly before the output projection to allow for cleaner SAE harvesting.
📊 Data
The repository uses the Hugging Face datasets library to automatically download and cache wikitext-103-raw-v1. No manual dataset preparation is required. All data loading logic is handled in the HFWikiTextLoader class found in each script.
📜 License
This project is licensed under the MIT License.
