# Janus Arc: Research Codebase

**Janus Arc** is a raw research artifact containing the complete implementation of the "Janus" language model architecture experiments. This repository is released primarily for **code review, architectural demonstration, and educational purposes**.

It serves as a transparency layer for the "Project Janus" initiatives, showcasing the custom "Clean Room" transformer implementation, the "Gauntlet" architecture search, and the forensic telemetry tools used to analyze model interpretability.

## ⚠️ Important: Execution & Retrofitting

**This code is a direct port from a Google Colab research environment.**

It is **not** a plug-and-play library. The scripts contain hardcoded file paths, directory structures, and Google Drive mounting logic specific to the original author's research environment.

**If you attempt to execute this code, you must perform a deep analysis and retrofit the `SETUP` sections.**

Specifically, you will need to:
1.  **Remove Google Drive dependencies**: The code calls `drive.mount('/content/drive')`. This must be replaced with local file system logic.
2.  **Refactor Data Paths**: Variable constants such as `DATA_FILE` and `RESULTS_DIR` point to specific paths (e.g., `/content/drive/MyDrive/Project_XAI_Physical_Janus/...`). These must be updated to point to your own local binary datasets (TinyStories, WikiText, etc.).
3.  **Check Data Formats**: The dataloader expects a specific raw binary format (`np.memmap`). You will need to ensure your dataset matches this schema or rewrite the `get_batch` function.

## 📂 Codebase Overview

The core logic is consolidated within `janus_hero_test.py`. This single file acts as a monolithic research suite containing several distinct modules:

### 1. The Clean Room Transformer
A custom PyTorch implementation of a GPT-style model designed for "Physics-Based Steering."
* **Key Class**: `CleanGPT` and `CleanBlock`.
* **Features**: Implements `lambda_diversity` (orthogonality pressure) to mechanically force attention heads apart during training, alongside standard features like RoPE (Rotary Positional Embeddings) and SwiGLU.

### 2. The Gauntlet (Architecture Search)
An automated testing loop that pits different model configurations against one another.
* **Function**: `run_gauntlet()`
* **Purpose**: Probes architectural choices (e.g., 32d vs. 64d head geometry) and pressure schedules (Cubic vs. Linear) to find optimal interpretability settings before a full training run.

### 3. Janus Autopsy (Forensics)
A post-run analysis tool built into the script.
* **Function**: `JanusAutopsy` class.
* **Purpose**: Parses the `.parquet` telemetry logs generated during training to visualize "Entropy," "Elasticity," and "Head Orthogonality."

## 🛠️ Navigating the Code

For reviewers looking to understand the core mechanisms, focus on these sections within `janus_hero_test.py`:

* **Line ~40**: `CleanConfig` - The hyperparameter schema.
* **Line ~80**: `CleanGPT` - The model architecture, specifically the `forward` pass where pressure is applied.
* **Line ~350**: `JanusTrainer` - The training loop that handles the "Burn Schedule" (Ignition, Pressurization, Cruising).

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

