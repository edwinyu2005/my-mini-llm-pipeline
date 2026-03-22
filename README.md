# My Mini LLM Pipeline 🚀

This repository contains a complete, end-to-end pipeline for developing Large Language Models locally. It is designed to run efficiently on consumer-grade hardware (specifically optimized for NVIDIA RTX 3070 with 8GB VRAM).

## 📌 Project Overview

The goal of this project is to demystify the LLM lifecycle by building and tuning models from the ground up. The pipeline is divided into three distinct phases:

- **Phase 1: Pre-training (The Toy GPT)** Building a Character-level GPT model from scratch using native PyTorch. This phase focuses on the fundamental math behind Self-Attention and Transformer blocks.
- **Phase 2: Supervised Fine-Tuning (SFT)** Adapting a pre-trained base model (e.g., Qwen2.5-1.5B) to specific instructions using QLoRA and `unsloth` for extreme memory efficiency.
- **Phase 3: Inference & Evaluation** Merging LoRA adapters and deploying a local interactive interface to test the fine-tuned model.

## 💻 Hardware Requirements

- **OS:** Linux
- **GPU:** NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070)
- **CUDA:** 13.0+ recommended

## 📂 Project Structure

```text
my-mini-llm-pipeline/
├── data/                  # Raw and processed datasets (ignored by git)
├── notebooks/             # Jupyter notebooks for EDA and quick prototyping
├── src/                   # Reusable source code (the core package)
│   ├── __init__.py
│   ├── model/             # Model architectures (e.g., your custom GPT)
│   ├── data/              # Data loaders and tokenization utilities
│   └── utils/             # Helper functions, logging, etc.
├── scripts/               # Executable scripts for the pipeline
│   ├── 01_pretrain.py     # Script to train the toy model from scratch
│   ├── 02_sft.py          # Script for LoRA fine-tuning
│   └── 03_chat.py         # CLI or Gradio interface for inference
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

*(Coming soon: Instructions on how to set up the environment and run the Phase 1 pre-training script.)*
