Covert Bias Measurement 

This repository contains all of the necessary code for replicating the findings of our paper, "Aligning to What? Limits to RLHF Based Alignment" 

```
├── data/              # Contains the AAE and SAE data used for matched-guise probing and our script for dataset generation
├── evals/             # Contains our implementations of matched-guise probing in the various settings
├── final_checkpoints/ # Contains the LoRA adapter weights collected after post-training experiments
├── models/            # Contains necessary model classes for the reward model used during training
│                      # To use a different reward model, you may be able to use the normal
│                      # Huggingface Trainer instead
├── train/             # Contains training scripts for relevant RLHF methods
└── trainer/           # Contains modified TRL scripts for training RLOO with our specific reward model
