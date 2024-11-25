Covert Bias Measurement 

This repository contains all of the necessary code for replicating the findings of our paper, "Aligning to What? Limits to RLHF Based Alignment" 

|-`data/` contains the AAE and SAE data used for matched-guise probing
|-`evals/` contains our implementations of matched-guise probing in the various settings
|-`final_checkpoints/` contains the LoRA adapter weights we collected after our post-training experiments
|-`models/` contains necessary model classes to utilize the reward model we used during training. To use a different reward model, you may be able to use the normal Huggingface Trainer insteead
|-`train/` contains our training scripts for relevant RLHF methods
|-`trainer/` contains the modified TRL scripts for training RLOO with our specific reward model 
