# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Run the ORPO training script with the following command with some example arguments.
In general, the optimal configuration for ORPO will be similar to that of DPO without the need for a reference model:

# regular:
python examples/scripts/orpo.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-orpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/orpo.py \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="/home/logan/covert-bias/weights/orpo_epoch_1" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --fp16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import os
# os.environ['TRANSFORMERS_CACHE'] = '/scratch/alpine/reak3132/.cache'
# os.environ['HF_HOME'] =  '/scratch/alpine/reak3132/.cache'

import multiprocessing
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ORPOConfig, ORPOTrainer, get_peft_config, get_quantization_config, get_kbit_device_map, get_peft_config
from peft import get_peft_model

@dataclass
class ScriptArguments:
    dataset: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style",
        metadata={"help": "The name of the dataset to use."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ORPOConfig, ModelConfig))
    args, orpo_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, use_cache=False if orpo_args.gradient_checkpointing else True,)
    if model_config.use_peft:
        lora_config = get_peft_config(model_config)
        model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    ################
    # Dataset
    ################
    eval_samples = 20
    train_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="train")
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="test").select(range(eval_samples))

    def preprocess_function(examples):
        new_examples = {
            "chosen": [],
            "rejected": [],
            "prompt": [],
        }
        for chosen, rejected, prompt in zip(examples["chosen"], examples["rejected"], examples["prompt"]):
            formatted_chosen = ""
            formatted_rejected = ""
            for message in chosen[1:]:
                role = "Human" if message["role"] == "user" else "Assistant"
                formatted_chosen += f"{role}: {message['content']}\n\n"
            for message in rejected[1:]:
                role = "Human" if message["role"] == "user" else "Assistant"
                formatted_rejected += f"{role}: {message['content']}\n\n"
            formatted_prompt = f"Human: {prompt}\n\n"

            new_examples["chosen"].append(formatted_chosen.strip())
            new_examples["rejected"].append(formatted_rejected.strip())
            new_examples["prompt"].append(formatted_prompt.strip())

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )
    # raw_datasets = raw_datasets.filter(
    #     lambda x: len(x["chosen"]) <= orpo_args.max_length and len(x["rejected"]) <= orpo_args.max_length
    # )
    # train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    ################
    # Training
    ################
    trainer = ORPOTrainer(
        model,
        args=orpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # train and save the model
    trainer.train()
    trainer.save_model(orpo_args.output_dir)
