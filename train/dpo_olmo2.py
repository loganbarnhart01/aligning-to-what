# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# regular:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=/home/logan/covert-bias/weights/llama3_sft/checkpoint-781 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="weights/dpo_sft" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

`CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --multi_gpu train/dpo_5.py \
    --dataset_name=allenai/olmo-2-1124-7b-preference-mix \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing=False \
    --logging_steps 10 \
    --eval_steps 1000 \
    --output_dir=/home/logan/covert-bias/weights/dpo_olmo_dataset \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --fp16 \
    --logging_first_step \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --num_train_epochs=3 \
    --torch_dtype=float16 \
    --save_steps=1500 &> nohup.out & 
    
# peft:
python train/dpo.py \
    --dataset_name=allenai/olmo-2-1124-7b-preference-mix \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing=True \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=/home/logan/covert-bias/weights/sft_dpo_1 \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --fp16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --num_train_epochs=1 \
    --torch_dtype=float16 \
    --save_steps=1000
"""
import os

import logging
import multiprocessing
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_peft_config,
)

from peft import PeftModel

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)

    def process(row):
        row["prompt"] = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template([row["chosen"][-1]], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template([row["rejected"][-1]], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds[args.dataset_train_split]
    # Pop off 100 samples for evaluation
    eval_dataset = train_dataset.select(range(100))
    train_dataset = train_dataset.select(range(100, 200100))

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_prompt_length=512,
            max_length=512,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
