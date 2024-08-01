import shutil

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import get_peft_model


from trl import ModelConfig, get_peft_config
from trl.trainer.ppov2_trainer import PPOv2Config
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.commands.cli_utils import TrlParser

from trainer.ppo_trainer import PPOv2Trainer
from models.reward_model import RewardModelWrapper

"""
python train/ppo.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --sft_model_path meta-llama/Meta-Llama-3-8B \
    --reward_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --output_dir <output_dir> \
    --per_device_train_batch_size 8 \
    --total_episodes 10000\
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing=True \
    --learning_rate=1e-4 \
    --report_to=wandb \
    --logging_steps=10 \
    --eval_steps=500 \
    --fp16     \
    --use_peft     \
    --lora_r=16     \
    --load_in_4bit     \
    --lora_alpha=16    \
"""

if __name__ == "__main__":
    parser = TrlParser((PPOv2Config, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

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
            use_cache=False if config.gradient_checkpointing else True,
            )
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(config.sft_model_path, num_labels=1, local_files_only=True, **model_kwargs)
    reward_model = RewardModelWrapper(config.reward_model_path)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, local_files_only=True)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, local_files_only=True, **model_kwargs)
    if model_config.use_peft:
        lora_config = get_peft_config(model_config)
        value_model = get_peft_model(value_model, lora_config)
        policy = get_peft_model(policy, lora_config)
    ################
    # Dataset
    ################
    eval_samples = 20
    train_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="train")
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="test").select(range(eval_samples))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=4,  # multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )

    ################
    # Training
    ################
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()