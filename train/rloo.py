import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from trl import ModelConfig, get_peft_config
from trl.trainer.rloo_trainer import RLOOConfig
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.commands.cli_utils import TrlParser
from peft import get_peft_model

from trl.trainer.rloo_trainer import RLOOTrainer
# from models.reward_model import RewardModelWrapper

# --reward_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1     \
"""
CUDA_VISIBLE_DEVICES=0,1,2 nohup accelerate launch --config_file train/deepspeed_zero3_1.yaml train/rloo.py         \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B         \
    --sft_model_path=meta-llama/Meta-Llama-3-8B         \
    --reward_model_path NCSOFT/Llama-3-OffsetBias-RM-8B     \
    --per_device_train_batch_size 4         \
    --learning_rate 3e-5         \
    --gradient_accumulation_steps 2         \
    --gradient_checkpointing=True         \
    --logging_steps 10         \
    --eval_steps 500         \
    --save_steps=500         \
    --output_dir=/home/logan/covert-bias/weights/rloo_1         \
    --warmup_steps 150         \
    --report_to wandb         \
    --logging_first_step=true         \
    --no_remove_unused_columns         \
    --use_peft=true         \
    --lora_r=16         \
    --lora_alpha=16        \
    --fp16=true      \
    --num_train_epochs=1 \
    --rloo_k=4   \
    --non_eos_penalty   \
    --stop_token eos    \
    --per_device_eval_batch_size=4      \
    &> nohup.out &


"""

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""
    def tokenize(element):
        dataset_text_field = "chosen"
        def process_conversation(conversation):
            if conversation[-1]["role"] == "assistant":
                return conversation[:-1]
            return conversation
        processed_conversations = [process_conversation(conv) for conv in element[dataset_text_field]]
        outputs = tokenizer.apply_chat_template(
            processed_conversations,
            padding=False,
        )
        return {"input_ids": outputs}
    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=4,  # multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

if __name__ == "__main__":
    parser = TrlParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    shutil.rmtree(config.output_dir, ignore_errors=True)
    ###############
    # Model & Tokenizer
    ###############
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    torch_dtype = torch.float16
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    # reward_model = RewardModelWrapper(config.reward_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1, torch_dtype=torch_dtype
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch_dtype)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch_dtype)
    if model_config.use_peft:
        lora_config = get_peft_config(model_config)
        policy = get_peft_model(policy, lora_config)

    ################
    # Dataset
    ################
    eval_samples = 20
    train_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="train")
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="test").select(range(eval_samples))
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()