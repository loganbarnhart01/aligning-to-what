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
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.commands.cli_utils import TrlParser

# from trainer.ppo_trainer import PPOv2Trainer
# from models.reward_model import RewardModelWrapper

# --reward_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file train/deepspeed_zero3.yaml train/ppo.py         \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B         \
    --sft_model_path=meta-llama/Meta-Llama-3-8B         \
    --reward_model_path NCSOFT/Llama-3-OffsetBias-RM-8B     \
    --per_device_train_batch_size 4         \
    --learning_rate 3e-5         \
    --gradient_accumulation_steps 4         \
    --gradient_checkpointing=True         \
    --logging_steps 10         \
    --eval_steps 500         \
    --save_steps=500         \
    --output_dir=/home/logan/covert-bias/weights/ppo_1         \
    --warmup_steps 150         \
    --report_to wandb         \
    --logging_first_step=true         \
    --no_remove_unused_columns         \
    --use_peft=true         \
    --lora_r=16         \
    --lora_alpha=16        \
    --fp16=true      \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size 1 \
    --non_eos_penalty   \
    --stop_token eos    \
    --per_device_eval_batch_size=4      \
    &> nohup.out &

python train/ppo.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --sft_model_path meta-llama/Meta-Llama-3-8B \
    --reward_model_path NCSOFT/Llama-3-OffsetBias-RM-8B     \
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
    print("torch_dtype: ", torch_dtype)
    model_kwargs = dict(
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if config.gradient_checkpointing else True,
            )
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(config.sft_model_path, num_labels=1, )
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, local_files_only=True)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, local_files_only=True, **model_kwargs)
    policy.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    ref_policy.config.pad_token_id = tokenizer.pad_token_id

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