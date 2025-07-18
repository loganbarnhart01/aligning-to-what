import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import ModelConfig, get_peft_config
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.commands.cli_utils import TrlParser
from peft import get_peft_model

# from trainer.rloo_trainer import RLOOTrainer
# from models.reward_model import RewardModelWrapper


"""
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
CUDA_VISIBLE_DEVICES=0,1,2 nohup accelerate launch --config_file train/deepspeed_zero3_1.yaml train/rloo.py         \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B         \
    --sft_model_path=meta-llama/Meta-Llama-3-8B         \
    --reward_model_path NCSOFT/Llama-3-OffsetBias-RM-8B     \
    --per_device_train_batch_size 2         \
    --learning_rate 1e-4         \
    --gradient_accumulation_steps 3         \
    --gradient_checkpointing=True         \
    --logging_steps 10         \
    --eval_steps 1000         \
    --output_dir=/home/logan/covert-bias/weights/rloo_1         \
    --warmup_steps 150         \
    --report_to wandb         \
    --logging_first_step         \
    --no_remove_unused_columns         \
    --use_peft         \
    --lora_r=16         \
    --lora_alpha=16        \
    --fp16      \
    &> nohup.out &

    --save_steps 1500         \

"""


if __name__ == "__main__":
    parser = TrlParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    shutil.rmtree(config.output_dir, ignore_errors=True)
    ###############
    # Model & Tokenizer
    ###############
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.chat_template is None:
       tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    # reward_model = RewardModelWrapper(config.reward_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.sft_model_path, num_classes=1, trust_remote_code=True)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    if model_config.use_peft:
        lora_config = get_peft_config(model_config)
        policy = get_peft_model(policy, lora_config)

    ################
    # Dataset
    ################
    eval_samples = config.per_device_train_batch_size * config.gradient_accumulation_steps * 4
    train_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="train")
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="test").select(range(eval_samples))
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
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()