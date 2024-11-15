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
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python train/sft.py \
    --model_name_or_path="meta-llama/Meta-Llama-3-8B" \
    --dataset_text_field="text" \
    --dataset_name="Open-Orca/SlimOrca" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=16 \
    --output_dir="weights/llama3_sft" \
    --logging_steps=1 \
    --num_train_epochs=2 \
    --save_steps=2500 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 \
    --max_seq_length=2048 \
    --trust_remote_code=True 
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################

    def process_conversations(example):
        mapped_conversations = [
            {"role": "system" if conv["from"] == "system" else "user" if conv["from"] == "human" else "assistant", 
            "content": conv["value"]}
            for conv in example["conversations"]
        ]
        return {"text": tokenizer.apply_chat_template(mapped_conversations, tokenize=False)}

    dataset = load_dataset(script_args.dataset_name)
    dataset = dataset.map(process_conversations, remove_columns=["conversations"])
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset["train"].select(range(100000))
    eval_dataset = dataset["train"].select(range(100000, 100500))

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    # )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)