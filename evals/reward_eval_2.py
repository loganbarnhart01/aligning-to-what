import gc
import pickle
import argparse

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main(args):
    if not args.completion_path:
        base_model_name = args.base_model_path
        print(f"Loading base model {base_model_name}...")
        weight_path = args.checkpoint_path
        # checkpoints = [f"checkpoint-{i}" for i in range(500, 8500, 500)] + ['checkpoint-8345']
        
        eval_samples = 1000
        eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="test").select(range(eval_samples))

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

        print("Generating Completions...")

        def generate_completions(model, prompt):
            inputs = {"input_ids": tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)}
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            input_len = inputs['input_ids'].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # Adjust as needed
                temperature=0.7,    # Adjust as needed
                do_sample=True,
            )
            outputs = outputs[:, input_len:]
            return ''.join([tokenizer.decode(output, skip_special_tokens=True) for output in outputs.tolist()])

        completions = []

        if weight_path:
            peft_model = PeftModel.from_pretrained(base_model, weight_path)
            peft_model.eval()
            
            for row in tqdm(eval_dataset, desc=f"Generating completions...", leave=False):
                prompt = row['chosen'][:-1]
                completion = generate_completions(peft_model, prompt)
                message = prompt + [{"role": "assistant", "content": completion}]
                completions.append(message)
            
            del peft_model
        else:
            for row in tqdm(eval_dataset, desc=f"Generating completions...", leave=False):
                prompt = row['chosen'][:-1]
                completion = generate_completions(base_model, prompt)
                message = prompt + [{"role": "assistant", "content": completion}]
                completions.append(message)
            
        del base_model, tokenizer#, eval_dataset
        gc.collect()
        print("Writing completions to file...")
        output_file = f"{args.output_path}_completions.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(completions, f)
    else:
        print("Loading completions from file...")
        with open(args.completion_path, 'rb') as f:
            completions = pickle.load(f)

    print("Scoring completions...")

    rm_tokenizer = AutoTokenizer.from_pretrained("NCSOFT/Llama-3-OffsetBias-RM-8B", trust_remote_code=True)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model='NCSOFT/Llama-3-OffsetBias-RM-8B',
        device="cuda",
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.float16}
        )
    
    pipe_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 1
    }

    #scores = {}
    avg_scores = []
    # for checkpoint in tqdm(checkpoints, desc="Checkpoints"):
    #     #scores[checkpoint] = []
    #     avg_scores[checkpoint] = 0
    for message in tqdm(completions, desc=f"Scoring...", leave=False):
        # check if message is already in chat format:
        if not isinstance(message[0], dict):
            message = [{"role": "user", "content": message[0]}, {"role": "assistant", "content": message[1]}]
        inputs = [rm_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
        pipe_outputs = rm_pipe(inputs, **pipe_kwargs)
        score = [output[0]["score"] for output in pipe_outputs][0]
        avg_scores.append(score)
    #print("Writing to file...")
    output_file = f"{args.output_path}_scores.pkl"
    print(f"Average score: {np.mean(avg_scores):.4f}")
    with open(output_file, 'wb') as f:
       pickle.dump(avg_scores, f)

    if args.score_original:
        if not eval_dataset:
            eval_samples = 1000
            eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="test").select(range(eval_samples))
        model_win_rate_chosen = 0
        model_win_rate_rejected = 0
        print("Scoring original completions...")
        for idx, row in tqdm(enumerate(eval_dataset), desc=f"Scoring original completions...", leave=False):
            chosen = row['chosen']
            rejected = row['rejected']
            chosen_inputs = [rm_tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
            rejected_inputs = [rm_tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
            inputs = chosen_inputs + rejected_inputs
            pipe_outputs = rm_pipe(inputs, **pipe_kwargs)
            score = [output[0]["score"] for output in pipe_outputs]
            if avg_scores[idx] > score[0]:
                model_win_rate_chosen += 1
            if avg_scores[idx] > score[1]:
                model_win_rate_rejected += 1
            print(f"Score: {avg_scores[idx]:.4f}, Chosen: {score[0]:.4f}, Rejected: {score[1]:.4f}")
            
        print(f"Average score of model completions: {np.mean(avg_scores):.4f}")
        print(f"Model win rate vs. chosen: {model_win_rate_chosen / len(eval_dataset):.4f}")
        print(f"Model win rate vs. rejected: {model_win_rate_rejected / len(eval_dataset):.4f}")


            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--checkpoint_path", type=str, required=False) # path to model weights
    parser.add_argument("--output_path", type=str, required=True) 
    parser.add_argument("--score_original", type=bool, default=False)
    parser.add_argument("--completion_path", type=str, required=False) # path to model weights
    args = parser.parse_args()
    main(args)
