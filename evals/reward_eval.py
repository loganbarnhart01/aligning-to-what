from typing import List, Dict
import pickle
import argparse

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main(args):
    base_model_name = args.base_model_path
    print(f"Loading base model {base_model_name}...")
    weight_path = args.checkpoint_path
    # checkpoints = [f"checkpoint-{i}" for i in range(500, 8500, 500)] + ['checkpoint-8345']
    
    eval_samples = 1000
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="test").select(range(eval_samples))

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Generating Completions...")

    if not args.completion_path:
        def generate_completions(model, prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
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

        # for checkpoint in tqdm(checkpoints, desc="Checkpoints"):
            # checkpoint_path = os.path.join(weight_path, checkpoint)
            # Load adapter weights
        if weight_path:
            peft_model = PeftModel.from_pretrained(base_model, weight_path)
            peft_model.eval()
            
            for row in tqdm(eval_dataset, desc=f"Generating completions...", leave=False):
                prompt = row['prompt']
                completion = generate_completions(peft_model, prompt)
                completions.append((prompt, completion))
            
            del peft_model
        else:
            for row in tqdm(eval_dataset, desc=f"Generating completions...", leave=False):
                prompt = row['prompt']
                completion = generate_completions(base_model, prompt)
                completions.append((prompt, completion))
            
        del base_model, tokenizer, eval_dataset
        print("Writing completions to file...")
        output_file = f"{args.output_path}_completions.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(completions, f)
    else:
        with open(args.completion_path, 'rb') as f:
            completions = pickle.load(f)

    print("Scoring completions...")

    reward_model = ArmoRMPipeline('RLHFlow/ArmoRM-Llama3-8B-v0.1', trust_remote_code=True)
        
    #scores = {}
    avg_scores = []
    # for checkpoint in tqdm(checkpoints, desc="Checkpoints"):
    #     #scores[checkpoint] = []
    #     avg_scores[checkpoint] = 0
    for prompt, completion in tqdm(completions, desc=f"Scoring...", leave=False):
            message = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
            score = reward_model(message)
#            scores[checkpoint].append((score['score'], prompt, completion))
            avg_scores.append(score['score'])
    #print("Writing to file...")
    output_file = f"{args.output_path}_scores.pkl"
    print(f"Average score: {np.mean(avg_scores):.4f}")
    with open(output_file, 'wb') as f:
       pickle.dump(avg_scores, f)

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--checkpoint_path", type=str, required=False) # path to model weights
    parser.add_argument("--output_path", type=str, required=True) 
    parser.add_argument("--completion_path", type=str, required=False) 
    args = parser.parse_args()
    main(args)
