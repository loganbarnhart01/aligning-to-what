import json
import pandas as pd

from openai import OpenAI
from datasets import load_dataset, DatasetDict

api_key = "your-api-key"

client = OpenAI(api_key=api_key)

dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style")

system_prompt = """
You are an assistant responsible for translating English text into African American English (AAE) for educational purposes. Your task is to accurately translate **all** "content" fields, including slang and informal language, from Standard English to African American English (AAE). You must ensure that the translation retains the original meaning and intent, while adjusting the style to reflect natural AAE speech patterns. Offensive language must not be censored, but in the context of this paper, we avoid including such examples.

You will be provided with an input conversation in a dictionary format, where each entry includes "content" and "role". Your output should be a JSON object that retains the same format but translates all the "content" fields to AAE.

Ensure that **all** sentences, including short, long, and complex ones, are properly translated into AAE.

Input Example:
{
    "conversation": 
        [
            {"content": "What are some common words in English?", "role": "user"},
            {"content": "Here is a simple list:\n\nGood morning, please, thank you, excuse me...", "role": "assistant"},
            {"content": "Why did you do that?", "role": "user"},
            {"content": "I didn't mean to. Please remain calm.", "role": "assistant"}
        ]
}

Expected Output Example:
{
    "translated_conversation": 
        [
            {"content": "What’s some common words in English?", "role": "user"},
            {"content": "Here go a short list:\n\nGood mornin', please, thank you, 'scuse me...", "role": "assistant"},
            {"content": "Why you do that?", "role": "user"},
            {"content": "I ain't mean to. Just relax.", "role": "assistant"}
        ]
}

Translate **all** "content" fields to AAE, including long and complex sentences, while keeping the structure intact.
"""

tasks = []

def return_task(j, content):
    task = {
        "custom_id": f"task-{j}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
        }
    }
    return task

list_of_contents_roles = []

for i, row in enumerate(dataset['train']):
    row['chosen'].append(row['rejected'][-1])
    task = return_task(i, str(row['chosen']))
    tasks.append(task)


# Creating the file
for i in range(0, len(tasks), 1000):
    file_name = f"data/hh/batch_{i}_{i+1000}.jsonl"
    with open(file_name, 'w') as file:
        for obj in tasks[i:i+1000]:
            file.write(json.dumps(obj) + '\n')


for i in range(0, len(tasks), 1000):
  file_name = f'data/hh/batch_{i}_{i+1000}.jsonl'
  batch_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
  )
  batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
  )
  # wait for the batch job to complete
  batch_job = client.batches.wait_for(batch_job.id)