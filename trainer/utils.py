import os
import gc
from typing import Tuple, Union
from contextlib import contextmanager

import torch
from trl.trainer.utils import first_true_indices

from accelerate import Accelerator
from accelerate.utils import is_deepspeed_available
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.nn.parallel.distributed import DistributedDataParallel
from trl.models import PreTrainedModelWrapper
from trl.models.utils import remove_hooks, add_hooks

from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel

if is_deepspeed_available():
    import deepspeed

import torch
from safetensors.torch import save_file, load_file

def get_generation_model(base_model_name, adapter_path, device='cuda'):
    # Load the base model without quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # or whatever dtype you prefer for generation
        device_map=device
    )
    
    # Load the PEFT adapter
    generation_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Optionally, merge the adapter weights into the base model
    generation_model = generation_model.merge_and_unload()
    
    return generation_model.to(device)

@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"], 
    accelerator: "Accelerator", 
    is_peft_model: bool = False
) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
    if is_peft_model:
        base_model_name = model.config._name_or_path  # Adjust this if needed
        adapter_path = model.peft_config['default'].path
        print(f"Base model name: {base_model_name}")
        print(f"Adapter path: {adapter_path}")
        
        # Load the generation model on a single GPU
        generation_model = get_generation_model(base_model_name, adapter_path)
        
        yield generation_model
        
        torch.cuda.empty_cache()
        del generation_model
        gc.collect()


def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = output.rewards
    scores = output.score
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        scores,
        sequence_lengths,
    )