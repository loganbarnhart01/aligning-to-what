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

if is_deepspeed_available():
    import deepspeed

@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"], 
    accelerator: "Accelerator", 
    is_peft_model: bool = False
) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    
    if is_peft_model:
        if hasattr(unwrapped_model, 'disable_adapter'):
            unwrapped_model.disable_adapter()
        elif hasattr(unwrapped_model, 'base_model') and hasattr(unwrapped_model.base_model, 'disable_adapter'):
            unwrapped_model.base_model.disable_adapter()
    
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield unwrapped_model
            add_hooks(model)
    else:
        yield unwrapped_model
    
    if is_peft_model:
        if hasattr(unwrapped_model, 'enable_adapter'):
            unwrapped_model.enable_adapter()
        elif hasattr(unwrapped_model, 'base_model') and hasattr(unwrapped_model.base_model, 'enable_adapter'):
            unwrapped_model.base_model.enable_adapter()


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