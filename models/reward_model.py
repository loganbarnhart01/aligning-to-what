from torch import nn
from transformers import AutoModelForSequenceClassification

class RewardModelWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.base_model_prefix = self.model.base_model_prefix
    
    def forward(self, *args, **kwargs):
        output= self.model(*args, **kwargs)
        if hasattr(output, 'hidden_state'):
            # output.hidden_states = output.hidden_state
            output.hidden_states = [output.score]
        return output
    
    def score(self, hidden_states):
        return hidden_states[0]