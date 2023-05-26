import torch
from src.models.base import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MT5(BaseModel):
    def __init__(self, model_name, device='cpu'):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device

    @torch.no_grad()
    def generate(self, text, promt='', max_length=20):
        inputs = self.tokenizer(promt + text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


class MT5Small(MT5):
    def __init__(self, device='cpu'):
        super().__init__('google/mt5-small', device)


class MT5Base(MT5):
    def __init__(self, device='cpu'):
        super().__init__('google/mt5-base', device)


class MT5Large(MT5):
    def __init__(self, device='cpu'):
        super().__init__('google/mt5-large', device)


class MT5XL(MT5):
    def __init__(self, device='cpu'):
        super().__init__('google/mt5-xl', device)
