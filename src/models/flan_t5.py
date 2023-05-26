import torch
from src.models.base import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class FlanT5(BaseModel):
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


class FlanT5Small(FlanT5):
    def __init__(self, device='cpu'):
        super().__init__('google/flan-t5-small', device)


class FlanT5Base(FlanT5):
    def __init__(self, device='cpu'):
        super().__init__('google/flan-t5-base', device)


class FlanT5Large(FlanT5):
    def __init__(self, device='cpu'):
        super().__init__('google/flan-t5-large', device)


class FlanT5XL(FlanT5):
    def __init__(self, device='cpu'):
        super().__init__('google/flan-t5-xl', device)
