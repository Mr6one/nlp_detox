import torch
from src.models.base import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class Bloom(BaseModel):
    def __init__(self, model_name, device='cpu'):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    @torch.no_grad()
    def generate(self, text, promt='', max_length=20):
        inputs = self.tokenizer(promt + text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


class Bloom560m(Bloom):
    def __init__(self, device='cpu'):
        super().__init__('bigscience/bloom-560m', device)


class Bloom1b1(Bloom):
    def __init__(self, device='cpu'):
        super().__init__('bigscience/bloom-1b1', device)


class Bloom3b(Bloom):
    def __init__(self, device='cpu'):
        super().__init__('bigscience/bloom-3b', device)


class Bloom7b1(Bloom):
    def __init__(self, device='cpu'):
        super().__init__('bigscience/bloom-7b1', device)
