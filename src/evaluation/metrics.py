import torch
import numpy as np
from tqdm.auto import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


class DetoxificationMetrics:
    def __init__(
            self, 
            batch_size=32, 
            use_cuda=False, 
            verbose=True, 
            aggregate=False, 
            style_calibration=None, 
            meaning_calibration=None, 
            fluency_calibration=None
        ):
        
        self.style_model, self.style_tokenizer = self._load_model('SkolkovoInstitute/russian_toxicity_classifier', use_cuda=use_cuda, requires_grad=False)
        self.meaning_model, self.meaning_tokenizer = self._load_model('cointegrated/LaBSE-en-ru', use_cuda=use_cuda, model_class=AutoModel, requires_grad=False)
        self.fluency_model, self.fluency_tokenizer = self._load_model('SkolkovoInstitute/rubert-base-corruption-detector', use_cuda=use_cuda, requires_grad=False)

        self.batch_size = batch_size
        self.verbose = verbose
        self.aggregate = aggregate
        self.style_calibration = style_calibration
        self.meaning_calibration = meaning_calibration
        self.fluency_calibration = fluency_calibration
    
    def _load_model(self, model_name=None, model=None, tokenizer=None, model_class=AutoModelForSequenceClassification, use_cuda=True, requires_grad=True):
        if model is None:
            if model_name is None:
                raise ValueError('Either model or model_name should be provided')
            model = model_class.from_pretrained(model_name)
            if torch.cuda.is_available() and use_cuda:
                model.cuda()
        if tokenizer is None:
            if model_name is None:
                raise ValueError('Either tokenizer or model_name should be provided')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad

        return model, tokenizer
    
    def _rotation_calibration(self, data, coef=1.0, px=1, py=1, minimum=0, maximum=1):
        result = (data - px) * coef + py
        result = np.clip(result, minimum, maximum)
        return result

    def _prepare_target_label(self, model, target_label):
        if target_label in model.config.id2label:
            pass
        elif target_label in model.config.label2id:
            target_label = model.config.label2id.get(target_label)
        elif target_label.isnumeric() and int(target_label) in model.config.id2label:
            target_label = int(target_label)
        else:
            raise ValueError(f'target_label "{target_label}" is not in model labels or ids: {model.config.id2label}.')
        return target_label

    def _classify_texts(self, model, tokenizer, texts, second_texts=None, target_label=None, batch_size=32, verbose=False):
        target_label = self._prepare_target_label(model, target_label)
        
        res = []
        if verbose:
            tq = trange
        else:
            tq = range

        for i in tq(0, len(texts), batch_size):
            inputs = [texts[i:i+batch_size]]
            if second_texts is not None:
                inputs.append(second_texts[i:i+batch_size])
            inputs = tokenizer(*inputs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                preds = torch.softmax(model(**inputs).logits, -1)[:, target_label].cpu().numpy()
            res.append(preds)

        return np.concatenate(res)

    def evaluate_style(self, model, tokenizer, texts, target_label=1, batch_size=32, verbose=False):
        target_label = self._prepare_target_label(model, target_label)

        scores = self._classify_texts(
            model,
            tokenizer,
            texts, 
            batch_size=batch_size, verbose=verbose, target_label=target_label
        )

        return self._rotation_calibration(scores, 0.90)

    def _encode_cls(self, texts, model, tokenizer, batch_size=32, verbose=False):
        results = []
        if verbose:
            tq = trange
        else:
            tq = range
        for i in tq(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            with torch.no_grad():
                out = model(**tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device))
                embeddings = out.pooler_output
                embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
                results.append(embeddings)

        return np.concatenate(results)

    def evaluate_cosine_similarity(self, model, tokenizer, original_texts, rewritten_texts, batch_size=32, verbose=False):
        scores = (
            self._encode_cls(original_texts, model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose)
            * self._encode_cls(rewritten_texts, model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose)
        ).sum(1)

        return self._rotation_calibration(scores, 1.50)

    def evaluate_cola_relative(self, model, tokenizer, original_texts, rewritten_texts, target_label=1, batch_size=32, verbose=False, maximum=0):
        target_label = self._prepare_target_label(model, target_label)

        original_scores = self._classify_texts(
            model, tokenizer,
            original_texts,
            batch_size=batch_size, 
            verbose=verbose, 
            target_label=target_label
        )

        rewritten_scores = self._classify_texts(
            model, tokenizer,
            rewritten_texts,
            batch_size=batch_size, 
            verbose=verbose, 
            target_label=target_label
        )

        scores = rewritten_scores - original_scores
        if maximum is not None:
            scores = np.minimum(0, scores)

        return self._rotation_calibration(scores, 1.15, px=0)
    
    @torch.no_grad()
    def __call__(self, original_texts, rewritten_texts):

        if self.verbose: print('Style evaluation')
        accuracy = self.evaluate_style(
            self.style_model,
            self.style_tokenizer,
            rewritten_texts,
            target_label=0,
            batch_size=self.batch_size, 
            verbose=self.verbose
        )

        if self.verbose: print('Meaning evaluation')
        similarity = self.evaluate_cosine_similarity(
            self.meaning_model,
            self.meaning_tokenizer,
            original_texts, 
            rewritten_texts,
            batch_size=self.batch_size, 
            verbose=self.verbose
        )

        if self.verbose: print('Fluency evaluation')
        fluency = self.evaluate_cola_relative(
            self.fluency_model,
            self.fluency_tokenizer,
            rewritten_texts=rewritten_texts,
            original_texts=original_texts,
            target_label=1,
            batch_size=self.batch_size, 
            verbose=self.verbose,
        )

        joint = accuracy * similarity * fluency

        if self.verbose and (self.style_calibration or self.meaning_calibration or self.fluency_calibration):
            print('Scores:')
            print(f'Style transfer accuracy (STA):  {np.mean(accuracy)}')
            print(f'Meaning preservation (SIM):     {np.mean(similarity)}')
            print(f'Fluency score (FL):             {np.mean(fluency)}')
            print(f'Joint score (J):                {np.mean(joint)}')

        result = dict(
            accuracy=accuracy,
            similarity=similarity,
            fluency=fluency,
            joint=joint
        )

        if self.aggregate:
            return {k: float(np.mean(v)) for k, v in result.items()}
        
        return result

    @torch.no_grad()
    def evaluate_model(self, model, toxic_texts, neutral_texts, propmts=''):

        if self.verbose:
            tq = trange
        else:
            tq = range

        rewritten_texts = []
        for i in tq(len(toxic_texts)):
            outputs = model.generate(propmts + toxic_texts[i])
            rewritten_texts.append(outputs)

        scores = self(neutral_texts, rewritten_texts)
        return scores
