# coding=utf-8
"""PyTorch LLM Prompt Attribution with Density Function"""
import functools
import string


import tqdm
import numpy as np
import torch as tc
import transformers as trf


def zero_grad(*obj):
    if len(obj) > 1:
        for subobj in obj:
            zero_grad(subobj)
    elif hasattr(obj[0], "parameters"):
        for subobj in obj[0].parameters():
            zero_grad(subobj)
    elif obj[0].grad is not None:
        obj[0].grad.data.zero_()


def format_llama_weight(layer, wtype):
    assert wtype in {"q", "k", "v", 'o', 'gate', 'down', 'up', 'down', 'norm'}
    if wtype == "norm":
        pattern = "input_layernorm"
    elif len(wtype) == 1:
        pattern = "self_attn.%s_proj" % wtype
    else:
        pattern = "mlp.%s_proj" % wtype
    return "model.layers.%s.%s.weight" % (layer, pattern)


class Generator:
    def __init__(self, model, device="cuda:0", **params):
        super().__init__()
        self._name = model
        self._device = device
        self._params = {}
        self.parameters = params
        self.build()      

    def build(self):
        print("Initializing LLM: %s" % self._name)
        maps = None if self._device == "cpu" else "auto"
        self._tokenizer = trf.AutoTokenizer.from_pretrained(self._name, use_fast=False, padding_side="right", cache_dir="./cache")
        self._model = trf.AutoModelForCausalLM.from_pretrained(self._name, cache_dir="./cache", device_map=maps).float()
        self._out_embed = self._model.get_output_embeddings().weight.data.detach()
        self._inp_embed = self._model.get_input_embeddings()
        if not self._tokenizer.eos_token:
            self._tokenizer.eos_token = "</s>"
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._tokenizer.eos_token_id  
        self._config = self._model.config
        self._headsize = self._config.hidden_size // self._config.num_attention_heads

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._params.copy()

    @parameters.setter
    def parameters(self, params):
        for key, val in self._params:
            if key not in params:
                params[key] = val
        self._params = {"min_length": params.get("minlen", 1),
                        "max_length": params.get("maxlen", 50),
                        "temperature": params.get("temperature", 0.0),
                        "top_p": params.get("top_p", 0.000),
                        "num_return_sequences": params.get("ngen", 1),
                        "penalty_alpha": params.get("penalty", 0.),
                        "do_sample": False}
    
    def pretrain(self, data_file, output_dir, **config):
        from datasets import load_dataset
        maxlen = config.get("max_length", 1024)
        def tokenize(content):
            outputs = self._tokenizer(content["text"], truncation=True,
                                     max_length=maxlen,
                                     return_overflowing_tokens=True,
                                     return_length=True)
            input_batch = []
            for length, ids in zip(outputs["length"], outputs["input_ids"]):
                if length <= maxlen:
                    input_batch.append(ids)
            return {"input_ids": input_batch}
        dataset = load_dataset("csv", data_files=[data_file], delimiter='\t', column_names=['text'])
        
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)
        args = trf.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.get("batch_size", 1),
            per_device_eval_batch_size=config.get("batch_size", 1),
            evaluation_strategy="steps",
            eval_steps=config.get("eval_steps", 1000),
            logging_steps=config.get("eval_steps", 1000),
            save_steps=config.get("eval_steps", 1000),
            gradient_accumulation_steps=config.get("gradient_accumulation", 4),
            num_train_epochs=config.get("epochs", 3),
            weight_decay=config.get("weight_decay", 1e-7),
            warmup_steps=config.get("warmup", 1000),
            lr_scheduler_type=config.get("scheduler", "cosine"),
            learning_rate=config.get("learn_rate", 1e-5),
            fp16=config.get("fp16", True),
            )
        self._model.train()
        print("Pre-training info:", dataset["train"])
        trainer = trf.Trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=args,
            data_collator=trf.DataCollatorForLanguageModeling(self._tokenizer, mlm=False),
            train_dataset=dataset["train"],
            eval_dataset=dataset["train"].select(range(100))
            )
        trainer.train()
        trainer.save_model()

    def get_inputs(self, texts):
        inputs = self._tokenizer(texts, padding=True, max_length=1024,
                                 truncation=True, return_tensors="pt")
        for key in list(inputs.keys()):
            if key not in ["input_ids", "attention_mask"]:
                del inputs[key]
            else:
                inputs[key] = inputs[key].to(self._device)
        return inputs
    
    def tokenize(self, text):
        return self._tokenizer.tokenize(text.strip())

    def prepare4generate(self, input_texts):
        inputs = self.get_inputs(input_texts) 
        batch_size, seq_len = inputs['input_ids'].shape
        
        inputs['attention_mask'] = tc.flip(inputs['attention_mask'], dims=[1])
        shifts = seq_len - inputs['attention_mask'].sum(dim=-1)
        for idx in range(batch_size):
            inputs['input_ids'][idx] = inputs['input_ids'][idx].roll(shifts[idx].item())

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()} | self._params
        if inputs['min_length'] is not None:
            inputs['min_length'] = inputs['min_length'] + seq_len
        if inputs['max_length'] is not None:
            inputs['max_length'] = min(self._model.config.max_position_embeddings,
                                       inputs['max_length'] + seq_len)
        return inputs, seq_len

    def generate(self, texts):
        with tc.no_grad():
            self._model.eval()
            inputs, seq_len = self.prepare4generate(texts)
            output_ids = self._model.generate(**inputs)[:, seq_len:]
            return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def forward(self, texts):
        inputs = self.get_inputs(texts)
        return inputs, self._model(**inputs)

    def prepare4explain(self, inps, refs):
        if isinstance(inps, str):
            inps = [inps]
        if isinstance(refs, str):
            refs = [refs]
        assert isinstance(inps, list) and isinstance(refs, list)
        assert len(inps) == len(refs)
        texts = [i.strip() + " " + r.strip() \
                 for i, r in zip(inps, refs)]
        new_inps, new_refs = [], []
        for inp, ref, txt in zip(inps, refs, texts): 
            inp = self.tokenize(inp)
            ref = self.tokenize(ref)
            txt = self.tokenize(txt)
            assert len(txt) - (len(inp) + len(ref)) <= 1, str(txt) + " | " + str(inp) + " | " + str(ref)
            # the insert blank may be splitted into multiple tokens
            ref = txt[len(inp):]
            new_inps.append(inp)
            new_refs.append(ref)
        return new_inps, new_refs, texts
        
    def input_explain(self, inps, refs, L=10, b=7, p=4, eps=1e-7):
        self._model.eval()
        inps, refs, texts = self.prepare4explain(inps, refs) 
        ids = self.get_inputs(texts)["input_ids"]
        embs = self._inp_embed(ids.to(self._device)).detach().requires_grad_()
        
        # LLaMA family may always automatically append a prefix to the begining
        # you need to set bias=1 for GPT family! 
        # For other family, like T5, OPT, bloomz, you may need to revise this code manually.
        bias = 1 if "gpt" in self._name else 0

        expls, attrs, confs = [], [], []
        probs = tc.softmax(self._model(inputs_embeds=embs)["logits"], -1)
        for i, (inp, ref) in enumerate(zip(inps, refs)): 
            ref = tc.tensor(self._tokenizer.convert_tokens_to_ids(ref)).long()
            obj = probs[i, tc.arange(len(inp) - bias, len(inp) + len(ref)- bias), ref]  
            confs.append(obj.cpu().detach().numpy())
            grad = []
            for j in range(len(ref)): 
                zero_grad(self._model, embs)
                obj[j].backward(retain_graph=True)
                grad.append(embs.grad.data[i, 1 - bias:1 + len(inp) - bias].detach().cpu())

            if len(grad) == 0:
                expls.append(np.array([]))
                attrs.append(np.array([]))
                continue
            
            with tc.no_grad():
                # importance
                emb = embs[i, 1 - bias:1 + len(inp) - bias].unsqueeze(0).cpu()
                grad = tc.stack(grad, 0).cpu()
                expl = (grad * emb).sum(axis=-1).T
                expls.append(expl.numpy())

                # sparsify and normalize
                zeros = tc.zeros_like(expl)
                expl = tc.maximum(zeros, expl)
                expl = expl / (expl.max(axis=0, keepdims=True).values + eps)
                expl = tc.ceil(expl * L)
                expl = tc.where(expl <= b, zeros, expl)
                #expls.append(expl.numpy())

                # word attribution with density
                l1 = expl.sum(axis=-1)
                lp = (expl ** p).sum(axis=-1) ** (1. / p) + eps
                attrs.append((l1 / lp).numpy())
        return inps, refs, expls, attrs, confs

    @tc.no_grad()
    def _get_embeds(self, words, batch_size=1024):
        def encode_batch(ids, Hi, Ho):
            M, maxlen = [], max(map(len, ids))
            for _ in ids:
                M.append([1] * len(_) + [0] * (maxlen - len(_)))
                _.extend([self._tokenizer.eos_token_id] * (maxlen - len(_)))
            M = tc.tensor(M).float().unsqueeze(-1).cpu()
            ids = tc.tensor(ids).long()
            Ho.append((Eo[ids] * M).sum(axis=1) / (1e-9 + M.sum(axis=1)))
            Hi.append((Ei[ids] * M).sum(axis=1) / (1e-9 + M.sum(axis=1)))
        
        Ei = self._inp_embed.weight.cpu().float()
        Eo = self._out_embed.cpu().float()
        Hi, Ho, batchE = [], [], []
        for word in words:
            tokens = self._tokenizer.tokenize(" " + word)
            if tokens[0] in (u'Ġ', u'▁'):
                tokens = tokens[1:]
            batchE.append(self._tokenizer.convert_tokens_to_ids(tokens))
            if len(batchE) == batch_size:
                encode_batch(batchE, Hi, Ho)
                batchE.clear()
        if len(batchE) > 0:
            encode_batch(batchE, Hi, Ho)
        return tc.cat(Hi, axis=0), tc.cat(Ho, axis=0)

    def _get_weights(self, layer, wtype):
        assert isinstance(layer, int) and layer >= 0
        assert wtype in {"qk", "vo", "down"}

        @functools.cache
        def get_weight(l, w):
            name = format_llama_weight(l, w)
            weight = self._model.get_parameter(name).detach()
            if w == "down":
                return weight.T
            if w == "o":
                return weight.reshape(self._config.num_attention_heads,
                        self._config.hidden_size // self._config.num_attention_heads,
                        self._config.hidden_size)
            norm = self._model.get_parameter(format_llama_weight(layer, "norm")).detach()
            weight = weight * norm.unsqueeze(1)
            return weight.reshape(self._config.hidden_size,
                    self._config.num_attention_heads,
                    self._config.hidden_size // self._config.num_attention_heads,
                    ).permute(1, 0, 2)
        if wtype in {"qk", "vo"}:
            return get_weight(layer, wtype[0]), get_weight(layer, wtype[1])
        return get_weight(layer, wtype)
            
