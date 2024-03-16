# Author: Xuansheng Wu (wuxsmail@163.com)
# Last Modify: 2024-01-11
# Description: hooks to collect intermediate outcomes of transformers.
from dataclasses import dataclass
from functools import partial, wraps
from bytecode import Bytecode, Instr

import transformers.models.gpt2.modeling_gpt2 as gpt2
import transformers.models.llama.modeling_llama as llama

import torch as tc


KEY = "__hook_cache"


@dataclass
class _Cache:
    LayerID: int
    @property
    def named_parameters(self):
        params = {}
        for name in sorted(self.__dataclass_fields__):
            if name != "LayerID":
                params[name] = getattr(self, name)
        return params

    def clear(self):
        for name, var in self.named_parameters.items():
            setattr(self, name, None)

    def zero_grad(self):
        for name, var in self.named_parameters.items():
            if var.grad is not None:
                var.grad = None

    def retain_grad(self):
        for name, var in self.named_parameters.items():
            if isinstance(var, tc.Tensor) and var.requires_grad:
                var.retain_grad()

    def check_type(self):
        for name, var in self.named_parameters.items():
            assert isinstance(var, tc.Tensor), "variable ``%s`` is not torch.Tensor!" % name


@dataclass
class MLPCache(_Cache):
    """Given MLP: y = W2(f(W1X)), we cache inputs = X, activate = f(W1X), outputs: y"""
    inputs: tc.Tensor = None
    hiddens: tc.Tensor = None
    activates: tc.Tensor = None
    outputs: tc.Tensor = None
    weights: tc.Tensor = None

    def collect_states(self):
        states = (self.inputs, self.hiddens,
                  self.activates, self.outputs)
        for each in states:
            assert isinstance(each, tc.Tensor)
        return states


class HookWrapper:

    @staticmethod
    def auto(forward_func, *var_names):
        """automatically collect your interested internal variables"""
        assert len(var_names) > 0
        code = Bytecode.from_code(forward_func.__code__)
        code[-1:-1] = [Instr("LOAD_FAST", var) for var in var_names]
        code.append(Instr("BUILD_TUPLE", len(temp)))
        forward_func.__code__ = code.to_code()
        
        @wraps(forward_func)
        def cached_forward(self, *args, **kwrds):
            if not hasattr(self, KEY):
                return forward_func(self, *args, **kwrds)
            cache = getattr(self, KEY)
            cache.clear()
            outputs, var_values = forward_func(self, *args, **kwrds) 
            for name, value in zip(var_names, var_values):
                setattr(cache, name, value)
            cache.check_type()
            cache.retain_grad()
            return outputs
        return cached_forward

    @staticmethod
    def manual(forward_func):
        """manually rewrite the forward function to collect variables you are interested in"""
        @wraps(forward_func)
        def cached_forward(self, *args, **kwrds):
            if not hasattr(self, KEY):
                return forward_func(self, *args, **kwrds)
            cache = getattr(self, KEY)
            cache.clear()
            outputs = forward_func(self, *args, **kwrds)
            cache.check_type()
            cache.retain_grad()
            return outputs
        return cached_forward


class _HookController:
    def __init__(self, model, target_block, cache_type):
        self._model = model
        self._target = target_block
        self._caches = {}
        self._modules = []

        for name, layer in self:
            layer_cache = cache_type(len(self._caches))
            setattr(layer, KEY, layer_cache)
            self._modules.append(name)
            self._caches[name] = layer_cache
        print("Target Block: %s | Cache Type: %s | Numbers: %d" %
              (target_block, cache_type, len(self._caches)))

    def __iter__(self):
        for name, layer in self._model.named_modules():
            if isinstance(layer, self._target):
                yield name, layer


class MLPHookController(_HookController):
    def __init__(self, model, target_block):
        _HookController.__init__(self, model, target_block, MLPCache)

    def collect_states(self):
        states = {}
        for l, (name, layer) in enumerate(self, 1):
            layer_states = self._caches[name].collect_states()
            states["Layer%d" % l] = (layer_states[0], layer_states[1].grad)
        return states

    def collect_weight_grads(self):
        grads = {}
        for l, (name, layer) in enumerate(self, 1):
            weight = self._caches[name].weights
            grads["Layer%d" % l] = (weight, weight.grad)
        return grads
    
    @classmethod
    def GPT2(cls, model):
        return cls(model, gpt2.GPT2MLP)

    @classmethod
    def LLaMA(cls, model):
        return cls(model, llama.LlamaMLP)


@HookWrapper.manual
def custom_LlamaMLP(self, x):
    if not hasattr(self, KEY):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    cache = getattr(self, KEY)
    cache.weights = self.gate_proj.weight
    #print(f"weights: {cache.weights.shape}")
    cache.inputs = x
    #print(f"inputs: {cache.inputs.shape}")
    cache.hiddens = self.gate_proj(x)
    #print(f"hiddens: {cache.hiddens.shape}")
    cache.activates = self.act_fn(cache.hiddens) * self.up_proj(x) 
    #print(f"activates: {cache.activates.shape}")
    cache.outputs = self.down_proj(cache.activates)
    #print(f"outputs: {cache.outputs.shape}")
    return cache.outputs


@HookWrapper.manual
def custom_GPT2MLP(self, hidden_states):
    if not hasattr(self, KEY):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    cache = getattr(self, KEY)
    cache.weights = self.c_fc.weight.T
    #print(f"weights: {cache.weights.shape}")
    cache.inputs = hidden_states
    #print(f"inputs: {cache.inputs.shape}")
    cache.hiddens = self.c_fc(cache.inputs)
    #print(f"hiddens: {cache.hiddens.shape}")
    cache.activates = self.act(cache.hiddens)
    #print(f"activates: {cache.activates.shape}")
    cache.outputs = self.dropout(self.c_proj(cache.activates))
    #print(f"outputs: {cache.outputs.shape}")
    return cache.outputs


llama.LlamaMLP.forward = custom_LlamaMLP
gpt2.GPT2MLP.forward = custom_GPT2MLP


if __name__ == "__main__":
    from generator import Generator

    generator = Generator("gpt2", device="cpu")
    hooker = MLPHookController.GPT2(generator._model)
    logits = generator.forward(["this is a sentence for"]).logits
    logits[:, -1, generator._tokenizer.convert_tokens_to_ids(["us"])[0]].backward()
    print(logits.shape, logits[0])
    for layer, (inputs, hiddens) in hooker.collect_states().items():
        print(layer, inputs.shape, hiddens.shape)

    for layer, (weight, gradient) in hooker.collect_weight_grads().items():
        print(layer, weight.shape, gradient.shape)
