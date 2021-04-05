from typing import List, Callable
from collections import defaultdict

from objax.module import Module, ModuleList
from objax.typing import JaxArray
from objax.util import class_name
from objax.variable import TrainRef, TrainVar, VarCollection

import jax.numpy as jn


class SAM(Module):

    def __init__(self, vc: VarCollection, base_optimizer: Callable, **kwargs):
        self.train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
        self.base_optimizer = base_optimizer(vc, **kwargs)
        self.state = defaultdict(dict)

    def first_step(self, rho: float, grads: List[JaxArray]):
        assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'
        # Create empty state dict
        self.state = defaultdict(dict)
        # norm grads
        grad_norm = self._grad_norm(grads)
        # create a scale factor
        scale = rho / (grad_norm + 1e-12)

        # loop through grads and params
        for g, p in zip(grads, self.train_vars):
            e_w = g * scale
            p.value = jn.add(p.value, e_w)
            self.state[str(p.ref)]["e_w"] = e_w

    def second_step(self, lr, grads):
        for g, p in zip(grads, self.train_vars):
            p.value = jn.subtract(p.value, self.state[str(p.ref)]["e_w"])

        self.base_optimizer(lr, grads)

    def _grad_norm(self, grads):
        norm = jn.linalg.norm(jn.stack([jn.linalg.norm(g) for g in grads]))
        return norm

    def __repr__(self):
        return f'{class_name(self)}()'
