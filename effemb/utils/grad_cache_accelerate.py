from grad_cache import GradCache
from typing import List, Union, Callable, Any
from contextlib import nullcontext

from torch.cuda.amp import GradScaler, autocast

from torch import nn, Tensor

from grad_cache.context_managers import RandContext
import torch

class GradCacheAccelerate(GradCache):
    def __init__(
            self,
            accelerator,
            models: List[nn.Module],
            chunk_sizes: Union[int, List[int]],
            loss_fn: Callable[..., Tensor],
            split_input_fn: Callable[[Any, int], Any] = None,
            get_rep_fn: Callable[..., Tensor] = None,
            fp16: bool = False,
            scaler: GradScaler = None,
    ):
        super(GradCacheAccelerate, self).__init__(models, chunk_sizes, loss_fn, split_input_fn, get_rep_fn, fp16, scaler)
        self.accelerator = accelerator

    def build_cache(self, *reps: Tensor, **loss_kwargs) -> [List[Tensor], Tensor]:
        """
        Overwrite to let accelerate handle backward
        """
        reps = [r.detach().requires_grad_() for r in reps]
        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps, **loss_kwargs)

        if self.fp16:
            self.accelerator.backward(self.scaler.scale(loss))
        else:
            self.accelerator.backward(loss)

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                self.accelerator.backward(surrogate)
