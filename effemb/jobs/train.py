import time
from os.path import basename, abspath, join

import os
import gin
import pickle
import peft
import torch
import torch.nn as nn
from transformers import AutoModel
from abc import abstractmethod


from effemb.jobs.abs import AbsJob
from effemb.utils.data import get_dataloader, Tokenizer, AVERAGE_BAAI_QUERY_TOKENS, AVERAGE_BAAI_POS_TOKENS
from effemb.utils.model_surgery import (freeze_first_k_transformer_layers,
                                        freeze_last_k_transformer_layers,
                                        only_bias_training,
                                        only_kth_transformer_layer)
from effemb.utils.grad_cache_accelerate import GradCacheAccelerate

from effemb.utils.pythia import get_original_lr, ALL_PYTHIA_MODELS


@gin.configurable
class TrainJob(AbsJob):
    def __init__(
        self,
        loggers,
        accelerator,
        loss,
        scheduler_class,
        checkpointing,
        embedder_class,
        datapath,
        optimizer_class,
        eval_job_class,
        tau,
        batch_size,
        checkpoint_freq,
        model_path,
        chunk_size=None,
        tokenizer_path = None,
        learning_rate=None, # If no lr is provided, we default to 1/10th of the original model's lr
        checkpoint_path=None,
        flop_limit = None,
        token_limit = None,
        training_steps = None,
        output_dir="result",
        use_neg=False
    ):
        self.loggers = loggers
        self.accelerator = accelerator
        self.loss = loss
        self.model_path = model_path
        self.learning_rate = get_original_lr(self.model_path) / 10 if learning_rate is None else learning_rate
        self.batch_size = batch_size

        self.flop_limit = flop_limit
        self.token_limit = token_limit
        self.training_steps = training_steps

        self.checkpoint_freq = checkpoint_freq
        self.checkpointing = checkpointing
        self.output_dir = output_dir
        self.tau = tau

        self.chunk_size = chunk_size

        self.datapath = datapath
        self.use_neg = use_neg

        assert model_path in ALL_PYTHIA_MODELS, \
        "The code is Pythia scaling suite specific."

        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path:
            self.loggers.log_message(f"Checkpoint path present: {self.checkpoint_path}")
            potential_checkpoint_dirs = []
            # Search for the latest checkpoint
            for d in os.listdir(self.checkpoint_path):
                if d.isnumeric():
                    potential_checkpoint_dirs.append(int(d))
            if potential_checkpoint_dirs:
                max_checkpoint_dir = str(max(potential_checkpoint_dirs))
                self.checkpoint_path = os.path.join(self.checkpoint_path, max_checkpoint_dir)
            self.loggers.log_message(f"Latest checkpoint: {self.checkpoint_path}")
            saved_step = int(basename(abspath(self.checkpoint_path)))
            self.step = saved_step
            self.loggers.log_message(f"Training step set to {self.step}")
            self.optimizer_checkpoint = join(self.checkpoint_path, 'optimizer')
            self.model_checkpoint = join(self.checkpoint_path, 'model.pt')
            data_file_offset_path = join(self.checkpoint_path, 'data_file_offset')
            with open(data_file_offset_path, 'r') as f:
                self.data_file_offset = int(f.read())
        else:
            self.step = 0
            self.optimizer_checkpoint = None
            self.model_checkpoint = None
            self.data_file_offset = 0

        self.loggers.log_property("resume step", self.step)

        self.model = self._build_model()
        self._model_surgery()
        self.datapath = datapath
        self.use_neg = use_neg
        # tokenizer's and model's path are the same
        self.tokenizer = Tokenizer(model_path)
        self.embedder = embedder_class(
            model=self.model,
            tokenizer=self.tokenizer,
            device=accelerator.device
        )
        self.optimizer = self._build_optimizer(optimizer_class)
        self.dataloader= get_dataloader(
            batch_size=self.batch_size,
            datapath=self.datapath,
            tokenizer=self.tokenizer,
            offset=self.data_file_offset,
            use_neg=self.use_neg
        )
        self.eval_job = eval_job_class(loggers=loggers)

        trainable_params = 0
        all_param = 0
        for n, param in self.model.named_parameters():
            if 'emb' in n:
                continue
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        self.loggers.log_property("parameter_count", float(all_param))
        self.loggers.log_property("trainable_count", float(trainable_params))

        if (self.flop_limit is not None) + \
           (self.token_limit is not None) + \
           (self.training_steps is not None) != 1:
            raise ValueError("Exactly one way of limiting the training duration should be provided.")

        if self.flop_limit:
            if self.use_neg:
                raise NotImplementedError()
            self.training_steps = self._approx_train_steps_from_flops(
                all_param,
                trainable_params,
                self.flop_limit
            )

        if self.token_limit:
            if self.use_neg:
                raise NotImplementedError()
            self.training_steps = self._approx_train_steps_from_tokens(self.token_limit)

        self.loggers.log_property("training steps", self.training_steps)

        self.scheduler = scheduler_class(
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            max_steps=self.training_steps,
            warmup_steps=self.training_steps // 10)


        # Prepare for distributed training
        (
            self.optimizer,
            self.scheduler,
            self.dataloader,
            self.embedder
        ) = accelerator.prepare(
            self.optimizer,
            self.scheduler,
            self.dataloader,
            self.embedder
        )

        # Prepare for gradien caching
        if self.chunk_size is not None:
            self.grad_cache = GradCacheAccelerate(
                models=[self.embedder, self.embedder],
                chunk_sizes=self.chunk_size,
                loss_fn=self.loss,
                accelerator=self.accelerator
            )
        else:
            self.grad_cache = None

    def _build_model(self):
        if self.model_checkpoint:
            model_spec = self.model_checkpoint
            self.loggers.log_message(
                f"Model loaded from the checkpoint file: {model_spec}")
        else:
            model_spec = self.model_path
            self.loggers.log_message(f"Model loaded from Hugging Face: {model_spec}")
        model = AutoModel.from_pretrained(model_spec)
        if self.checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        return model.to(self.accelerator.device)

    def _model_surgery(self):
        """
        Perform model surgery, like freezing layers, etc.
        """
        raise NotImplementedError

    def _build_optimizer(self, optimizer_class) -> torch.optim.Optimizer:
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        if self.optimizer_checkpoint:
            with open(self.optimizer_checkpoint, 'rb') as f:
                state_dict = pickle.load(f)
            optimizer.load_state_dict(state_dict)
            self.loggers.log_message(
                f"Optimizer state loaded from the file {self.optimizer_checkpoint}")
        return optimizer

    @abstractmethod
    def _approx_train_steps_from_flops(self, all_param, trainable_params, flop_limit):
        # various fine-tuning methods have different flop/train step dependencies
        raise NotImplementedError()

    def _approx_train_steps_from_tokens(self, token_limit):
        return token_limit / (AVERAGE_BAAI_POS_TOKENS + AVERAGE_BAAI_QUERY_TOKENS)

    def execute(self):

        for batch in self.dataloader:
            self.embedder.train()
            self.optimizer.zero_grad()

            if self.grad_cache is not None:
                # TODO implement possibility of using negatives
                c_loss: torch.Tensor = self.grad_cache(
                    batch['query'],
                    batch['pos'],
                    tau=self.tau,
                    cross_entropy_loss=nn.CrossEntropyLoss(),
                )

            else:
                x_tensor = self.embedder.forward(batch['query'])
                y_tensor = self.embedder.forward(batch['pos'])
                if self.use_neg:
                    # neg_tensor = self.embedder.forward(batch['neg'][0])
                    raise NotImplementedError
                else:
                    neg_tensor = None

                c_loss: torch.Tensor = self.loss(
                    x_tensor,
                    y_tensor,
                    self.tau,
                    nn.CrossEntropyLoss(),
                    negatives=neg_tensor
                )
                self.accelerator.backward(c_loss)

            self.loggers.log_scalar("loss", self.step, c_loss)
            self.loggers.log_scalar("forward_tokens", self.step,
                                    float(self.embedder.get_num_encoded_tokens()))
            current_lr = self.scheduler.set_lr(step=self.step)
            self.loggers.log_scalar("learning_rate", self.step, current_lr)

            self.optimizer.step()

            if self.accelerator.is_main_process:
                with torch.no_grad():
                    if (self.step // self.batch_size) % self.checkpoint_freq == 0 \
                        or self.step + len(batch['query']['input_ids']) > self.training_steps:
                        self.embedder.eval()
                        eval_start = time.time()
                        # self.eval_job.execute(step=self.step, embedder=self.embedder)
                        eval_end = time.time()

                        self.loggers.log_scalar("eval_time", self.step, eval_end - eval_start)

                        # saving model and meta-data
                        model_checkpoint_path= f"{self.output_dir}/{self.step}/model.pt"
                        optimizer_checkpoint_path= f"{self.output_dir}/{self.step}/optimizer"
                        data_file_offset_path= f"{self.output_dir}/{self.step}/data_file_offset"
                        unwrapped_model = self.accelerator.unwrap_model(self.embedder)
                        unwrapped_model.save_pretrained(model_checkpoint_path)
                        optimizer_checkpoint = self.optimizer.state_dict()
                        with open(optimizer_checkpoint_path, 'wb') as f:
                            pickle.dump(optimizer_checkpoint, f)
                        data_file_offset = self.dataloader.dataset.current_position()
                        with open(data_file_offset_path, 'w') as f:
                            f.write(f'{data_file_offset}\n')

            self.step += self.batch_size
            if self.step > self.training_steps:
                self.loggers.log_message(
                    f"Max training steps ({self.training_steps}) reached. Training stopped."
                )
                if self.accelerator.is_main_process:
                    with torch.no_grad():
                        self.embedder.eval()
                        eval_start = time.time()
                        self.eval_job.execute(step=self.step, embedder=self.embedder)
                        eval_end = time.time()

                        self.loggers.log_scalar("eval_time", self.step, eval_end - eval_start)

                        # saving model and meta-data
                        model_checkpoint_path= f"{self.output_dir}/{self.step}/model.pt"
                        optimizer_checkpoint_path= f"{self.output_dir}/{self.step}/optimizer"
                        data_file_offset_path= f"{self.output_dir}/{self.step}/data_file_offset"
                        unwrapped_model = self.accelerator.unwrap_model(self.embedder)
                        unwrapped_model.save_pretrained(model_checkpoint_path)
                        optimizer_checkpoint = self.optimizer.state_dict()
                        with open(optimizer_checkpoint_path, 'wb') as f:
                            pickle.dump(optimizer_checkpoint, f)
                        data_file_offset = self.dataloader.dataset.current_position()
                        with open(data_file_offset_path, 'w') as f:
                            f.write(f'{data_file_offset}\n')
                return

@gin.configurable
class GenericTrain(TrainJob):
    def __init__(self, loggers, accelerator, loss, **super_kwargs):
        super(GenericTrain, self).__init__(loggers, accelerator, loss, **super_kwargs)

    def _model_surgery(self):
        return

    def _approx_train_steps_from_flops(self, all_param, trainable_params, flop_limit):
        return flop_limit / (6 * all_param * (AVERAGE_BAAI_POS_TOKENS + AVERAGE_BAAI_QUERY_TOKENS))


@gin.configurable
class SingleTrain(TrainJob):
    def __init__(self, loggers, accelerator, loss, active_layer, **super_kwargs):
        self.active_layer = active_layer
        super(SingleTrain, self).__init__(loggers, accelerator, loss, **super_kwargs)

    def _model_surgery(self):
        self.model = only_kth_transformer_layer(self.active_layer, self.model)

    def _approx_train_steps_from_flops(self, all_param, trainable_params, flop_limit):
        return flop_limit / ((4 * all_param + 2 * trainable_params) * (AVERAGE_BAAI_POS_TOKENS + AVERAGE_BAAI_QUERY_TOKENS))


@gin.configurable
class SuffixTrain(TrainJob):
    def __init__(self, loggers, accelerator, loss, active_layers=1, **super_kwargs):
        self.active_layers = active_layers
        super(SuffixTrain, self).__init__(loggers, accelerator, loss, **super_kwargs)
        self.loggers.log_property("active layers", self.active_layers)

    def _model_surgery(self):
        if 0 < self.active_layers < 1:
            self.active_layers = int(self.active_layers * len(self.model.layers))
        else:
            self.active_layers = self.active_layers

        freeze_layers = len(self.model.layers) - self.active_layers

        self.model.embed_in.weight.requires_grad = False

        freeze_first_k_transformer_layers(freeze_layers, self.model)

    def _approx_train_steps_from_flops(self, all_param, trainable_params, flop_limit):
        return flop_limit / ((4 * trainable_params + 2 * all_param) * (AVERAGE_BAAI_POS_TOKENS + AVERAGE_BAAI_QUERY_TOKENS))


@gin.configurable
class PrefixTrain(TrainJob):
    def __init__(self, loggers, accelerator, loss, active_layers, **super_kwargs):
        self.active_layers = active_layers
        super(PrefixTrain, self).__init__(loggers, accelerator, loss, **super_kwargs)

    def _model_surgery(self):
        freeze_layers = len(self.model.layers) - self.active_layers
        freeze_last_k_transformer_layers(freeze_layers, self.model)


@gin.configurable
class PEFTTrain(TrainJob):
    def __init__(self, loggers, accelerator, loss, peft_config_class, **super_kwargs):

        self.peft_config = peft_config_class(
            task_type=peft.TaskType.FEATURE_EXTRACTION,
            inference_mode=False
        )
        assert isinstance(self.peft_config, peft.LoraConfig)

        super(PEFTTrain, self).__init__(loggers, accelerator, loss, **super_kwargs)

    def _approx_train_steps_from_flops(self, all_param, trainable_params, flop_limit):
        return flop_limit / ((4 * all_param + 2 * trainable_params) * (AVERAGE_BAAI_POS_TOKENS + AVERAGE_BAAI_QUERY_TOKENS))

    def _model_surgery(self):
        self.model = peft.get_peft_model(self.model, self.peft_config)


@gin.configurable
class OnlyBiasTrain(TrainJob):
    def __init__(self, loggers, accelerator, loss, **super_kwargs):
        super(OnlyBiasTrain, self).__init__(loggers, accelerator, loss, **super_kwargs)

    def _approx_train_steps_from_flops(self, all_param, trainable_params, flop_limit):
        return flop_limit / ((4 * all_param + 2 * trainable_params) * (AVERAGE_BAAI_POS_TOKENS + AVERAGE_BAAI_QUERY_TOKENS))

    def _model_surgery(self):
        only_bias_training(self.model)
