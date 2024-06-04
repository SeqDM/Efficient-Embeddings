#!/bin/env python3

import accelerate
import torch
import peft
from gin import parse_config_file, external_configurable

import effemb.embeddings
import effemb.jobs.eval
import effemb.losses
import effemb.schedulers
from effemb.jobs.train import PEFTTrain
from effemb.utils import metric_logging

loggers = metric_logging.Loggers()
loggers.register_logger(metric_logging.StdoutLogger())
accelerator = accelerate.Accelerator()

external_configurable(torch.optim.AdamW, module='torch.optim')
external_configurable(peft.LoraConfig, module='peft')
parse_config_file('experiments/train_and_eval_lora.gin')

job = PEFTTrain(loggers, accelerator)
job.execute()