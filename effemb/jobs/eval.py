import os
import time
from abc import abstractmethod
from collections import defaultdict

import gin
import torch
from mteb import MTEB
from transformers import AutoModel

from effemb.embeddings import AbsEmbedding
from effemb.jobs.abs import AbsJob
from effemb.utils.metric_logging import Loggers
from effemb.utils.metrics import extract_main_metrics
from effemb.utils.data import Tokenizer
from os.path import basename, abspath, join

# import paramiko
# from scp import SCPClient

class AbsEvalJob(AbsJob):
    @abstractmethod
    def __init__(self, loggers: Loggers):
        raise NotImplementedError()

    def execute():
        raise NotImplementedError()


@gin.configurable
class MTEBJob(AbsJob):
    # Passing embedder class to be able to use gin config
    def __init__(self, loggers: Loggers,
                 embedder_class: AbsEmbedding = None,
                 tasks=[]):
        self.loggers = loggers

        self.set_eval_tasks(tasks)


        if embedder_class is not None:
            self.embedder = embedder_class()
        else:
            self.embedder = None

    def set_eval_tasks(self, tasks):
        if isinstance(tasks, str):
            if os.path.exists(tasks):
                with open(tasks) as f:
                    self.tasks = f.read().splitlines()
            else:  # tasks is a name of one task only
                self.tasks = [tasks]
        elif isinstance(tasks, list):
            self.tasks = tasks
        else:
            raise AssertionError("Invalid task specification.")

    @gin.configurable
    def execute(self, output_folder='result', step=0, embedder=None):

        assert embedder is None or self.embedder is None

        if embedder is None:
            embedder = self.embedder

        evaluation = MTEB(tasks=self.tasks)

        task_descriptions = {}

        for task in evaluation.tasks:
            task_descriptions[task.description["name"]] = task.description

        results = evaluation.run(embedder, output_folder=output_folder)

        metrics = {}
        leaderboard_metrics = {}

        if len(results) == 0:
            print("No tasks were run")
            return

        # Get the main metrics from all the datasets
        for dataset in results:
            metrics[dataset], leaderboard_metrics[dataset] = extract_main_metrics(
                task_descriptions[dataset], results[dataset]
            )

        # Log all the main metrics from all the datasets in the evaluation
        for k, v in metrics.items():
            for vk, vv in v.items():
                self.loggers.log_scalar(f"{k}/{vk}", step, vv)  # No steps in evaluation

        # Log the leaderboard scores, together with their accumulated by task category versions
        all_scores = []
        category_score = defaultdict(list)

        for k, v in leaderboard_metrics.items():
            if v is None:
                continue
            all_scores.append(v.score)
            category_score[v.category].append(v)

        for category, scores in category_score.items():
            for score in scores:
                self.loggers.log_scalar(
                    f"leaderboard/{category}/{score.name}", step, score.score
                )

            if len(scores) > 0:
                self.loggers.log_scalar(
                    f"leaderboard/{category}/average",
                    step,
                    sum([score.score for score in scores]) / len(scores),
                )

        if len(all_scores) > 0:
            self.loggers.log_scalar(
                f"leaderboard/average", step, sum(all_scores) / len(all_scores)
            )


@gin.configurable
class StandaloneEvalJob(AbsEvalJob):
    def __init__(
        self,
        loggers,
        accelerator,
        checkpointing,
        embedder_class,
        eval_job_class,
        model_type,
        tokenizer_path,
        output_dir="result"
    ):
        self.loggers = loggers
        self.accelerator = accelerator
        self.model_type = model_type
        self.checkpointing = checkpointing
        self.output_dir = output_dir
        model = self._build_model()
        tokenizer = Tokenizer(tokenizer_path)
        self.embedder = embedder_class(
            model=model,
            tokenizer=tokenizer,
            device=accelerator.device
        )

        self.eval_job = eval_job_class(loggers=loggers)

    def execute(self):
        if self.accelerator.is_main_process:
            with torch.no_grad():
                self.embedder.eval()
                eval_start = time.time()
                self.eval_job.execute(step=0, embedder=self.embedder)
                eval_end = time.time()
                self.loggers.log_scalar("eval_time", 0, eval_end - eval_start)

    def _build_model(self):
        # TODO: enable reloading for model
        potential_checkpoint_dirs = []
        # Search for the latest checkpoint
        print(self.model_type)
        if os.path.exists(self.model_type):
            for d in os.listdir(self.model_type):
                if d.isnumeric():
                    potential_checkpoint_dirs.append(int(d))
            if potential_checkpoint_dirs:
                max_checkpoint_dir = str(max(potential_checkpoint_dirs))
                self.model_type = os.path.join(self.model_type, max_checkpoint_dir)
            self.loggers.log_message(f"Latest checkpoint: {self.model_type}")
            self.model_checkpoint = join(self.model_type, 'model.pt')

            model = AutoModel.from_pretrained(self.model_checkpoint)
        else:
            model = AutoModel.from_pretrained(self.model_type)


        if self.checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        return model.to(self.accelerator.device)

@gin.configurable
class StandaloneRemoteEvalJob(AbsEvalJob):
    def __init__(
        self,
        loggers,
        accelerator,
        checkpointing,
        embedder_class,
        eval_job_class,
        host,
        user,
        remote_model_path,
        tokenizer_path,
        output_dir="result",
        model_type="result",
    ):
        def createSSHClient(server, port, user):
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(server, port, user)
            return client

        self.loggers = loggers
        self.accelerator = accelerator
        self.model_type = model_type
        self.checkpointing = checkpointing
        self.output_dir = output_dir
        ssh = createSSHClient(host, 22, user)
        scp = SCPClient(ssh.get_transport())

        assert remote_model_path.split('/')[-1] == 'result'

        print(scp.get(remote_model_path, recursive=True))

        model = self._build_model()
        tokenizer = Tokenizer(tokenizer_path)
        self.embedder = embedder_class(
            model=model,
            tokenizer=tokenizer,
            device=accelerator.device
        )

        self.eval_job = eval_job_class(loggers=loggers)


    def execute(self):
        if self.accelerator.is_main_process:
            with torch.no_grad():
                self.embedder.eval()
                eval_start = time.time()
                self.eval_job.execute(step=0, embedder=self.embedder)
                eval_end = time.time()
                self.loggers.log_scalar("eval_time", 0, eval_end - eval_start)

    def _build_model(self):
        # TODO: enable reloading for model
        potential_checkpoint_dirs = []
        # Search for the latest checkpoint
        for d in os.listdir(self.model_type):
            if d.isnumeric():
                potential_checkpoint_dirs.append(int(d))
        if potential_checkpoint_dirs:
            max_checkpoint_dir = str(max(potential_checkpoint_dirs))
            self.model_type = os.path.join(self.model_type, max_checkpoint_dir)
        self.loggers.log_message(f"Latest checkpoint: {self.model_type}")
        self.model_checkpoint = join(self.model_type, 'model.pt')

        model = AutoModel.from_pretrained(self.model_checkpoint)

        if self.checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        return model.to(self.accelerator.device)
