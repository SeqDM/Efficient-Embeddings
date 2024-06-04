from abc import ABC, abstractmethod
from effemb.utils.metric_logging import Loggers


class AbsJob(ABC):
    def __init__(self, loggers: Loggers):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
