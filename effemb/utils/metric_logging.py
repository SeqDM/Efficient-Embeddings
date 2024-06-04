import sys
import atexit
import os
import yaml
import neptune
from abc import ABC, abstractmethod


class AbsLogger(ABC):
    @abstractmethod
    def log_scalar(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def log_property(self, name, value):
        raise NotImplementedError


class StdoutLogger(AbsLogger):
    """Logs to standard output."""

    def __init__(self, file=sys.stdout):
        self.file = file

    def log_scalar(self, name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        #   2137 | loss:                      1.0e-5
        if 0 < value < 1e-2:
            print(
                "{:>6} | {:64}{:>9.1e}".format(step, name + ":", value), file=self.file
            )
        else:
            print(
                "{:>6} | {:64}{:>9.3f}".format(step, name + ":", value), file=self.file
            )

    def log_property(self, name, value):
        # Not supported in this logger.
        pass

    def log_message(self, message):
        print(message, file=self.file)

def configure_neptune_run(specification):
    """Configures the Neptune experiment, then returns the Neptune logger."""
    # git_info = specification.get("git_info", None)
    # if git_info:
    #     git_info.commit_date = datetime.datetime.now()

    # Set pwd property with path to experiment.
    properties = {"pwd": os.getcwd()}

    run = neptune.init_run(
        project=specification["project"],
        name=specification["name"],
        tags=specification["tags"],
    )

    run["parameters"] = specification["parameters"]
    run["properties"] = properties

    return run


class NeptuneLogger(AbsLogger):
    def __init__(self, specification=None):
        self.run = configure_neptune_run(
            specification
        )

        atexit.register(self.stop)

    def log_scalar(self, name, step, value):
        self.run[name].append(value)

    def log_property(self, name, value):
        self.run[name] = value

    def log_message(self, message):
        pass

    def stop(self):
        self.run.stop()


class Loggers:
    def __init__(self):
        self.loggers = []

    def register_logger(self, logger: AbsLogger):
        self.loggers.append(logger)

    def log_scalar(self, name, step, value):
        for logger in self.loggers:
            logger.log_scalar(name, step, value)

    def log_property(self, name, value):
        for logger in self.loggers:
            logger.log_property(name, value)

    def log_parameters(self, parameters):
        for logger in self.loggers:
            logger.log_parameters(parameters)

    def log_message(self, message):
        for logger in self.loggers:
            logger.log_message(message)
