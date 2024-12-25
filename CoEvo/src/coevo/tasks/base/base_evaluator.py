from abc import abstractmethod

class Evaluator:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def evaluate_task(self, **kwargs):
        raise NotImplementedError