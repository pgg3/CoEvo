from ..base import Evaluator

class ProgramEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_task(self, code_string, **kwargs):
        raise NotImplementedError