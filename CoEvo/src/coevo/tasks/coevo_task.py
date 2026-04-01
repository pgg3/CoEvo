from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import TaskSpec, EvaluationResult


class CoEvoTask(PythonTask):
    def __init__(self, task_info, evaluator, **kwargs):
        self.task_info = task_info
        self.evaluator = evaluator
        super().__init__(data=None, **kwargs)

    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="coevo_task",
            prompt=self.task_info.task_info + "\n\n" + self.task_info.program_template,
            modality="python"
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        fitness_list, error_msg, is_valid = self.evaluator.evaluate_task(candidate_code)

        if not is_valid:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": error_msg}
            )

        return EvaluationResult(
            valid=True,
            score=-fitness_list[0],
            additional_info={
                "mse": fitness_list[0],
                "time": fitness_list[1]
            }
        )
