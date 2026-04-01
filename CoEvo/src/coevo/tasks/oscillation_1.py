import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import PythonTask


class Oscillation1Task(PythonTask):
    def __init__(self, dataset_dir: str, timeout_seconds: float = 30.0):
        self.evaluate_data = pd.read_csv(dataset_dir)
        self.evaluate_data.columns = ["x", "v", "a"]
        super().__init__(data=None, timeout_seconds=timeout_seconds)

    def build_python_spec(self, data) -> TaskSpec:
        task_description = (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "Complete the 'equation' function below, considering the physical meaning and relationships of inputs. "
            "Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator "
            "system with driving force, given data on position, and velocity. "
        )
        program_template = (
            "```python\n"
            "import numpy as np\n"
            "def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
            '    """ Mathematical function for acceleration in a damped nonlinear oscillator\n\n'
            "    Args:\n"
            "        x: A numpy array representing observations of current position.\n"
            "        v: A numpy array representing observations of velocity.\n"
            "        params: Array of numeric constants or parameters to be optimized, there are 10 params.\n\n"
            "    Return:\n"
            "        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n"
            '    """\n'
            "    dv = params[0] * x  +  params[1] * v +  + params[3]  # Example equation\n"
            "    return dv\n"
            "```\n"
        )
        return TaskSpec(
            name="oscillation_1",
            prompt=task_description + "\nProgram Template:\n" + program_template,
            modality="python",
            extras={
                "raw_task_description": task_description,
                "program_template": program_template,
            },
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                namespace: dict = {}
                exec(candidate_code, namespace)  # noqa: S102
                equation_fn = namespace["equation"]

                x = self.evaluate_data["x"].values
                v = self.evaluate_data["v"].values
                a = self.evaluate_data["a"].values

                start_time = time.time()

                def loss(params):
                    y_pred = equation_fn(x, v, params)
                    return np.mean((y_pred - a) ** 2)

                result = minimize(loss, np.array([1.0] * 10), method="BFGS")
                mse = float(result.fun)
                elapsed = time.time() - start_time

                if np.isnan(mse) or np.isinf(mse):
                    return EvaluationResult(
                        valid=False,
                        score=float("-inf"),
                        additional_info={"error": "residual error is NaN or inf"},
                    )

                fitness_string = f"The residual error between the output and the ground truth is {mse}"
                return EvaluationResult(
                    valid=True,
                    score=-mse,
                    additional_info={
                        "mse": mse,
                        "time": elapsed,
                        "fitness_list": [mse, elapsed],
                        "fitness_string": fitness_string,
                    },
                )
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": str(e)},
            )
