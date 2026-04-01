import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import PythonTask


class BactGrowTask(PythonTask):
    def __init__(self, dataset_dir: str, timeout_seconds: float = 30.0):
        self.evaluate_data = pd.read_csv(dataset_dir)
        self.evaluate_data.columns = ["b", "s", "temp", "pH", "db"]
        super().__init__(data=None, timeout_seconds=timeout_seconds)

    def build_python_spec(self, data) -> TaskSpec:
        task_description = (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "Complete the 'equation' function below, considering the physical meaning and relationships of inputs. "
            "Find the mathematical function skeleton that represents E. Coli bacterial growth rate, "
            "given data on population density, substrate concentration, temperature, and pH level. "
        )
        program_template = (
            "```python\n"
            "import numpy as np\n"
            "def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
            '    """ Mathematical function for bacterial growth rate\n\n'
            "    Args:\n"
            "        b: A numpy array representing observations of population density of the bacterial species.\n"
            "        s: A numpy array representing observations of substrate concentration.\n"
            "        temp: A numpy array representing observations of temperature.\n"
            "        pH: A numpy array representing observations of pH level.\n"
            "        params: Array of numeric constants or parameters to be optimized, there are 10 params.\n\n"
            "    Return:\n"
            "        A numpy array representing bacterial growth rate as the result of applying the mathematical function to the inputs.\n"
            '    """\n'
            "    grow_rate = params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]  # Example equation\n"
            "    return grow_rate\n"
            "```\n"
        )
        return TaskSpec(
            name="bactgrow",
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

                b = self.evaluate_data["b"].values
                s = self.evaluate_data["s"].values
                temp = self.evaluate_data["temp"].values
                ph = self.evaluate_data["pH"].values
                db = self.evaluate_data["db"].values

                start_time = time.time()

                def loss(params):
                    y_pred = equation_fn(b, s, temp, ph, params)
                    return np.mean((y_pred - db) ** 2)

                result = minimize(loss, [1.0] * 10, method="BFGS")
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
