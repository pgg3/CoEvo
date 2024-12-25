import os
import time
import numpy as np


import importlib
import warnings
import pandas as pd


from coevo.tasks.program_design import ProgramEvaluator
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

class BactGrowEvaluator(ProgramEvaluator):
    def __init__(self, dataset_dir, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_data = pd.read_csv(dataset_dir)
        self.evaluate_data.columns = ["b", "s", "temp", "pH", "db"]


    def evaluate_task(self, code_string, return_parameters=False, **kwargs):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Write code string into the temp module
                with open(os.path.join(ABS_PATH, "temp_module.py"), "w") as f:
                    f.write(code_string)

                # Create a new module object
                heuristic_module = importlib.import_module("coevo.examples.science.bactgrow.temp_module")
                eva = importlib.reload(heuristic_module)

                start_time = time.time()
                from scipy.optimize import minimize
                def loss(params):
                    y_pred = eva.equation(
                        self.evaluate_data["b"],
                        self.evaluate_data["s"],
                        self.evaluate_data["temp"],
                        self.evaluate_data["pH"],
                        params
                    )
                    return np.mean((y_pred - self.evaluate_data["db"]) ** 2)

                MAX_NPARAMS = 10
                PRAMS_INIT = [1.0] * MAX_NPARAMS

                loss_partial = lambda params: loss(params)
                result = minimize(loss_partial, [1.0] * MAX_NPARAMS, method='BFGS')

                # Return evaluation score
                optimized_params = result.x
                loss = result.fun

                if np.isnan(loss) or np.isinf(loss):
                    return [], "Error, the residual error is NaN or inf", False
                else:
                    if not return_parameters:
                        return [loss, time.time() - start_time], "The residual error between the output and the ground truth is {}".format(loss), True
                    else:
                        return [loss, time.time() - start_time, optimized_params], "The residual error between the output and the ground truth is {}".format(loss), True

        except Exception as e:
            return [], "Error:{}".format(str(e)), False

    def _evaluate_with_fixed_params(self,code_string, params):
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Write code string into the temp module
            with open(os.path.join(ABS_PATH, "temp_module.py"), "w") as f:
                f.write(code_string)

            # Create a new module object
            heuristic_module = importlib.import_module("coevo.examples.science.bactgrow.temp_module")
            eva = importlib.reload(heuristic_module)

            start_time = time.time()
            y_pred = eva.equation(
                self.evaluate_data["b"],
                self.evaluate_data["s"],
                self.evaluate_data["temp"],
                self.evaluate_data["pH"],
                params
            )

            return  [np.mean((y_pred - self.evaluate_data["db"]) ** 2), time.time() - start_time]