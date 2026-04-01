"""Main entry point for CoEvo runs.

Supports two modes:
  - eoh:   Use evotoolkit's built-in EoH algorithm with EoHPythonInterface
  - coevo: Use the full CoEvo algorithm
"""

import argparse
import os

from evotoolkit.evo_method.eoh import EoH
from evotoolkit.task.python_task import EoHPythonInterface
from evotoolkit.tools import HttpsApi

from coevo.core.coevo_interface import CoEvoInterface
from coevo.core.coevo_method import CoEvoMethod
from coevo.tasks.bactgrow import BactGrowTask
from coevo.tasks.oscillation_1 import Oscillation1Task
from coevo.tasks.oscillation_2 import Oscillation2Task
from coevo.tasks.stress_strain import StressStrainTask

TASK_MAP = {
    "oscillation_1": Oscillation1Task,
    "oscillation_2": Oscillation2Task,
    "bactgrow": BactGrowTask,
    "stress_strain": StressStrainTask,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="oscillation_1", choices=list(TASK_MAP.keys()))
    parser.add_argument("--mode", type=str, default="coevo", choices=["eoh", "coevo"])
    parser.add_argument("--max_gen", type=int, default=97)
    parser.add_argument("--pop_size", type=int, default=2)
    args = parser.parse_args()

    task_cls = TASK_MAP[args.task]
    task = task_cls(dataset_dir=f"./CoEvo/data/{args.task}/train.csv")

    llm = HttpsApi(
        api_url=os.getenv("API_URL", "https://api.openai.com/v1"),
        key=os.getenv("API_KEY", ""),
        model=os.getenv("MODEL", "gpt-4o"),
    )

    if args.mode == "eoh":
        interface = EoHPythonInterface(task)
        method = EoH(
            interface=interface,
            running_llm=llm,
            output_path=f"./results/{args.task}_eoh",
            max_generations=args.max_gen,
            pop_size=args.pop_size,
        )
    else:
        interface = CoEvoInterface(
            task,
            num_idea=[6, 3, 3],
            rep_list=[{"name": "Python Code", "definition": "Runnable python code implementation."}],
            rep_use_name="Python Code",
        )
        method = CoEvoMethod(
            interface=interface,
            running_llm=llm,
            output_path=f"./results/{args.task}_coevo",
            max_generations=args.max_gen,
            pop_size=args.pop_size,
        )

    best = method.run()
    if best is not None and best.evaluation_res is not None:
        print(f"Best solution score: {best.evaluation_res.score}")
    else:
        print("No valid solution found.")
