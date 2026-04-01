import os
import argparse
from coevo.examples.science.oscillation_1 import Oscillation1TaskInfo, Oscillation1Evaluator
from coevo.examples.science.oscillation_2 import Oscillation2TaskInfo, Oscillation2Evaluator
from coevo.examples.science.bactgrow import BactGrowTaskInfo, BactGrowEvaluator
from coevo.examples.science.stress_strain import StressStrainTaskInfo, StressStrainEvaluator
from coevo.tasks.coevo_task import CoEvoTask
from coevo.core.coevo_interface import CoEvoInterface
from coevo.core.coevo_method import CoEvoMethod
from evotoolkit.tools import HttpsApi

TASK_MAP = {
    "oscillation_1": (Oscillation1TaskInfo, Oscillation1Evaluator),
    "oscillation_2": (Oscillation2TaskInfo, Oscillation2Evaluator),
    "bactgrow": (BactGrowTaskInfo, BactGrowEvaluator),
    "stress_strain": (StressStrainTaskInfo, StressStrainEvaluator)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='oscillation_1', choices=TASK_MAP.keys())
    parser.add_argument('--max_gen', type=int, default=97)
    parser.add_argument('--pop_size', type=int, default=2)
    args = parser.parse_args()

    task_info_cls, evaluator_cls = TASK_MAP[args.task]
    task_info = task_info_cls()
    evaluator = evaluator_cls(dataset_dir=f"./CoEvo/data/{args.task}/train.csv")

    task = CoEvoTask(task_info, evaluator)
    interface = CoEvoInterface(task)

    llm = HttpsApi(
        api_url=os.getenv("API_URL", "https://api.openai.com/v1"),
        key=os.getenv("API_KEY"),
        model=os.getenv("MODEL", "gpt-4o")
    )

    method = CoEvoMethod(
        interface=interface,
        output_path=f"./results/{args.task}",
        running_llm=llm,
        max_generations=args.max_gen,
        pop_size=args.pop_size
    )

    best = method.run()
    print(f"Best solution: {best.evaluation_res.score}")
