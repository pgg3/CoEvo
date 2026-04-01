"""Main entry point for CoEvo runs.

Supports two modes:
  - eoh:   Use evotoolkit's built-in EoH algorithm with EoHPythonInterface
  - coevo: Use the full CoEvo algorithm
"""

import argparse
import json
import os

from evotoolkit.evo_method.eoh import EoH
from evotoolkit.task.python_task import EoHPythonInterface
from evotoolkit.tools import HttpsApi

from coevo.core.coevo_interface import CoEvoInterface
from coevo.core.coevo_method import CoEvoMethod
from coevo.core.summarizer import CoEvoSummarizer
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
    # Resolve data path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task = task_cls(dataset_dir=os.path.join(script_dir, "data", args.task, "train.csv"))

    # Load LLM config from model_config.json, fall back to env vars
    config_path = os.path.join(script_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_config = json.load(f)
        api_url = model_config["host"] + model_config["url"]
        api_key = model_config["key"]
        api_model = model_config["model"]
        api_timeout = model_config.get("timeout", 120)
    else:
        api_url = os.getenv("API_URL", "https://api.openai.com/v1/chat/completions")
        api_key = os.getenv("API_KEY", "")
        api_model = os.getenv("MODEL", "gpt-4o")
        api_timeout = int(os.getenv("API_TIMEOUT", "120"))

    llm = HttpsApi(
        api_url=api_url,
        key=api_key,
        model=api_model,
        timeout=api_timeout,
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
            num_idea=[3, 3],
            rep_list=[
                {"name": "Natural Language English", "definition": "Verbal descriptions of the solution."},
                {"name": "Python Code", "definition": "Runnable python code implementation."},
                {"name": "Mathematical Formula", "definition": "Mathematical formula or equation."},
            ],
            rep_use_name="Python Code",
        )

        # Second LLM instance for summarizer (v1 used two separate LLM instances)
        summarizer_llm = HttpsApi(
            api_url=api_url,
            key=api_key,
            model=api_model,
            timeout=api_timeout,
        )

        summarizer = CoEvoSummarizer(
            prompt_builder=interface.prompt_builder,
            llm=summarizer_llm,
            pool_size=100,
            num_idea_to_return=5,
            cluster_summary=True,
            tokenizer_path="openai-community/gpt2",
        )

        method = CoEvoMethod(
            interface=interface,
            running_llm=llm,
            output_path=f"./results/{args.task}_coevo",
            max_generations=args.max_gen,
            pop_size=args.pop_size,
            num_init_per_gen=6,
            use_summarizer=True,
            summarizer=summarizer,
        )

    best = method.run()
    if best is not None and best.evaluation_res is not None:
        print(f"Best solution score: {best.evaluation_res.score}")
    else:
        print("No valid solution found.")
