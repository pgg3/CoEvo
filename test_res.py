import os
import json
import re
import argparse
import time

import numpy as np

from coevo.examples.science.oscillation_1 import Oscillation1TaskInfo, Oscillation1Evaluator
from coevo.examples.science.oscillation_2 import Oscillation2TaskInfo, Oscillation2Evaluator
from coevo.examples.science.bactgrow import BactGrowTaskInfo, BactGrowEvaluator
from coevo.examples.science.stress_strain import StressStrainTaskInfo, StressStrainEvaluator

EVAL_DICT = {
    "oscillation_1": Oscillation1Evaluator,
    "oscillation_2": Oscillation2Evaluator,
    "bactgrow": BactGrowEvaluator,
    "stress_strain": StressStrainEvaluator
}

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# TOKENIZER_PATH = os.path.join(ABS_PATH, "gpt2")
TOKENIZER_PATH = "openai-community/gpt2"

if __name__ == '__main__':
    res_format = ".4e"
    parser = argparse.ArgumentParser(description="Science Discovery Experiment.")
    parser.add_argument(
        '--problem', type=str, choices=['oscillation_1', 'oscillation_2', 'bactgrow', 'stress_strain'],
        default='oscillation_1', help='Choose the problem to solve.'
    )
    parser.add_argument(
        '--paper', type=str, choices=['coevo', 'llm_sr'],
        default='llm_sr', help='Choose the problem to solve.'
    )
    parser.add_argument(
        '--model', type=str, choices=['gpt35', 'gpt4'],
        default='gpt35', help='Choose the problem to solve.'
    )

    ARGS = parser.parse_args()
    TRAIN_DATA_PATH = os.path.join(ABS_PATH, "data", ARGS.problem, "train.csv")  # Path to store victim models
    TEST_ID_DATA_PATH = os.path.join(ABS_PATH, "data", ARGS.problem, "test_id.csv")
    TEST_OOD_DATA_PATH = os.path.join(ABS_PATH, "data", ARGS.problem, "test_ood.csv")
    solution_path = os.path.join(ABS_PATH, "paper_res", ARGS.paper, ARGS.problem, f"{ARGS.model}.json")

    with open(solution_path, "r") as f:
        solution_data = json.load(f)

    if ARGS.paper == "coevo":
        code_string = solution_data[-1]["Solutions"]["Python Code"]
        contained_fit = solution_data[-1]['fitness_list'][0]
    elif ARGS.paper == "llm_sr":
        code_string = f"import numpy as np\n\n{solution_data['function']}"
        contained_fit = -solution_data["score"]

    train_evaluator = EVAL_DICT[ARGS.problem](dataset_dir=TRAIN_DATA_PATH)
    test_id_evaluator = EVAL_DICT[ARGS.problem](dataset_dir=TEST_ID_DATA_PATH)
    test_ood_evaluator = EVAL_DICT[ARGS.problem](dataset_dir=TEST_OOD_DATA_PATH)
    start_time = time.time()
    train_fitness_list, train_fitness_string, train_success = train_evaluator.evaluate_task(
        code_string, return_parameters=True
    )
    fit_time = time.time() - start_time
    test_id_res = test_id_evaluator._evaluate_with_fixed_params(code_string, train_fitness_list[-1])
    test_ood_res = test_ood_evaluator._evaluate_with_fixed_params(code_string, train_fitness_list[-1])

    print(f"Paper:{ARGS.paper}, Problem: {ARGS.problem}, Model: {ARGS.model}")
    print(
        f"Fit time: {fit_time:.4f}\t"
        f"Contained fit: {contained_fit:{res_format}}\t"
        f"Train fit:{train_fitness_list[0]:{res_format}}\t"
        f"Test id fit:{test_id_res[0]:{res_format}}\t"
        f"Test ood fit:{test_ood_res[0]:{res_format}}"
    )