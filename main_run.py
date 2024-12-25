import json
import os
import time
import argparse

from coevo.llm.llm_api_https import HttpsApi


from coevo.examples.science.oscillation_1 import Oscillation1TaskInfo, Oscillation1Evaluator
from coevo.examples.science.oscillation_2 import Oscillation2TaskInfo, Oscillation2Evaluator
from coevo.examples.science.bactgrow import BactGrowTaskInfo, BactGrowEvaluator
from coevo.examples.science.stress_strain import StressStrainTaskInfo, StressStrainEvaluator

from coevo.methods.coevo import CoEvoAgent, CoEvoParas
from coevo.methods.coevo.reps import EnglishRep, PythonCodeRep, MathRep

TASK_INFO_DICT = {
    "oscillation_1": Oscillation1TaskInfo,
    "oscillation_2": Oscillation2TaskInfo,
    "bactgrow": BactGrowTaskInfo,
    "stress_strain": StressStrainTaskInfo
}
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
    parser = argparse.ArgumentParser(description="Science Discovery Experiment.")
    parser.add_argument(
        '--problem', type=str, choices=['oscillation_1', 'oscillation_2', 'bactgrow', 'stress_strain'],
        default='oscillation_1', help='Choose the problem to solve.'
    )
    ARGS = parser.parse_args()
    DATA_PATH = os.path.join(ABS_PATH, "data", ARGS.problem, "train.csv")

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(ABS_PATH, "res", ARGS.problem, time_stamp)
    model_config_file = os.path.join(ABS_PATH, "model_config.json")
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    llm_inst_list = [
        HttpsApi(
            host=model_config["host"],
            key=model_config["key"],
            model=model_config["model"],
            url=model_config["url"],
            timeout=model_config["timeout"]
    ) for _ in range(2)]

    run_config = {
        "max_gen": 97,
        "pop_size": 2,
        "num_init_gen": 6
    }
    with  open(os.path.join(output_path, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    paras = CoEvoParas(
        output_dir=output_path,
        max_gen=run_config["max_gen"], pop_size=run_config["pop_size"], num_init_gen=run_config["num_init_gen"],
        management="nds_evoda",
        selection="prob_rank",

        use_summary=True,
        llm_summarizer_inst=llm_inst_list[0],
        tokenizer_path=TOKENIZER_PATH,
        pool_size=100,

        rep_list=[EnglishRep(), PythonCodeRep(), MathRep()],
        rep_use=PythonCodeRep(),
        num_idea= [3, 3],

        use_profiler=True
    )

    task_info = TASK_INFO_DICT[ARGS.problem]()
    evaluator = EVAL_DICT[ARGS.problem](dataset_dir=DATA_PATH)
    agent = CoEvoAgent(
        llm_inst=llm_inst_list[1],
        task_info_inst=task_info,
        evaluator_inst=evaluator,
        paras=paras
    )

    agent.run()