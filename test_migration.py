import os
from coevo.examples.science.oscillation_1.oscillation_1_task_info import Oscillation1TaskInfo
from coevo.examples.science.oscillation_1.oscillation_1_evaluator import Oscillation1Evaluator
from coevo.tasks.coevo_task import CoEvoTask
from coevo.core.coevo_interface import CoEvoInterface
from coevo.core.coevo_method import CoEvoMethod
from evotoolkit.tools import HttpsApi

task_info = Oscillation1TaskInfo()
evaluator = Oscillation1Evaluator(dataset_dir="./CoEvo/data/oscillation_1.csv")
task = CoEvoTask(task_info, evaluator)
interface = CoEvoInterface(task, num_idea=[6, 3, 3])

llm = HttpsApi(
    api_url=os.getenv("API_URL", "https://api.openai.com/v1"),
    key=os.getenv("API_KEY"),
    model=os.getenv("MODEL", "gpt-4o")
)

method = CoEvoMethod(
    interface=interface,
    output_path="./test_results",
    running_llm=llm,
    max_generations=3,
    pop_size=2,
    num_idea=[6, 3, 3]
)

print("Starting CoEvo migration test...")
best = method.run()
print(f"Best solution score: {best.evaluation_res.score}")
print("Test completed successfully!")
