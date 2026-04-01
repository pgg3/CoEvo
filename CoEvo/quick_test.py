import os

from coevo.core.coevo_interface import CoEvoInterface
from coevo.core.coevo_method import CoEvoMethod
from coevo.tasks.oscillation_1 import Oscillation1Task
from evotoolkit.tools import HttpsApi

print("Initializing task...")
task = Oscillation1Task(dataset_dir="./data/oscillation_1/train.csv")
interface = CoEvoInterface(task, num_idea=[2, 1, 1])

print("Connecting to LLM...")
llm = HttpsApi(
    api_url=os.getenv("API_URL", "https://api.openai.com/v1"),
    key=os.getenv("API_KEY", ""),
    model=os.getenv("MODEL", "gpt-4o"),
)

print("Creating method...")
method = CoEvoMethod(
    interface=interface,
    running_llm=llm,
    output_path="./test_results",
    max_generations=1,
    pop_size=1,
)

print("Running evolution...")
best = method.run()
if best is not None and best.evaluation_res is not None:
    print(f"\nBest solution score: {best.evaluation_res.score}")
else:
    print("\nNo valid solution found.")
print("Test completed!")
