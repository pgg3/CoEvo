from evotoolkit.core.method_interface import MethodInterface
from evotoolkit.core import Solution
import re


class CoEvoInterface(MethodInterface):
    def __init__(self, task, num_idea=None, **kwargs):
        super().__init__(task, **kwargs)
        self.num_idea = num_idea or [6, 3, 3]

    def get_init_prompt(self, **kwargs):
        task_info = self.task.task_info
        prompt = f"{task_info.task_info}\n\n{task_info.program_template}\n\n"
        prompt += f"Brainstorm {self.num_idea[0]} ideas and provide a Python solution."
        return prompt

    def get_layer_prompt(self, layer, previous_solutions, **kwargs):
        task_info = self.task.task_info
        prompt = f"{task_info.task_info}\n\nPrevious solutions:\n"
        for i, sol in enumerate(previous_solutions[:3]):
            prompt += f"{i+1}. Score: {sol.evaluation_res.score if sol.evaluation_res else 'N/A'}\n"
        prompt += f"\nDerive {self.num_idea[layer]} new ideas and provide an improved Python solution.\n"
        prompt += f"\n{task_info.program_template}"
        return prompt

    def parse_response(self, response_text, **kwargs):
        code_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response_text
        return Solution(sol_string=code, metadata={"raw_response": response_text})
