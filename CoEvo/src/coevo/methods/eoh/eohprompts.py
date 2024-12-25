import copy
from typing import List, Dict
from coevo.tasks.program_design import ProgramTaskInfo

class EoHPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def get_prompt_i1(cls, task_info: ProgramTaskInfo):
        prompt_content = \
            f"{task_info.task_info}\n" \
            f"1. First, describe your new algorithm and main steps in one sentence. "\
            f"The description must be inside within boxed {{}}. "\
            f"Next, implement it in Python as a function named {task_info.program_name}. " \
            f"This function should accept input(s): {task_info.program_input}. " \
            f"The function should return output(s): {task_info.program_output}. " \
            f"{task_info.other_info}. Do not give additional explanations.\n" \
            f"Response Example:\n" \
            f"{{The function is to...}}\n\n" \
            f"```python\nimport...\n```\n"
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_info: ProgramTaskInfo, indivs: List[dict]):
        indivs_prompt = f""
        for i, indi in enumerate(indivs):
            indivs_prompt = \
                f"{indivs_prompt}"\
                f"No. {i + 1} algorithm and the corresponding code are:\n"\
                f"{indi['description']}\n"\
                f"{indi['code']}"
        # create prmpt content
        prompt_content = \
            f"{task_info.task_info}"\
            f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
            f"{indivs_prompt}"\
            f"Please help me create a new algorithm that has a totally different form from the given ones.\n"\
            f"1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.\n "\
            f"2. Next, implement the following Python function:\n"\
            f"This function should accept input(s): {task_info.program_input}."\
            f"The function should return output(s): {task_info.program_output}."\
            f"{task_info.other_info}. Do not give additional explanations.\n"\
            f"Response Example:\n"\
            f"{{The function is to...}}\n\n"\
            f"```python\nimport...\n```\n"
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_info: ProgramTaskInfo, indivs: List[dict]):
        indivs_prompt = f""
        for i, indi in enumerate(indivs):
            indivs_prompt = \
                f"{indivs_prompt}" \
                f"No. {i + 1} algorithm and the corresponding code are:\n" \
                f"{indi['description']}\n" \
                f"{indi['code']}"
        # create prmpt content
        prompt_content = \
            f"{task_info.task_info}" \
            f"I have {len(indivs)} existing algorithms with their codes as follows:\n" \
            f"{indivs_prompt}" \
            f"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n" \
            f"1. Firstly, identify the common backbone idea in the provided algorithms.\n" \
            f"2. Secondly, based on the backbone idea describe your new algorithm in one sentence. "\
            f"The description must be inside within boxed {{}}.\n" \
            f"3. Thirdly, implement the following Python function:\n" \
            f"This function should accept input(s): {task_info.program_input}." \
            f"The function should return output(s): {task_info.program_output}." \
            f"{task_info.other_info}. Do not give additional explanations.\n" \
            f"Response Example:\n" \
            f"{{The function is to...}}\n\n" \
            f"```python\nimport...\n```\n"
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_info: ProgramTaskInfo, indi: List[dict]):
        # create prmpt content
        prompt_content = \
            f"{task_info.task_info}" \
            f"I have one algorithm with its code as follows. Algorithm description:\n" \
            f"{indi[0]['description']}\n" \
            f"Code:\n" \
            f"{indi[0]['code']}\n" \
            f"Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n" \
            f"1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.\n" \
            f"2. Next, implement the following Python function:\n" \
            f"This function should accept input(s): {task_info.program_input}." \
            f"The function should return output(s): {task_info.program_output}." \
            f"{task_info.other_info}. Do not give additional explanations.\n" \
            f"Response Example:\n" \
            f"{{The function is to...}}\n\n" \
            f"```python\nimport...\n```\n"
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_info: ProgramTaskInfo, indi: List[dict]):
        # create prmpt content
        prompt_content = \
            f"{task_info.task_info}" \
            f"I have one algorithm with its code as follows. Algorithm description:\n" \
            f"{indi[0]['description']}\n" \
            f"Code:\n" \
            f"{indi[0]['code']}\n" \
            f"Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.\n" \
            f"1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.\n" \
            f"2. Next, implement the following Python function:\n" \
            f"This function should accept input(s): {task_info.program_input}." \
            f"The function should return output(s): {task_info.program_output}." \
            f"{task_info.other_info}. Do not give additional explanations.\n" \
            f"Response Example:\n" \
            f"{{The function is to...}}\n\n" \
            f"```python\nimport...\n```\n"
        return prompt_content