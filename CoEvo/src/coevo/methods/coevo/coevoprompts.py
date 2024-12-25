from coevo.tasks.base import TaskInfo
from .coevoparas import CoEvoParas
from .prompts_str import *

class CoEvoPrompt:
    @classmethod
    def get_init_prompt(
        cls, task_info: TaskInfo, coevo_paras: CoEvoParas,
        use_summary: bool=False, summarizer=None
    ):
        if use_summary:
            assert summarizer is not None, "Summarizer is not provided."

        prompt_content = f"{task_info.task_info}\n"

        # Response Instructions
        prompt_content += \
            f'# How to Respond\n' \
            f'First brainstorm ideas, then write down your thoughts process for solving the task '\
            f'using the ideas, and finally provide the solution to the task in the required formats.'

        if use_summary:
            prompt_content += f' You will be provided with some effective ideas for inspiration, which may be helpful for you to solve the task.\n'
            idea_pool_search = summarizer.select_inspirations()
            prompt_content += \
                f'\nHere are some effective ideas which will help in solving the task:\n'
            prompt_content += f'{list_pool(idea_pool_search)}\n'

        prompt_content += \
            '\nHints:\n' \
            f'- Ideas: Brainstorm at least {coevo_paras.num_idea[0]} potential useful ideas for solving the task. '\
            'Each idea should be innovative, creative, and non-obvious. Include the name, definition (brief description), and reasoning for each idea.\n' \
            f'- Thoughts: Think deeply step-by-step for solving the task using the ideas.\n' \
            f'- Solutions: Provide solution to the task in {len(coevo_paras.rep_list)} formats, '\
            f'[{coevo_paras.rep_use.rep_name}] will be used for evaluation without any edit. Here are the formats:\n'
        for rep_i, rep in enumerate(coevo_paras.rep_list):
            prompt_content += f'    [{rep.rep_name}]: {rep.rep_def}\n'

        prompt_content += f'\n{init_sol_response_format(coevo_paras)}'
        return prompt_content

    @classmethod
    def get_continue_prompt(
            cls, task_info: TaskInfo, coevo_paras: CoEvoParas,
            inspiration_layer_i, previous_result_dict_list,
            use_summary: bool=False, summarizer=None
    ):
        if use_summary:
            assert summarizer is not None, "Summarizer is not provided."
        prompt_content = f"{task_info.task_info}\n"

        # Adding Response Instruction.
        prompt_content += \
            f'# How to Respond\n' \
            f'First, reason about implications about the task and previous solutions to derive ideas. '\
            f'Then, write down your thoughts process for solving the task using the ideas. '\
            f'Finally, provide the solution to the task in the required formats.'
        if use_summary:
            prompt_content += f' You will be provided with some effective ideas for inspiration, which may be helpful for you to solve the task.\n'
            ideas_in_current_round = previous_result_dict_list[-1]["Ideas"]
            idea_pool_search = summarizer.select_inspirations(ideas_in_current_round)
            prompt_content += \
                f'\nHere are some effective ideas which will help in solving the task:\n'
            prompt_content += f'{list_pool(idea_pool_search)}\n'
        prompt_content += f'\nHere are the previous solutions to the task:\n{list_single_sequential(coevo_paras, previous_result_dict_list)}\n'
        prompt_content += \
            f'Hints:\n' \
            f'- Ideas: Reason about implications from the task and previous solutions to derive at least {coevo_paras.num_idea[inspiration_layer_i]} useful ideas. '\
            f'Each idea should be innovative, creative, non-obvious, and derived from the previous solutions with clear reasoning and citations. ' \
            f'Include the Citations (Direct from the task or the previous solutions.), Implications (Your step-by-step reasoned-through implications), '\
            f'name, definition (brief description), and reasoning for each idea.\n'\
            f'- Thoughts: Think deeply step-by-step for solving the task using the ideas. Avoid errors in previous solutions.\n'\
            f'- Solutions: Provide solution to the task in {len(coevo_paras.rep_list)} formats, '\
            f'[{coevo_paras.rep_use.rep_name}] will be used for evaluation without any edit. Here are the formats:\n'
        for rep_i, rep in enumerate(coevo_paras.rep_list):
            prompt_content += f'    [{rep.rep_name}]: {rep.rep_def}\n'

        prompt_content += f'\n{continue_sol_response_format(coevo_paras)}'
        return prompt_content

    @classmethod
    def get_offspring_prompt(
            cls, task_info: TaskInfo, coevo_paras: CoEvoParas,
            parents: list[list], mode: str = "crossover_positive",
            use_summary: bool = False, summarizer=None
    ):
        if use_summary:
            assert summarizer is not None, "Summarizer is not provided."

        prompt_content = f"{task_info.task_info}\n# How to Respond\n"
        prompt_content += f"{get_offspring_how_to(mode)}"
        prompt_content += \
            f'First brainstorm ideas, then write down your thoughts process for solving the task ' \
            f'using the ideas, and finally provide the solution to the task in the required formats.'

        if use_summary:
            prompt_content += f' You will be provided with some effective ideas for inspiration, which may be helpful for you to solve the task.\n'
            idea_pool_search = summarizer.select_inspirations()
            prompt_content += \
                f'\nHere are some effective ideas which will help in solving the overall task:\n'
            prompt_content += f'{list_pool(idea_pool_search)}\n'

        prompt_content += \
            f'\nHere are {len(parents)} existing solutions with their ideas and evaluation results:\n' \
            f"{list_parents(coevo_paras, parents)}\n"

        prompt_content += \
            '\nHints:\n' \
            f'- Ideas: Brainstorm at least {coevo_paras.num_idea[0]} potential useful ideas for solving the task. ' \
            'Each idea should be innovative, creative, and non-obvious. Include the name, definition (brief description), and reasoning for each idea.\n' \
            f'- Thoughts: Think deeply step-by-step for solving the task using the ideas.\n' \
            f'- Solutions: Provide solution to the task in {len(coevo_paras.rep_list)} formats, ' \
            f'[{coevo_paras.rep_use.rep_name}] will be used for evaluation without any edit. Here are the formats:\n'
        for rep_i, rep in enumerate(coevo_paras.rep_list):
            prompt_content += f'    [{rep.rep_name}]: {rep.rep_def}\n'

        prompt_content += f'{init_sol_response_format(coevo_paras)}'
        return prompt_content

    @classmethod
    def get_single_summarizer_prompt(cls, task_info: TaskInfo, coevo_paras: CoEvoParas, previous_result_dict_list, inspiration_pool):


        prompt_content = f'{get_summarizer_how_to()}\n'

        prompt_content += f"Here is the task information:\n{task_info.task_info}\n"
        prompt_content += f'Here are the previous solutions:\n{list_single_sequential(coevo_paras, previous_result_dict_list)}\n'
        prompt_content += \
            f'This are the ideas you’ve summarized:\n'\
            f'{list_pool(inspiration_pool)}\n'

        # Adding Summarization Instruction.
        prompt_content += f'{get_summarizer_end()}'
        return prompt_content

    @classmethod
    def get_offspring_summarizer_prompt(
            cls, task_info: TaskInfo, coevo_paras: CoEvoParas,
            parents_list, offspring_list,
            inspiration_pool
    ):
        prompt_content = f'{get_summarizer_how_to()}'

        prompt_content += f"Here is the task information:\n{task_info.task_info}\n"
        prompt_content += f'Here are the previous solutions:\n{list_parents(coevo_paras, parents_list)}\n'
        prompt_content += f"{list_offsprings(coevo_paras, offspring_list, len(parents_list))}"
        prompt_content += \
            f'This are the ideas you’ve summarized:\n' \
            f'{list_pool(inspiration_pool)}\n'

        # Adding Summarization Instruction.
        prompt_content += f'{get_summarizer_end()}'
        return prompt_content

