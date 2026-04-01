"""High-level prompt builder for the CoEvo algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .prompt_templates import (
    continue_sol_response_format,
    get_offspring_how_to,
    get_summarizer_end,
    get_summarizer_how_to,
    init_sol_response_format,
    list_offsprings,
    list_parents,
    list_pool,
    list_single_sequential,
)

if TYPE_CHECKING:
    from evotoolkit.core import Solution, TaskSpec


class CoEvoPromptBuilder:
    """Builds all prompts required by the CoEvo algorithm.

    Parameters
    ----------
    task_spec:
        TaskSpec from the task.  ``extras["raw_task_description"]`` is used as
        the task description.
    num_idea:
        Number of ideas to request per layer, e.g. [6, 3, 3].
    rep_list:
        List of representation dicts with keys ``"name"`` and ``"definition"``.
    rep_use_name:
        The representation name whose code will be evaluated.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        num_idea: list[int],
        rep_list: list[dict],
        rep_use_name: str,
    ) -> None:
        raw_desc = task_spec.extras.get("raw_task_description", task_spec.prompt)
        program_template = task_spec.extras.get("program_template", "")
        if program_template:
            self.task_description = f"{raw_desc}\nProgram Template:\n{program_template}"
        else:
            self.task_description = raw_desc
        self.num_idea = num_idea
        self.rep_list = rep_list
        self.rep_use_name = rep_use_name

    # ------------------------------------------------------------------
    # Initialisation prompt
    # ------------------------------------------------------------------

    def get_init_prompt(self, idea_pool: list[dict] | None = None) -> str:
        content = f"{self.task_description}\n"
        content += (
            "# How to Respond\n"
            "First brainstorm ideas, then write down your thoughts process for solving the task "
            "using the ideas, and finally provide the solution to the task in the required formats."
        )

        if idea_pool:
            content += (
                " You will be provided with some effective ideas for inspiration, "
                "which may be helpful for you to solve the task.\n"
                "\nHere are some effective ideas which will help in solving the task:\n"
            )
            content += list_pool(idea_pool)

        content += (
            "\nHints:\n"
            f"- Ideas: Brainstorm at least {self.num_idea[0]} potential useful ideas for solving the task. "
            "Each idea should be innovative, creative, and non-obvious. Include the name, definition (brief description), and reasoning for each idea.\n"
            "- Thoughts: Think deeply step-by-step for solving the task using the ideas.\n"
            f"- Solutions: Provide solution to the task in {len(self.rep_list)} formats, "
            f"[{self.rep_use_name}] will be used for evaluation without any edit. Here are the formats:\n"
        )
        for rep in self.rep_list:
            content += f'    [{rep["name"]}]: {rep["definition"]}\n'

        content += f"\n{init_sol_response_format(self.rep_list)}"
        return content

    # ------------------------------------------------------------------
    # Continue prompt (multi-layer reasoning)
    # ------------------------------------------------------------------

    def get_continue_prompt(
        self,
        layer: int,
        history_chain: list[Solution],
        idea_pool: list[dict] | None = None,
    ) -> str:
        content = f"{self.task_description}\n"
        content += (
            "# How to Respond\n"
            "First, reason about implications about the task and previous solutions to derive ideas. "
            "Then, write down your thoughts process for solving the task using the ideas. "
            "Finally, provide the solution to the task in the required formats."
        )

        if idea_pool:
            # Use ideas from the last solution as search context
            last_ideas = history_chain[-1].metadata.extras.get("Ideas", []) if history_chain else []
            content += (
                " You will be provided with some effective ideas for inspiration, "
                "which may be helpful for you to solve the task.\n"
                "\nHere are some effective ideas which will help in solving the task:\n"
            )
            content += list_pool(idea_pool)

        content += (
            f"\nHere are the previous solutions to the task:\n"
            f"{list_single_sequential(history_chain, self.rep_use_name)}\n"
        )

        content += (
            "Hints:\n"
            f"- Ideas: Reason about implications from the task and previous solutions to derive at least {self.num_idea[layer]} useful ideas. "
            "Each idea should be innovative, creative, non-obvious, and derived from the previous solutions with clear reasoning and citations. "
            "Include the Citations (Direct from the task or the previous solutions.), Implications (Your step-by-step reasoned-through implications), "
            "name, definition (brief description), and reasoning for each idea.\n"
            "- Thoughts: Think deeply step-by-step for solving the task using the ideas. Avoid errors in previous solutions.\n"
            f"- Solutions: Provide solution to the task in {len(self.rep_list)} formats, "
            f"[{self.rep_use_name}] will be used for evaluation without any edit. Here are the formats:\n"
        )
        for rep in self.rep_list:
            content += f'    [{rep["name"]}]: {rep["definition"]}\n'

        content += f"\n{continue_sol_response_format(self.rep_list)}"
        return content

    # ------------------------------------------------------------------
    # Offspring prompt (crossover / mutation)
    # ------------------------------------------------------------------

    def get_offspring_prompt(
        self,
        parents: list[list[Solution]],
        mode: str,
        idea_pool: list[dict] | None = None,
    ) -> str:
        content = f"{self.task_description}\n# How to Respond\n"
        content += get_offspring_how_to(mode)
        content += (
            "First brainstorm ideas, then write down your thoughts process for solving the task "
            "using the ideas, and finally provide the solution to the task in the required formats."
        )

        if idea_pool:
            content += (
                " You will be provided with some effective ideas for inspiration, "
                "which may be helpful for you to solve the task.\n"
                "\nHere are some effective ideas which will help in solving the overall task:\n"
            )
            content += list_pool(idea_pool)

        content += (
            f"\nHere are {len(parents)} existing solutions with their ideas and evaluation results:\n"
            f"{list_parents(parents, self.rep_use_name)}\n"
        )

        content += (
            "\nHints:\n"
            f"- Ideas: Brainstorm at least {self.num_idea[0]} potential useful ideas for solving the task. "
            "Each idea should be innovative, creative, and non-obvious. Include the name, definition (brief description), and reasoning for each idea.\n"
            "- Thoughts: Think deeply step-by-step for solving the task using the ideas.\n"
            f"- Solutions: Provide solution to the task in {len(self.rep_list)} formats, "
            f"[{self.rep_use_name}] will be used for evaluation without any edit. Here are the formats:\n"
        )
        for rep in self.rep_list:
            content += f'    [{rep["name"]}]: {rep["definition"]}\n'

        content += init_sol_response_format(self.rep_list)
        return content

    # ------------------------------------------------------------------
    # Summarizer prompts
    # ------------------------------------------------------------------

    def get_summarizer_prompt_single(
        self,
        chain: list[Solution],
        idea_pool: list[dict],
    ) -> str:
        content = f"{get_summarizer_how_to()}\n"
        content += f"Here is the task information:\n{self.task_description}\n"
        content += f"Here are the previous solutions:\n{list_single_sequential(chain, self.rep_use_name)}\n"
        content += (
            "This are the ideas you\u2019ve summarized:\n"
            f"{list_pool(idea_pool)}\n"
        )
        content += get_summarizer_end()
        return content

    def get_summarizer_prompt_offspring(
        self,
        parents: list[list[Solution]],
        offspring: list[list[Solution]],
        idea_pool: list[dict],
    ) -> str:
        content = get_summarizer_how_to()
        content += f"Here is the task information:\n{self.task_description}\n"
        content += f"Here are the previous solutions:\n{list_parents(parents, self.rep_use_name)}\n"
        content += list_offsprings(offspring, self.rep_use_name, len(parents))
        content += (
            "This are the ideas you\u2019ve summarized:\n"
            f"{list_pool(idea_pool)}\n"
        )
        content += get_summarizer_end()
        return content
