"""
Prompt string template functions for CoEvo.

These replace the scattered prompts_str/ files.  All functions are pure string
builders with no dependency on legacy CoEvoParas or Agent classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evotoolkit.core import Solution


# ---------------------------------------------------------------------------
# Idea pool listing
# ---------------------------------------------------------------------------

def list_pool(idea_pool: list[dict]) -> str:
    """Format the idea pool for prompt injection."""
    if not idea_pool:
        return "{}\n"
    content = ""
    for idx, idea in enumerate(idea_pool):
        content += (
            f'{idx + 1}. {idea["Name"]}:\n'
            f'  Definition: {idea["Definition"]}\n'
            f'  Example: {idea["Example"]}\n\n'
        )
    return content


# ---------------------------------------------------------------------------
# Sequential solution listing (for continue_reason prompts)
# ---------------------------------------------------------------------------

def list_single_sequential(
    history_chain: list[Solution],
    rep_use_name: str,
) -> str:
    """Format a chain of solutions sequentially for the continue prompt."""
    content = ""
    for prev_res_id, sol in enumerate(history_chain):
        if prev_res_id == 0:
            content += f"**Solution No.{prev_res_id + 1}**\n"
        else:
            content += f"**Solution No.{prev_res_id + 1}** (Continued from Solution No.{prev_res_id})\n"

        ideas = sol.metadata.extras.get("Ideas", [])
        content += "1. Its Ideas\n"
        for idea in ideas:
            content += f'- {idea.get("Name", "")}: {idea.get("Definition", "")}\n'
            if prev_res_id > 0:
                content += f'  Quotes: {idea.get("Quote", "")}\n'
                content += f'  Implications: {idea.get("Implication", "")}\n\n'

        solutions_dict = sol.metadata.extras.get("Solutions", {})
        code = solutions_dict.get(rep_use_name, "")
        content += f"\n2. Its Format {rep_use_name}:\n{code}\n\n"

        content += "3. Its Evaluation Results\n"
        if sol.evaluation_res is None or not sol.evaluation_res.valid:
            error = (
                sol.metadata.extras.get("parse_error")
                or (sol.evaluation_res.additional_info.get("error") if sol.evaluation_res else None)
                or "Invalid solution"
            )
            content += f"{error}\n\n"
        else:
            fitness_string = sol.evaluation_res.additional_info.get("fitness_string", "")
            content += f"{fitness_string}\n\n"

    return content


# ---------------------------------------------------------------------------
# Parent listing (for offspring prompts)
# ---------------------------------------------------------------------------

def list_parents(
    parents: list[list[Solution]],
    rep_use_name: str,
) -> str:
    """Format parent chains for offspring prompt."""
    content = ""
    for indiv_idx, chain in enumerate(parents):
        best_sol = chain[-1]
        ideas = best_sol.metadata.extras.get("Ideas", [])

        content += f"**Solution No.{indiv_idx + 1}**\n1. Its Ideas\n"
        for idea in ideas:
            content += f'- {idea.get("Name", "")}: {idea.get("Definition", "")}\n'

        solutions_dict = best_sol.metadata.extras.get("Solutions", {})
        code = solutions_dict.get(rep_use_name, "")
        content += f"\n2. Its Format {rep_use_name}:\n{code}\n\n"

        content += "3. Its Evaluation Results\n"
        if best_sol.evaluation_res is None or not best_sol.evaluation_res.valid:
            error = (
                best_sol.metadata.extras.get("parse_error")
                or (best_sol.evaluation_res.additional_info.get("error") if best_sol.evaluation_res else None)
                or "Invalid solution"
            )
            content += f"{error}\n\n"
        else:
            fitness_string = best_sol.evaluation_res.additional_info.get("fitness_string", "")
            content += f"{fitness_string}\n\n"

    return content


def list_offsprings(
    offspring_list: list[list[Solution]],
    rep_use_name: str,
    parents_num: int,
) -> str:
    """Format offspring chains for summarizer prompt."""
    content = ""
    for indiv_idx, chain in enumerate(offspring_list):
        best_sol = chain[-1]
        ideas = best_sol.metadata.extras.get("Ideas", [])

        content += (
            f"**Solution No.{indiv_idx + 1 + parents_num}** "
            f"(Offspring from Solution No.1 - No.{parents_num})\n1. Its Ideas\n"
        )
        for idea in ideas:
            content += f'- {idea.get("Name", "")}: {idea.get("Definition", "")}\n'

        solutions_dict = best_sol.metadata.extras.get("Solutions", {})
        code = solutions_dict.get(rep_use_name, "")
        content += f"\n2. Its Format {rep_use_name}:\n{code}\n\n"

        content += "3. Its Evaluation Results\n"
        if best_sol.evaluation_res is None or not best_sol.evaluation_res.valid:
            error = (
                best_sol.metadata.extras.get("parse_error")
                or (best_sol.evaluation_res.additional_info.get("error") if best_sol.evaluation_res else None)
                or "Invalid solution"
            )
            content += f"{error}\n\n"
        else:
            fitness_string = best_sol.evaluation_res.additional_info.get("fitness_string", "")
            content += f"{fitness_string}\n\n"

    return content


# ---------------------------------------------------------------------------
# Offspring mode preamble
# ---------------------------------------------------------------------------

def get_offspring_how_to(mode: str) -> str:
    content = "You will be presented with some existing solutions to the task. "
    content += {
        "crossover_positive": "Now create a new solution that has a totally different form from the given solutions but can be motivated from the existing ones. ",
        "crossover_negative": "Now create a new solution that has a totally different form from the given solutions. ",
        "mutation_positive": "Now create a solution in different forms but can be a modified version of the existing solution. ",
        "mutation_negative": "Now create a totally different solution from the existing solution. ",
    }[mode]
    return content


# ---------------------------------------------------------------------------
# Response format templates
# ---------------------------------------------------------------------------

def init_sol_response_format(rep_list: list[dict]) -> str:
    content = (
        "Response Format (Replace ...): \n\n"
        "## Ideas\n"
        "- Idea 1:\n"
        "  - Name:...\n"
        "  - Reasoning:...\n"
        "  - Definition:...\n...\n\n"
        "## Thoughts\n"
        "...\n\n"
        "## Solutions\n"
    )
    for rep in rep_list:
        content += f'### {rep["name"]}:\n...\n'
    return content


def continue_sol_response_format(rep_list: list[dict]) -> str:
    content = (
        "Response Format: (Replace ...)\n\n"
        "## Ideas\n"
        "  - Idea 1:\n"
        "    - Quotes:...\n"
        "    - Implications:...\n"
        "    - Name:...\n"
        "    - Reasoning:...\n"
        "    - Definition:...\n"
        "...\n\n"
        "## Thoughts\n"
        "...\n\n"
        "## Solutions\n"
    )
    for rep in rep_list:
        content += f'### {rep["name"]}:\n...\n'
    return content


# ---------------------------------------------------------------------------
# Summarizer templates
# ---------------------------------------------------------------------------

def get_summarizer_how_to() -> str:
    return (
        "As a smart analyzer, you will be presented with the task information and some solutions to the task, "
        "each with a list of ideas and their evaluation results.\n"
        "Your task is to to analyze why the last solution has the best performance by "
        "comparing the differences in the solutions' ideas, implementations, and evaluation results.\n"
        "You will also receive an idea pool, which contains all the ideas you have already summarized.\n"
    )


def get_summarizer_end() -> str:
    return (
        "Hints: You need to analyze why the last solution has the best performance and summarize the ideas that "
        "are effective for evaluation performance improvement in the format "
        "of: Reasoning (reasoning and analysis, why it is useful for solving the task or improving the performance), "
        "Name, Definition (brief description of the idea), and Example. Example part is extremely important for later reuse of the summarized idea, "
        "so make sure to provide a clear and concise example.\n\n"
        "Response Format: : (Replace ...)\n\n"
        "## New Ideas\n"
        "- Idea 1:\n"
        "  - Reasoning:...\n"
        "  - Name:...\n"
        "  - Definition:...\n"
        "  - Example:...\n\n"
        "...\n\n"
        "## Analysis\n\n"
        "...\n\n"
    )
