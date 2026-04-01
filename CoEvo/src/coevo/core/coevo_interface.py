"""CoEvo MethodInterface: prompt construction and response parsing."""

from __future__ import annotations

import re

from evotoolkit.core import MethodInterface, Solution, SolutionMetadata
from evotoolkit.task.python_task import PythonTask

from .prompts.coevo_prompts import CoEvoPromptBuilder


class CoEvoInterface(MethodInterface):
    """Adapter between CoEvoMethod and a PythonTask.

    Parameters
    ----------
    task:
        A PythonTask whose spec contains ``extras["raw_task_description"]``.
    num_idea:
        Number of ideas per layer, e.g. [6, 3, 3].
    rep_list:
        Representation descriptors, e.g.
        ``[{"name": "Python Code", "definition": "Runnable python code implementation."}]``.
    rep_use_name:
        Which representation to evaluate (must match a key in ``rep_list``).
    """

    def __init__(
        self,
        task: PythonTask,
        *,
        num_idea: list[int] | None = None,
        rep_list: list[dict] | None = None,
        rep_use_name: str = "Python Code",
    ) -> None:
        super().__init__(task)
        self.num_idea: list[int] = num_idea if num_idea is not None else [6, 3, 3]
        self.rep_list: list[dict] = rep_list if rep_list is not None else [
            {"name": "Python Code", "definition": "Runnable python code implementation."}
        ]
        self.rep_use_name: str = rep_use_name
        self.prompt_builder = CoEvoPromptBuilder(
            task_spec=task.spec,
            num_idea=self.num_idea,
            rep_list=self.rep_list,
            rep_use_name=self.rep_use_name,
        )

    # ------------------------------------------------------------------
    # Prompt helpers (delegated to prompt_builder)
    # ------------------------------------------------------------------

    def get_init_prompt(self, idea_pool: list[dict] | None = None) -> str:
        return self.prompt_builder.get_init_prompt(idea_pool)

    def get_continue_prompt(
        self,
        layer: int,
        history_chain: list[Solution],
        idea_pool: list[dict] | None = None,
    ) -> str:
        return self.prompt_builder.get_continue_prompt(layer, history_chain, idea_pool)

    def get_offspring_prompt(
        self,
        parents: list[list[Solution]],
        mode: str,
        idea_pool: list[dict] | None = None,
    ) -> str:
        return self.prompt_builder.get_offspring_prompt(parents, mode, idea_pool)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, response_str: str, *, layer: int = 0) -> Solution:
        """Parse an LLM response and return a Solution.

        Extracts Ideas, Thoughts, and Solutions sections.  The ``sol_string``
        of the returned Solution is the code from ``Solutions[rep_use_name]``
        (empty string on parse failure, which triggers an invalid evaluation).
        """
        parsed_dict, success = self._parse_response_impl(response_str, layer)

        if not success or self.rep_use_name not in parsed_dict.get("Solutions", {}):
            return MethodInterface.make_solution(
                "",
                extras={
                    "parse_error": "Parser Error, Response NOT VALID for evaluation.",
                    "raw_content": response_str,
                    "Ideas": [],
                    "Thoughts": "",
                    "Solutions": {},
                },
            )

        code = parsed_dict["Solutions"][self.rep_use_name]
        return MethodInterface.make_solution(
            code,
            description=parsed_dict.get("Thoughts", ""),
            extras={
                "Ideas": parsed_dict.get("Ideas", []),
                "Thoughts": parsed_dict.get("Thoughts", ""),
                "Solutions": parsed_dict["Solutions"],
                "raw_content": response_str,
            },
        )

    # ------------------------------------------------------------------
    # Internal parsing logic (ported from CoEvoAgent._parse_response)
    # ------------------------------------------------------------------

    def _parse_response_impl(self, response_str: str, layer: int = 0) -> tuple[dict, bool]:
        try:
            parsed_dict: dict = {"Ideas": [], "Thoughts": "", "Solutions": {}}

            inspiration_heading = r"(?:idea|Idea|IDEA)[sS]?\s*:?"
            thought_heading = r"(?:thought|Thought|THOUGHT)[sS]?\s*:?"
            solution_heading = r"(?:solution|Solution|SOLUTION)[sS]?\s*:?"

            # -- Ideas --
            ideas_pattern = re.compile(
                r"##\s*" + inspiration_heading + r"\s*(.*?)##\s*" + thought_heading,
                re.DOTALL,
            )
            match = ideas_pattern.search(response_str)
            parsed_ideas = []
            if match:
                ideas_text = match.group(1).strip()
                if layer == 0:
                    inspiration_pattern = re.compile(
                        r"(?:name|Name|NAME)\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:reasoning|Reasoning|REASONING|reason|Reason|REASON)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:definition|Definition|DEFINITION)[sS]?\s*:?\s*(.*?)(?=\n|$)",
                        re.DOTALL,
                    )
                    for m in inspiration_pattern.findall(ideas_text):
                        parsed_ideas.append({
                            "Name": m[0].strip(),
                            "Reasoning": m[1].strip(),
                            "Definition": m[2].strip(),
                        })
                else:
                    inspiration_pattern = re.compile(
                        r"(?:quote|Quote|QUOTE)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:implication|Implication|IMPLICATION)\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:name|Name|NAME)\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:reasoning|Reasoning|REASONING|reason|Reason|REASON)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:definition|Definition|DEFINITION)[sS]?\s*:?\s*(.*?)(?=\n|$)",
                        re.DOTALL,
                    )
                    for m in inspiration_pattern.findall(ideas_text):
                        parsed_ideas.append({
                            "Quote": m[0].strip(),
                            "Implication": m[1].strip(),
                            "Name": m[2].strip(),
                            "Reasoning": m[3].strip(),
                            "Definition": m[4].strip(),
                        })
            parsed_dict["Ideas"] = parsed_ideas

            # -- Thoughts --
            thoughts_pattern = re.compile(
                r"##\s*" + thought_heading + r"\s*(.*?)##\s*" + solution_heading,
                re.DOTALL,
            )
            thoughts = thoughts_pattern.findall(response_str)
            parsed_dict["Thoughts"] = thoughts[0] if thoughts else ""

            # -- Solutions --
            solutions_pattern = re.compile(
                r"##\s*" + solution_heading + r"\s*(.*?)$",
                re.DOTALL,
            )
            solutions_match = solutions_pattern.search(response_str)
            if not solutions_match:
                return parsed_dict, False

            solutions_string = solutions_match.group(1).strip()
            for rep in self.rep_list:
                rep_name = rep["name"]
                rep_match = re.search(
                    r"###\s*" + re.escape(rep_name) + r"\s*:?\s*(.*?)(?=###\s*|$)",
                    solutions_string,
                    re.DOTALL,
                )
                if rep_match:
                    rep_text = rep_match.group(1).strip()
                    if rep_text.startswith("```"):
                        code_match = re.search(r"```[a-zA-Z]*\n(.*?)(?=```|$)", rep_text, re.DOTALL)
                        if code_match:
                            parsed_dict["Solutions"][rep_name] = code_match.group(1).strip()
                    else:
                        parsed_dict["Solutions"][rep_name] = rep_text
                else:
                    return parsed_dict, False

        except Exception:
            return {}, False

        return parsed_dict, True
