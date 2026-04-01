"""CoEvoMethod: full CoEvo algorithm as a PopulationMethod.

Ported from CoEvoAgent (methods/coevo/coevoagent.py) into the evotoolkit
PopulationMethod lifecycle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from evotoolkit.core import PopulationMethod, Solution

from .coevo_interface import CoEvoInterface
from .coevo_state import CoEvoState
from .nds import nds_select

if TYPE_CHECKING:
    from .summarizer import CoEvoSummarizer


class CoEvoMethod(PopulationMethod):
    """Continual Evolution algorithm integrated with evotoolkit.

    Parameters
    ----------
    interface:
        CoEvoInterface connecting the task and this method.
    running_llm:
        evotoolkit HttpsApi instance.
    output_path:
        Directory for checkpoints and history.
    verbose:
        Whether to print progress.
    max_generations:
        Total number of generations (init = gen 0, then gen 1..max_generations-1).
    pop_size:
        Target population size after management.
    num_init_per_gen:
        Number of brand-new chains to generate each generation (in addition to
        the 4 offspring modes).  Default 0 matches the original paper setting.
    num_crossover:
        Number of parents for crossover operators.
    use_nds:
        Use non-dominated sorting for population management.  If False, falls
        back to score-based _trim_population.
    use_summarizer:
        Enable the summarization mechanism.
    summarizer:
        A CoEvoSummarizer instance (required when use_summarizer=True).
    """

    algorithm_name = "coevo"
    startup_title = "COEVO ALGORITHM STARTED"
    state_cls = CoEvoState

    def __init__(
        self,
        interface: CoEvoInterface,
        *,
        running_llm,
        output_path: str = "./results",
        verbose: bool = True,
        max_generations: int = 97,
        pop_size: int = 2,
        num_init_per_gen: int = 0,
        num_crossover: int = 2,
        use_nds: bool = True,
        use_summarizer: bool = False,
        summarizer: CoEvoSummarizer | None = None,
    ) -> None:
        super().__init__(
            interface=interface,
            output_path=output_path,
            running_llm=running_llm,
            verbose=verbose,
        )
        self.max_generations = max_generations
        self.pop_size = pop_size
        self.num_init_per_gen = num_init_per_gen
        self.num_crossover = num_crossover
        self.use_nds = use_nds
        self.use_summarizer = use_summarizer
        self.summarizer: CoEvoSummarizer | None = summarizer

        self._generate_sol_modes = [
            "crossover_positive",
            "crossover_negative",
            "mutation_positive",
            "mutation_negative",
        ]

    # ------------------------------------------------------------------
    # evotoolkit lifecycle
    # ------------------------------------------------------------------

    def initialize_iteration(self) -> None:
        """Generate the initial population (generation 0)."""
        self.verbose_gen(f" Gen 0 ")
        # Create 2 * pop_size chains then trim — same as original _init_population
        for _ in range(2):
            for _ in range(self.pop_size):
                chain = self._init_a_sol()
                self._register_chain(chain)

        # Pad if we somehow have fewer than pop_size valid chains
        while len(self.state.population) < self.pop_size:
            chain = self._init_a_sol()
            self._register_chain(chain)

        self._manage_population()
        self._print_population()

    def step_iteration(self) -> None:
        """One generation of evolution."""
        gen = self.state.generation
        self.verbose_gen(f" Gen {gen} ")

        new_chains: list[list[Solution]] = []

        # New initializations
        for _ in range(self.num_init_per_gen):
            new_chains.append(self._init_a_sol())

        # 4 offspring modes
        for mode in self._generate_sol_modes:
            new_chains.append(self._generate_offspring(mode))

        for chain in new_chains:
            self._register_chain(chain)

        self._manage_population()
        self._print_population()
        self.state.generation += 1

    def should_stop_iteration(self) -> bool:
        return self.state.generation >= self.max_generations

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def _register_chain(self, chain: list[Solution]) -> None:
        """Register all solutions in a chain with evotoolkit state machinery."""
        chain_id = self.state.next_chain_id
        self.state.next_chain_id += 1
        self.state.layer_chains[chain_id] = chain

        for sol in chain:
            sol.metadata.extras["chain_id"] = chain_id
            self._register_population_solution(sol)

        # The chain representative in population is the last solution
        # (Already added via _register_population_solution above)

    def _manage_population(self) -> None:
        """Trim population to pop_size using NDS or score-based selection."""
        if self.use_nds:
            self.state.population = nds_select(self.state.population, self.pop_size)
        else:
            self._trim_population(self.pop_size)

    # ------------------------------------------------------------------
    # Solution generation
    # ------------------------------------------------------------------

    def _init_a_sol(self) -> list[Solution]:
        """Generate one chain starting from the init prompt."""
        start = time.time()
        self.verbose_info("INIT: ")

        idea_pool = self._get_idea_pool()
        prompt = self.interface.get_init_prompt(idea_pool)
        init_sol = self._prompt_parse_evaluate(prompt, layer=0)

        elapsed = time.time() - start
        self.verbose_info(f"Done:{elapsed:.1f}s\n")

        chain = self._continue_reason(init_sol)
        full_chain = [init_sol] + chain

        self._maybe_summarize_indiv(full_chain)
        return full_chain

    def _continue_reason(self, one_solution: Solution) -> list[Solution]:
        """Multi-layer reasoning: derive improved solutions from one_solution."""
        history_list = [one_solution]
        new_gen_list: list[Solution] = []

        for idea_layer_i in range(len(self.interface.num_idea)):
            if idea_layer_i == 0:
                continue

            start = time.time()
            self.verbose_info("\tCONTINUE: ")

            idea_pool = self._get_idea_pool_for_layer(history_list)
            prompt = self.interface.get_continue_prompt(idea_layer_i, history_list, idea_pool)
            finalized_sol = self._prompt_parse_evaluate(prompt, layer=idea_layer_i)

            elapsed = time.time() - start
            self.verbose_info(f"Done:{elapsed:.1f}s\n")

            # Early stop: invalid solution
            if finalized_sol.evaluation_res is None or not finalized_sol.evaluation_res.valid:
                return new_gen_list

            # Early stop: no improvement
            prev = history_list[-1]
            if prev.evaluation_res is not None and prev.evaluation_res.valid:
                prev_fitness = prev.evaluation_res.additional_info.get("fitness_list", [float("inf")])[0]
                curr_fitness = finalized_sol.evaluation_res.additional_info.get("fitness_list", [float("inf")])[0]
                if prev_fitness - curr_fitness < 1e-8:
                    return new_gen_list

            new_gen_list.append(finalized_sol)
            history_list.append(one_solution)  # intentional: always cite original solution

        return new_gen_list

    def _generate_offspring(self, mode: str) -> list[Solution]:
        """Generate one offspring chain via crossover or mutation."""
        op_mode = mode.split("_")[0]
        if op_mode == "crossover":
            parent_chains = self._select_parent_chains(self.num_crossover)
        elif op_mode == "mutation":
            parent_chains = self._select_parent_chains(1)
        else:
            raise ValueError(f"Unknown operation mode: {op_mode}")

        start = time.time()
        self.verbose_info(f"{mode.upper()} ")

        idea_pool = self._get_idea_pool()
        prompt = self.interface.get_offspring_prompt(parent_chains, mode, idea_pool)
        offspring_sol = self._prompt_parse_evaluate(prompt, layer=0)

        elapsed = time.time() - start
        self.verbose_info(f"Done:{elapsed:.1f}s\n")

        self._maybe_summarize_offspring(offspring_sol, parent_chains)

        chain = self._continue_reason(offspring_sol)
        full_chain = [offspring_sol] + chain

        self._maybe_summarize_indiv(full_chain)
        return full_chain

    # ------------------------------------------------------------------
    # LLM prompt→parse→evaluate pipeline
    # ------------------------------------------------------------------

    def _prompt_parse_evaluate(self, prompt: str, *, layer: int = 0) -> Solution:
        """Call LLM with retry, parse response, and evaluate."""
        n_retry = 0
        sol: Solution | None = None
        usage: dict = {}

        while n_retry <= 3:
            try:
                response_str, usage = self.running_llm.get_response(prompt)
                sol = self.interface.parse_response(response_str, layer=layer)

                # If parse succeeded (non-empty sol_string), exit retry loop
                if sol.sol_string:
                    break
            except Exception:
                pass
            n_retry += 1

        if sol is None or not sol.sol_string:
            # Return an invalid solution
            from evotoolkit.core import EvaluationResult, SolutionMetadata
            sol = Solution(
                sol_string="",
                metadata=SolutionMetadata(extras={"parse_error": "Failed after retries"}),
                evaluation_res=EvaluationResult(
                    valid=False,
                    score=float("-inf"),
                    additional_info={"error": "Failed to get valid response after retries"},
                ),
            )
        else:
            eval_res = self.task.evaluate(sol)
            sol.evaluation_res = eval_res

        self._record_generation_usage(usage)
        return sol

    # ------------------------------------------------------------------
    # Parent selection
    # ------------------------------------------------------------------

    def _select_parent_chains(self, num: int) -> list[list[Solution]]:
        """Select parent chains using ranked probabilistic sampling."""
        import random

        valid_pop = [
            sol for sol in self.state.population
            if sol.evaluation_res and sol.evaluation_res.valid
        ]
        if not valid_pop:
            # Fallback: any available
            candidates = self.state.population[:num] if self.state.population else []
        else:
            # prob_rank: probability proportional to 1/(rank + N) where rank starts at 0
            n = len(valid_pop)
            probs = [1 / (r + 1 + n) for r in range(n)]
            total = sum(probs)
            probs = [p / total for p in probs]

            selected = []
            for _ in range(num):
                r = random.random()
                cumulative = 0.0
                chosen = valid_pop[-1]
                for sol, p in zip(valid_pop, probs):
                    cumulative += p
                    if r <= cumulative:
                        chosen = sol
                        break
                selected.append(chosen)
            candidates = selected

        # Retrieve full chains from layer_chains
        chains = []
        for sol in candidates:
            chain_id = sol.metadata.extras.get("chain_id")
            if chain_id is not None and chain_id in self.state.layer_chains:
                chains.append(self.state.layer_chains[chain_id])
            else:
                chains.append([sol])
        return chains

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    def _get_idea_pool(self) -> list[dict] | None:
        if self.use_summarizer and self.summarizer:
            return self.summarizer.select_inspirations()
        return None

    def _get_idea_pool_for_layer(self, history_list: list[Solution]) -> list[dict] | None:
        if self.use_summarizer and self.summarizer:
            last_ideas = history_list[-1].metadata.extras.get("Ideas", [])
            return self.summarizer.select_inspirations(last_ideas)
        return None

    def _maybe_summarize_indiv(self, chain: list[Solution]) -> None:
        if len(chain) <= 1:
            return
        if not (self.use_summarizer and self.summarizer):
            return
        last = chain[-1]
        if last.evaluation_res is None or not last.evaluation_res.valid:
            return

        # Only summarize if the last solution is strictly better than all predecessors
        last_fitness = last.evaluation_res.additional_info.get("fitness_list", [float("inf")])[0]
        for sol in chain[:-1]:
            if sol.evaluation_res and sol.evaluation_res.valid:
                fit = sol.evaluation_res.additional_info.get("fitness_list", [float("inf")])[0]
                if fit - last_fitness < 1e-8:
                    return

        start = time.time()
        self.verbose_info("\tSUMMARIZE: ")
        self.summarizer.summarize_indiv(chain)
        self.verbose_info(f"Summarized:{time.time() - start:.1f}s.\n")

    def _maybe_summarize_offspring(
        self,
        offspring_sol: Solution,
        parent_chains: list[list[Solution]],
    ) -> None:
        if not (self.use_summarizer and self.summarizer):
            return
        if offspring_sol.evaluation_res is None or not offspring_sol.evaluation_res.valid:
            return

        offspring_fitness = offspring_sol.evaluation_res.additional_info.get("fitness_list", [float("inf")])[0]
        for chain in parent_chains:
            best_parent = chain[-1]
            if best_parent.evaluation_res and best_parent.evaluation_res.valid:
                p_fit = best_parent.evaluation_res.additional_info.get("fitness_list", [float("inf")])[0]
                if p_fit - offspring_fitness < 1e-8:
                    return

        start = time.time()
        self.verbose_info("\tSUMMARIZE: ")
        self.summarizer.summarize_offspring(parent_chains, [[offspring_sol]])
        self.verbose_info(f"Summarized:{time.time() - start:.1f}s.\n")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_population(self) -> None:
        if not self.verbose:
            return
        pop = self.state.population
        if not pop:
            return

        fitness_strings = []
        for sol in pop:
            if sol.evaluation_res and sol.evaluation_res.valid:
                fl = sol.evaluation_res.additional_info.get("fitness_list", [])
                fs = "[" + ", ".join(f"{x:<10.4f}" if x > 1e-4 else f"{x:<10.4e}" for x in fl) + "]"
            else:
                fs = "None"
            fitness_strings.append(fs)

        max_len = max(len(s) for s in fitness_strings)
        total_len = max_len + 10
        print(f"{'':=<{total_len}}")
        print(f"{'Idx.':<10s}{'Obj.':<{max_len}s}")
        print(f"{'':-<{total_len}}")
        for i, fs in enumerate(fitness_strings):
            print(f"{i:<10d}{fs}")
        print(f"{'':=<{total_len}}")
