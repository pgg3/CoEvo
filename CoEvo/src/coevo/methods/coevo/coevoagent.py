import os
import re
import time
import json
import torch
import concurrent
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter


from coevo.utils.string_formatter import format_float_or_none, format_str_or_none, format_list_float_or_none

from ..base_agent import Agent
from .coevoparas import CoEvoParas
from .coevoprompts import CoEvoPrompt
from .coevosummarizer import CoEvoSummarizer
from .utils.management import Management
from .utils.selection import Selection
from ..base_logger import BaseLogger



class CoEvoAgent(Agent):
    def __init__(self, llm_inst, task_info_inst, evaluator_inst, paras: CoEvoParas, verbose=True, **kwargs):
        """

        Parameters
        ----------
        :params llm_inst: the instance of the LLM model
        :params task_info_inst: the instance of the task information
        :params evaluator_inst: the instance of the evaluator
        :params paras: the parameters of the agent
        :params verbose: whether to print the details. This will prevent the multiprocessing from printing the details.
        """
        super().__init__(llm_inst, task_info_inst, evaluator_inst, verbose=verbose, **kwargs)
        self.name = "CoEvoAgent"
        self.paras = paras

        self.management = Management.management_dict[self.paras.management]
        self.selection = Selection.selection_dict[self.paras.selection]
        self.generate_sol_mode = [
            "crossover_positive", "crossover_negative", "mutation_positive", "mutation_negative"
        ]

        self.logger = BaseLogger(self.paras.log_file, self.paras.cmd_log_level, self.paras.file_log_level)

        if self.paras.use_profiler:
            self._init_profiler()

        self.full_history_step = 0

    def _init_profiler(self):
        self.profiler = SummaryWriter(os.path.join(self.paras.output_dir, "logs"))
        self.global_profile_step = 0
        self.global_best_list = []
        self.global_num_invalid_sol = 0
        self.global_num_valid_sol = 0
        self.global_num_invalid_each_layer = [0 for _ in range(len(self.paras.num_idea))]
        self.global_num_valid_each_layer = [0 for _ in range(len(self.paras.num_idea))]

    def _step_profiler(self, parsed_prompt, idea_layer_i, mode="init"):
        # three modes: init, offspring, continue_reason

        if parsed_prompt["error_msg"] is not None:
            self.global_num_invalid_sol += 1
            self.global_num_invalid_each_layer[idea_layer_i] += 1
        else:
            self.global_num_valid_sol += 1
            self.global_num_valid_each_layer[idea_layer_i] += 1
            if len(self.global_best_list) == 0:
                self.global_best_list = parsed_prompt["fitness_list"]
            else:
                for fit_idx in range(len(self.global_best_list)):
                    if parsed_prompt["fitness_list"][fit_idx] < self.global_best_list[fit_idx]:
                        self.global_best_list[fit_idx] = parsed_prompt["fitness_list"][fit_idx]

            for fit_idx in range(len(parsed_prompt["fitness_list"])):
                self.profiler.add_scalar(
                    f"mode_fit/fitness_{fit_idx}_{mode}", parsed_prompt["fitness_list"][fit_idx], self.global_profile_step
                )

        self.profiler.add_scalar("global_num_invalid_sol", self.global_num_invalid_sol, self.global_profile_step)
        self.profiler.add_scalar("global_num_valid_sol", self.global_num_valid_sol, self.global_profile_step)
        for layer_idx in range(len(self.paras.num_idea)):
            self.profiler.add_scalar(
                f"global_num_invalid_each_layer_{layer_idx}", self.global_num_invalid_each_layer[layer_idx], self.global_profile_step
            )
            self.profiler.add_scalar(
                f"global_num_valid_each_layer_{layer_idx}", self.global_num_valid_each_layer[layer_idx], self.global_profile_step
            )

        for fitness_idx in range(len(self.global_best_list)):
            self.profiler.add_scalar(
                f"fitness_{fitness_idx}", self.global_best_list[fitness_idx], self.global_profile_step
            )

        self.global_profile_step += 1



    def run(self):
        self._print_header()
        self._init_summary()

        # initialization
        self._print_gen_header(0)
        population = self._init_population()
        population = self.management(population, self.paras.pop_size)
        self._print_population(population)
        self._save_population(population, 0)
        self._save_summary(0)

        for gen_idx in range(1, self.paras.max_gen):
            self._print_gen_header(gen_idx)
            new_solutions = []
            for _ in range(self.paras.num_init_gen):
                one_solution = self._init_a_sol()
                new_solutions.append(one_solution)
            for each_mode in self.generate_sol_mode:
                new_sol = self._generate_new_sol(population, mode=each_mode)
                new_solutions.append(new_sol)

            population.extend(new_solutions)
            population = self.management(population, self.paras.pop_size)
            self._print_population(population)
            self._save_population(population, gen_idx)
            self._save_summary(gen_idx)

    def _verbose_info(self, info_str: str):
        if self.verbose:
            self.logger.logger.info(info_str)
            print(info_str, end="")
        else:
            self.logger.logger.info(info_str)

    def _init_summary(self):
        if self.paras.use_summary:
            self.summarizer = CoEvoSummarizer(self.task_info_inst, self.paras)
        else:
            self.summarizer = None
            return

        if self.paras.load_summary:
            with open(self.paras.load_summary_path, "r") as f:
                summary_content = json.load(f)
            self.summarizer.load_summary(summary_content)
            self._verbose_info("Summary loaded.\n")
        else:
            self._verbose_info("Using empty summary.\n")

    def _init_population(self):
        print("Loading initial population..." if self.paras.load_pop else "Creating initial population...")
        if self.paras.load_pop:
            population = self._load_population()
        else:
            population = []
            for _ in range(2):
                for pop_i in range(self.paras.pop_size):
                    one_solution = self._init_a_sol()
                    population.append(one_solution)
        for _ in range(self.paras.pop_size - len(population)):
            one_solution = self._init_a_sol()
            population.append(one_solution)
        print("Initial population created." if not self.paras.load_pop else "Initial population loaded.")
        return population

    def _init_a_sol(self):
        start_time = time.time()
        self._verbose_info("INIT: ")
        prompts_content = CoEvoPrompt.get_init_prompt(
            self.task_info_inst, self.paras, use_summary=self.paras.use_summary, summarizer=self.summarizer
        )
        finalized_sol = self._parse_and_evaluate(prompts_content, start_time)
        if self.paras.use_profiler:
            self._step_profiler(finalized_sol, 0, mode="init")
        new_gen_sol = self._continue_reason(finalized_sol)
        to_return_sol = [finalized_sol]
        to_return_sol.extend(new_gen_sol)
        self._summarize_indiv(to_return_sol)
        self._save_full_history(to_return_sol)
        return to_return_sol

    def _summarize_indiv(self, to_return_sol):
        if len(to_return_sol) > 1 and to_return_sol[-1]['error_msg'] is None and self.paras.use_summary and self.paras.do_summary:
            last_best_flag = True

            for each_sol in range(len(to_return_sol)-1):
                if to_return_sol[each_sol]['error_msg'] is None:
                    if to_return_sol[each_sol]['fitness_list'][-2] - to_return_sol[-1]['fitness_list'][-2] < 1e-8:
                        last_best_flag = False
                        break

            if last_best_flag:
                start_time = time.time()
                self._verbose_info("\tSUMMARIZE: ")
                self.summarizer.summarize_indiv(to_return_sol)
                self._verbose_info(f"Summarized:{time.time() - start_time:.1f}s.\n")

    def _continue_reason(self, one_solution):
        history_list = [one_solution]
        new_gen_list = []
        for idea_layer_i in range(len(self.paras.num_idea)):
            if idea_layer_i == 0:
                continue
            start_time = time.time()
            self._verbose_info("\tCONTINUE: ")
            prompts_content = CoEvoPrompt.get_continue_prompt(
                self.task_info_inst, self.paras, idea_layer_i, history_list,
                use_summary=self.paras.use_summary, summarizer=self.summarizer
            )
            finalized_sol = self._parse_and_evaluate(prompts_content, start_time, idea_layer_i=idea_layer_i)
            if self.paras.use_profiler:
                self._step_profiler(finalized_sol, idea_layer_i, mode="continue_reason")
            # If invalid, early stop
            if finalized_sol['error_msg'] is not None:
                return new_gen_list
            elif history_list[-1]['error_msg'] is None:
                if history_list[-1]['fitness_list'][-2] - finalized_sol['fitness_list'][-2] < 1e-8:
                    return new_gen_list
            new_gen_list.append(finalized_sol)
            history_list.append(one_solution)
        return new_gen_list

    def _generate_new_sol(self, population, mode:str):
        op_mode, state_mode = mode.split("_")
        if op_mode == "crossover":
            parents = self.selection(population, self.paras.num_crossover)
        elif op_mode == "mutation":
            parents = self.selection(population, 1)
        else:
            raise ValueError("Unknown operation mode: {}".format(op_mode))

        new_sol = self._gen_offspring(parents, mode=mode)
        self._save_full_history(new_sol)
        # capitalize the mode
        if self.verbose:
            self._print_generate_new(mode.upper(), parents, new_sol)
        return new_sol

    def _gen_offspring(self, parents, mode:str="crossover_positive"):
        start_time = time.time()
        self._verbose_info(f"{mode.upper()} ")
        prompts_content = CoEvoPrompt.get_offspring_prompt(
            self.task_info_inst, self.paras, parents, mode=mode, use_summary=self.paras.use_summary,
            summarizer=self.summarizer
        )
        finalized_sol = self._parse_and_evaluate(prompts_content, start_time)
        if self.paras.use_profiler:
            self._step_profiler(finalized_sol, 0, mode="offspring")

        self._summarize_offspring(finalized_sol, parents)

        new_gen_sol = self._continue_reason(finalized_sol)
        to_return_sol = [finalized_sol]
        to_return_sol.extend(new_gen_sol)
        self._summarize_indiv(to_return_sol)
        return to_return_sol

    def _summarize_offspring(self, finalized_sol, parents):
        if finalized_sol['error_msg'] is None and self.paras.use_summary and self.paras.do_summary:
            last_best_flag = True

            for each_parent in parents:
                if each_parent[-1]["error_msg"] is None:
                    if each_parent[-1]['fitness_list'][-2] - finalized_sol['fitness_list'][-2] < 1e-8:
                        last_best_flag = False
                        break

            if last_best_flag:
                start_time = time.time()
                self._verbose_info("\tSUMMARIZE: ")
                self.summarizer.summarize_offspring(parents, [[finalized_sol]])
                self._verbose_info(f"Summarized:{time.time() - start_time:.1f}s.\n")

    def _parse_and_evaluate(self, prompts_content: str, start_time: float, idea_layer_i=0, **kwargs):
        parsed_prompt, response_content, _ = self._prompt_till_valid(prompts_content, idea_layer_i=idea_layer_i, **kwargs)
        parse_time = time.time()
        self._verbose_info(f"Response Get and Parsed:{parse_time - start_time:.1f}s, ")

        parsed_prompt = self._renew_parsed_prompt_with_result(parsed_prompt, response_content)
        evaluation_time = time.time()
        self._verbose_info(f"Evaluated:{evaluation_time - parse_time:.1f}s\n")


        return parsed_prompt

    def _renew_parsed_prompt_with_result(self, parsed_prompt, response_content):
        parsed_prompt["raw_content"] = response_content
        if not self.paras.rep_use.rep_name in parsed_prompt["Solutions"]:
            parsed_prompt["error_msg"] = "Parser Error, Response NOT VALID for evaluation."
            parsed_prompt["fitness_string"] = ""
            parsed_prompt["fitness_list"] = []
            return parsed_prompt
        string_to_eval = parsed_prompt["Solutions"][self.paras.rep_use.rep_name]
        fitness_list, fitness_string, success_flag = self.evaluator_inst.evaluate_task(string_to_eval)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     = executor.submit(self.evaluator_inst.evaluate_task, string_to_eval).result()

        if not success_flag:
            parsed_prompt["error_msg"] = fitness_string
        else:
            parsed_prompt["error_msg"] = None

        parsed_prompt["fitness_string"] = fitness_string
        if len(fitness_list) == 0:
            parsed_prompt["fitness_list"] = []
        else:
            parsed_prompt["fitness_list"] = []
            parsed_prompt["fitness_list"].extend(fitness_list)
        return parsed_prompt

    def _parse_response(self, response_str: str, idea_layer_i: int=0, **kwargs):
        try:
            parsed_dict = {
                "Ideas": [],
                "Thoughts": "",
                "Solutions": {
                }
            }
            inspiration_heading = r"(?:idea|Idea|IDEA)[sS]?\s*:?"
            thought_heading = r"(?:thought|Thought|THOUGHT)[sS]?\s*:?"
            solution_heading = r"(?:solution|Solution|SOLUTION)[sS]?\s*:?"

            # Parse Ideas
            ideas_pattern = re.compile(r"##\s*" + inspiration_heading + r"\s*(.*?)##\s*" + thought_heading, re.DOTALL)
            match = ideas_pattern.search(response_str)
            parsed_ideas = []
            if match:
                ideas_text = match.group(1).strip()
                if idea_layer_i == 0:
                    inspiration_pattern = re.compile(
                        r"(?:name|Name|NAME)\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:reasoning|Reasoning|REASONING|reason|Reason|REASON)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:definition|Definition|DEFINITION)[sS]?\s*:?\s*(.*?)(?=\n|$)", re.DOTALL
                    )
                    inspirations = inspiration_pattern.findall(ideas_text)
                    for inspiration in inspirations:
                        parsed_ideas.append({
                            "Name": inspiration[0].strip(),
                            "Reasoning": inspiration[1].strip(),
                            "Definition": inspiration[2].strip()
                        })
                else:
                    inspiration_pattern = re.compile(
                        r"(?:quote|Quote|QUOTE)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:implication|Implication|IMPLICATION)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:name|Name|NAME)\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:reasoning|Reasoning|REASONING|reason|Reason|REASON)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                        r"(?:definition|Definition|DEFINITION)[sS]?\s*:?\s*(.*?)(?=\n|$)", re.DOTALL
                    )
                    inspirations = inspiration_pattern.findall(ideas_text)
                    for inspiration in inspirations:
                        parsed_ideas.append({
                            "Quote": inspiration[0].strip(),
                            "Implication": inspiration[1].strip(),
                            "Name": inspiration[2].strip(),
                            "Reasoning": inspiration[3].strip(),
                            "Definition": inspiration[4].strip()
                        })
            parsed_dict["Ideas"] = parsed_ideas

            # Extract thoughts
            thoughts_pattern = re.compile(r"##\s*" + thought_heading + r"\s*(.*?)##\s*" + solution_heading, re.DOTALL)
            thoughts = thoughts_pattern.findall(response_str)
            parsed_dict["Thoughts"] = thoughts[0] if not len(thoughts) == 0 else ""

            # Extract solutions
            solutions_pattern = re.compile(r"##\s*" + solution_heading + r"\s*(.*?)$", re.DOTALL)

            solutions = solutions_pattern.search(response_str)
            if solutions:
                solutions_string = solutions.group(1).strip()
                for rep_idx in range(len(self.paras.rep_list)):
                    rep_match_pattern = r"###\s*" + self.paras.rep_list[rep_idx].rep_name + r"\s*:?\s*(.*?)(?=###\s*|$)"
                    rep_match = re.search(rep_match_pattern, solutions_string, re.DOTALL)
                    if rep_match:
                        rep_match_string = rep_match.group(1).strip()
                        if rep_match_string.startswith("```"):
                            rep_quote_match_string = r"```[a-zA-Z]*\n(.*?)(?=```|$)"
                            rep_quote_string = re.search(rep_quote_match_string, rep_match_string, re.DOTALL)
                            if rep_quote_string:
                                rep_match_string = rep_quote_string.group(1).strip()
                                parsed_dict["Solutions"][self.paras.rep_list[rep_idx].rep_name] = rep_match_string
                        else:
                            parsed_dict["Solutions"][self.paras.rep_list[rep_idx].rep_name] = rep_match_string
                    else:
                        print(f"Error parsing response: No solution found")
                        return None, False
            else:
                print(f"Error parsing response: No solution found")
                return None, False
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None, False
        return parsed_dict, True

    def _print_gen_header(self, gen_idx):
        gen_string = f" Gen {gen_idx} "
        print(f"{gen_string:-^{self.header_width}}")

    def _save_population(self, population, gen_idx):
        pop_dir = os.path.join(self.paras.output_dir, "pops", f"gen_{gen_idx}")
        if not os.path.exists(pop_dir):
            os.makedirs(pop_dir)
        for indiv_id, indiv in enumerate(population):
            indiv_file = os.path.join(pop_dir, f"indiv_{indiv_id}.json")
            with open(indiv_file, "w") as f:
                json.dump(indiv, f, indent=4)

    def _print_population(self, population):
        obj_string_list = []
        for pop_idx in range(len(population)):
            obj_string = format_list_float_or_none(population[pop_idx][-1]['fitness_list'], width=10, sig=4, align="<")
            obj_string_list.append(obj_string)
        # get max string len
        max_string_len = max([len(obj_string) for obj_string in obj_string_list])
        obj_string_len = max_string_len
        total_string_len = obj_string_len + 10

        print(f"{'':=<{total_string_len}}")
        print(f"{'Idx.':<10s}{'Obj.':<{max_string_len}s}")
        print(f"{'':-<{total_string_len}}")
        for pop_idx in range(len(population)):
            print(f"{pop_idx:<10d}{obj_string_list[pop_idx]}")
        print(f"{'':=<{total_string_len}}")

    def _load_population(self):
        load_pop_path = self.paras.load_pop_path
        all_file_under_path = os.listdir(load_pop_path)
        all_file_under_path = natsorted(all_file_under_path)
        pop_list = []
        for each_file in all_file_under_path:
            full_file_path = os.path.join(load_pop_path, each_file)
            pop_name_pattern = re.compile(r"indiv_\d+\.json", re.DOTALL)
            match_res = pop_name_pattern.search(full_file_path)
            if match_res:
                with open(full_file_path, "r") as f:
                    pop_list.append(json.load(f))
        return pop_list

    def _print_generate_new(self, identifier, parents, children):
        if len(parents) > 1:
            parents_str = []
            for each_parent in parents:
                parents_str.append(format_list_float_or_none(each_parent[-1]['fitness_list'], width=10, sig=4, align="<"))
            print_str = f"\t{identifier}:"
            for each_parent_str in parents_str:
                print_str += f" {each_parent_str}"
            print_str += f" -> {format_list_float_or_none(children[-1]['fitness_list'], width=10, sig=4, align='<')}"
        else:
            print_str = \
                f"\t{identifier}: {format_list_float_or_none(parents[0][-1]['fitness_list'], width=10, sig=4, align='<')} "\
                f"-> {format_list_float_or_none(children[-1]['fitness_list'], width=10, sig=4, align='<')}"

        print(print_str)

    def _save_summary(self, gen_idx):
        if self.paras.use_summary:
            pop_dir = os.path.join(self.paras.output_dir, "pops", f"gen_{gen_idx}")
            if not os.path.exists(pop_dir):
                os.makedirs(pop_dir)
            summary_file = os.path.join(pop_dir, f"summary.json")
            with open(summary_file, "w") as f:
                json.dump(self.summarizer.idea_pool, f, indent=4)

    def _save_full_history(self, indiv):
        if self.paras.save_full_history:
            history_dir = os.path.join(self.paras.output_dir, "history")
            if not os.path.exists(history_dir):
                os.makedirs(history_dir)
            indiv_file = os.path.join(history_dir, f"indiv_{self.full_history_step}.json")
            with open(indiv_file, "w") as f:
                json.dump(indiv, f, indent=4)
            self.full_history_step += 1