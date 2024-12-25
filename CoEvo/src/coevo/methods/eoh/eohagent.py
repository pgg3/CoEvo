import os
import re
import json

from coevo.utils.string_formatter import format_float_or_none

from ..base_agent import Agent
from .eohparas import EoHParas
from .eohprompts import EoHPrompt
from .utils.management import Management
from .utils.selection import Selection


class EoHAgent(Agent):
    def __init__(self, llm_inst, task_info_inst, evaluator_inst, paras: EoHParas, verbose=True, **kwargs):
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
        self.name = "EoHAgent"
        self.paras = paras

        self.management = Management.management_dict[self.paras.management]
        self.selection = Selection.selection_dict[self.paras.selection]

        self.ec_operators = ["e1", "e2", "m1", "m2"]
        self.cross_operators = ["e1", "e2"]
        self.mutate_operators = ["m1", "m2"]
        self.operator_dict = {
            "e1": EoHPrompt.get_prompt_e1,
            "e2": EoHPrompt.get_prompt_e2,
            "m1": EoHPrompt.get_prompt_m1,
            "m2": EoHPrompt.get_prompt_m2
        }


    def run(self):

        self._print_header()

        # initialization
        population = []
        if self.paras.load_pop:
            print("Loading initial population...")
            with open(self.paras.load_pop_file) as file:
                data = json.load(file)
            for individual in data:
                population.append(individual)
            print("Initial population loaded.")
            self._print_gen_header(0)
            self._print_population(population)
            self._save_population(population, 0)
        else:
            print("Creating initial population...")
            for _ in range(2):
                for pop_i in range(self.paras.pop_size):
                    prompts_content = EoHPrompt.get_prompt_i1(self.task_info_inst)
                    population.append(self._get_code_and_evaluate(prompts_content))
            self.renew_evaluate_res(population)
            population = self.management(population, self.paras.pop_size)
            print("Initial population created.")
            self._print_gen_header(0)
            self._print_population(population)
            self._save_population(population, 0)

        for gen_idx in range(1, self.paras.max_gen):
            self._print_gen_header(gen_idx)
            for each_op in self.ec_operators:
                parents, prompts_content = self._get_operation_prompts(population, each_op)
                new_pop = self._get_code_and_evaluate(prompts_content)
                if self.verbose:
                    self.renew_evaluate_res(new_pop)
                    self._print_op_res(each_op, parents, new_pop)
                population.append(new_pop)
            if not self.verbose:
                self.renew_evaluate_res(population)
            population = self.management(population, self.paras.pop_size)
            self._print_population(population)
            self._save_population(population, gen_idx)

    def _get_operation_prompts(self, population, op):
        if op in self.cross_operators:
            parents = self.selection(population, self.paras.num_co_crossover)
        elif op in self.mutate_operators:
            parents = self.selection(population, 1)
        else:
            raise ValueError(f"Unknown operation: {op}")
        return parents, self.operator_dict[op](self.task_info_inst, parents)

    def renew_evaluate_res(self, population):
        if isinstance(population, list):
            for each_indiv in population:
                each_indiv["objective"] = each_indiv["objective"].result()
        elif isinstance(population, dict):
            population["objective"] = population["objective"].result()
        else:
            raise ValueError(f"Unknown population type: {type(population)}")

    def _get_code_and_evaluate(self, prompts_content):
        try:
            parsed_response, original_response, parse_success = self._prompt_till_valid(prompts_content)
            if parse_success:
                description = parsed_response[0]
                code = parsed_response[1]
                new_pop = {
                    "description": description,
                    "code": code,
                    "objective": self._evaluate(code)
                }
            else:
                new_pop = {
                    "description": None,
                    "code": None,
                    "objective": None
                }
        except Exception as e:
            new_pop = {
                "description": None,
                "code": None,
                "objective": None
            }
        return new_pop

    def _parse_response(self, response_str: str, **kwargs):
        description = re.findall(r"\{(.*)\}", response_str, re.DOTALL)

        if len(description) == 0:
            if '```' in response_str:
                description = re.findall(r'^(.*?)```', response_str, re.DOTALL)
            elif 'import' in response_str:
                description = re.findall(r'^.*?(?=import)', response_str, re.DOTALL)
            else:
                description = re.findall(r'^.*?(?=def)', response_str, re.DOTALL)

        code = re.findall(r"```[pP]ython(.*?)```", response_str, re.DOTALL)
        if len(description) == 0 or len(code) == 0:
            return [None, None], False
        else:
            return [description[0], code[0]], True

    def _save_population(self, population, gen_idx):
        pop_dir = os.path.join(self.paras.output_dir, "pops")
        if not os.path.exists(pop_dir):
            os.makedirs(pop_dir)
        pop_file = os.path.join(pop_dir, f"pop_{gen_idx}.json")
        with open(pop_file, "w") as f:
            #noinspection PyTypeChecker
            json.dump(population, f, indent=4)

    def _print_population(self, population):
        print(f"{'':=<20}")
        print(f"{'Idx.':<10s}{'Obj.':<10s}")
        print(f"{'':-<20}")
        for pop_idx in range(len(population)):
            obj_string = format_float_or_none(population[pop_idx]['objective'], width=10, sig=4, align="<")
            print(f"{pop_idx:<10d}{obj_string}")
        print(f"{'':=<20}")

    def _print_op_res(self, op, parents, new_pop):
        print(f"Applying {op}...", end=" ")
        parents_string = [f"p{i+1}.obj" for i in range(len(parents))]
        parents_obj_string = [
            format_float_or_none(parents[i]['objective'], width=6, sig=4, align="<")
            for i in range(len(parents))
        ]
        new_obj_string = format_float_or_none(new_pop['objective'], width=6, sig=4, align="<")
        for i in range(len(parents)):
            print(f"{parents_string[i]}: {parents_obj_string[i]}", end=" ")
        print(f"-> new.obj: {new_obj_string}")

    def _print_gen_header(self, gen_idx):
        gen_string = f" Gen {gen_idx} "
        print(f"{gen_string:-^{self.header_width}}")

