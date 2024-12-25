import concurrent

from abc import abstractmethod
from coevo.llm.llm_api_https import HttpsApi

class Agent:
    def __init__(
            self, llm_inst: HttpsApi, task_info_inst, evaluator_inst, verbose=True, **kwargs
    ):
        self.name = "BaseAgent"
        self.llm_inst = llm_inst
        self.task_info_inst = task_info_inst
        self.evaluator_inst = evaluator_inst
        self.header_width = 100
        self.verbose = verbose

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def _print_header(self):
        name_string = f" {self.name} "
        print(f"{name_string:=^{self.header_width}}")

    def _evaluate(self, task_string):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_fitness = executor.submit(self.evaluator_inst.evaluate_task, task_string)
            return future_fitness

    def _prompt_till_valid(self, prompt_content, **kwargs):
        n_retry = 0
        parse_success = False
        while not parse_success:
            response = self.llm_inst.get_response(prompt_content)
            parsed_response, parse_success = self._parse_response(response, **kwargs)

            if parse_success:
                return parsed_response, response, parse_success
            else:
                n_retry += 1

            if n_retry > 3:
                return None, response, False

    @abstractmethod
    def _parse_response(self, response_str, **kwargs):
        raise NotImplementedError
