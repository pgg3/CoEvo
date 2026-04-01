from evotoolkit.method.population_method import PopulationMethod
from .coevo_state import CoEvoState


class CoEvoMethod(PopulationMethod):
    state_cls = CoEvoState

    def __init__(self, interface, output_path, running_llm, max_generations=97, pop_size=2, num_idea=None, **kwargs):
        self.max_generations = max_generations
        self.num_idea = num_idea or [6, 3, 3]
        super().__init__(interface=interface, output_path=output_path, running_llm=running_llm, pop_size=pop_size, **kwargs)

    def prepare_initialization(self):
        prompt = self.interface.get_init_prompt()
        solutions = []
        for _ in range(self.pop_size):
            response = self.running_llm.call(prompt)
            sol = self.interface.parse_response(response)
            sol = self.task.evaluate(sol)
            solutions.append(sol)
        self.state.population = solutions
        self.state.layer_solutions[0] = solutions

    def step_iteration(self):
        all_solutions = list(self.state.population)
        for layer in range(1, len(self.num_idea)):
            previous_layer = self.state.layer_solutions.get(layer - 1, self.state.population)
            layer_solutions = []
            prompt = self.interface.get_layer_prompt(layer, previous_layer)
            for _ in range(self.num_idea[layer]):
                response = self.running_llm.call(prompt)
                sol = self.interface.parse_response(response)
                sol = self.task.evaluate(sol)
                layer_solutions.append(sol)
            self.state.layer_solutions[layer] = layer_solutions
            all_solutions.extend(layer_solutions)
        valid_solutions = [s for s in all_solutions if s.evaluation_res and s.evaluation_res.valid]
        valid_solutions.sort(key=lambda s: s.evaluation_res.score, reverse=True)
        self.state.population = valid_solutions[:self.pop_size]

    def should_stop_iteration(self):
        return self.state.generation >= self.max_generations
