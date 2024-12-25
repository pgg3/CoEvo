from ..base_paras import Paras

class EoHParas(Paras):
    def __init__(
            self,
            output_dir: str,
            max_gen: int,
            pop_size: int,
            num_co_crossover: int = 2,
            management: str = "pop_greedy",
            selection: str = "prob_rank",
            load_pop: bool = False,
            load_pop_file: str = None,
            **kwargs
    ):
        """
        Initializes the EoHParas class with the specified parameters.

        Parameters:
        -----------
        output_dir : str
            The directory where the output will be saved.
        max_gen : int
            The maximum number of generations to be processed.
        pop_size : int
            The size of the population.
        num_co_crossover : int, optional
            The number of crossover points to be used. Default is 2.
        management : str, optional
            The management strategy to be used. Default is "pop_greedy".
        selection : str, optional
            The selection strategy to be used. Default is "prob_rank".
        load_pop : bool, optional
            Whether to load the population from a file. Default is False.
        load_pop_file : str, optional
            The file from which to load the population. Default is None.
        """
        super().__init__(output_dir)
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.num_co_crossover = num_co_crossover

        self.management = management
        self.selection = selection

        self.load_pop = load_pop
        self.load_pop_file = load_pop_file
