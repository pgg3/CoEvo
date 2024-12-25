import os.path

from ..base_paras import Paras
from .reps import BaseRep
from coevo.llm.llm_api_https import HttpsApi

class CoEvoParas(Paras):
    def __init__(
            self,
            output_dir: str,
            max_gen: int,
            pop_size: int,
            num_init_gen: int,
            management: str="nds_evoda",
            selection: str="prob_rank",
            num_crossover: int=2,
            load_pop: bool = False,
            load_pop_path: str = None,
            save_full_history: bool = True,

            use_summary:bool = False,
            llm_summarizer_inst: HttpsApi = None,
            tokenizer_path: str = None,
            num_idea_to_return: int = 5,
            pool_size:int = 100,
            cluster_summary:bool=True,
            do_summary:bool=True,
            load_summary=False,
            load_summary_path: str = None,

            rep_list: list = None,
            rep_use: BaseRep = None,
            num_idea: list[int] = None,

            log_file: str = None,
            cmd_log_level: str = None,
            file_log_level: str = None,
            use_profiler=False,
            **kwargs
    ):
        super().__init__(output_dir)
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.num_init_gen = num_init_gen
        self.management = management
        self.selection = selection
        self.num_crossover = num_crossover

        self.load_pop = load_pop
        self.load_pop_path = load_pop_path
        if self.load_pop:
            assert self.load_pop_path is not None, "load_pop_path must be specified if load_pop is True"

        self.save_full_history = save_full_history

        self.use_summary = use_summary
        self.llm_summarizer_inst = llm_summarizer_inst
        self.tokenizer_path = tokenizer_path
        self.num_idea_to_return = num_idea_to_return
        self.pool_size = pool_size
        self.cluster_summary = cluster_summary
        self.do_summary = do_summary


        self.load_summary = load_summary
        self.load_summary_path = load_summary_path
        if self.load_summary:
            assert self.load_summary_path is not None, "load_summary_path must be specified if load_summary is True"

        self.rep_list = rep_list
        self.rep_use = rep_use
        self.num_idea = num_idea

        if log_file is None:
            log_file = os.path.join(output_dir, "evo.log")
        self.log_file = log_file
        self.cmd_log_level = "ERROR" if cmd_log_level is None else cmd_log_level
        self.file_log_level = "DEBUG" if file_log_level is None else file_log_level

        self.use_profiler = use_profiler

