from ..base import TaskInfo

class ProgramTaskInfo(TaskInfo):
    def __init__(
            self, task_info,
            program_template, other_info=None,
            **kwargs
    ):

        super().__init__(task_info, **kwargs)
        self.program_template = program_template
        self.other_info = other_info
        self.task_info = \
            f"{self.task_info}\n"\
            f"Program Template:\n{program_template}\n"
        if self.other_info:
            self.task_info += f"Other Info.:\n{self.other_info}\n"