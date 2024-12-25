from .base_rep import BaseRep

class EnglishRep(BaseRep):
    def __init__(self):
        rep_name = "Natural Language English"
        rep_def = "Verbal descriptions of the solution."
        super().__init__(rep_name, rep_def)