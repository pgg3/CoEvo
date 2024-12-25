from .base_rep import BaseRep
class CodeRep(BaseRep):
    def __init__(self):
        rep_name = "Representation in Code"
        rep_def = "A representation of the code."
        super().__init__(rep_name, rep_def)
