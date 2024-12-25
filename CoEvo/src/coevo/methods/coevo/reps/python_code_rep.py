from .base_rep import BaseRep
class PythonCodeRep(BaseRep):
    def __init__(self):
        rep_name = "Python Code"
        rep_def = "Runnable python code implementation."
        super().__init__(rep_name, rep_def)