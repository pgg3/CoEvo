from .base_rep import BaseRep
class MathRep(BaseRep):
    def __init__(self):
        rep_name = "Mathematical Formula"
        rep_def = "Mathematical formula or equation."
        super().__init__(rep_name, rep_def)