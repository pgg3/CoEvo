from .base_rep import BaseRep
class VerilogCodeRep(BaseRep):
    def __init__(self):
        rep_name = "Representation in Verilog Code"
        rep_def = "A representation in Verilog code."
        super().__init__(rep_name, rep_def)