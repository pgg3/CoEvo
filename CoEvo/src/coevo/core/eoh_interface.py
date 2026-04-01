from evotoolkit.task.python_task import EoHPythonInterface

# Direct re-export. CoEvo tasks use the standard EoHPythonInterface since
# each task's TaskSpec.prompt already contains the full task description and
# program template that EoHPythonInterface needs for its I1/E1/E2/M1/M2 operators.
CoEvoEoHInterface = EoHPythonInterface

__all__ = ["CoEvoEoHInterface"]
