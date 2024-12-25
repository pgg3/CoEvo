from coevo.tasks.program_design import ProgramTaskInfo

class BactGrowTaskInfo(ProgramTaskInfo):
    def __init__(self, **kwargs):
        """
        Initializes the ProgramTaskInfo class.
        :param threat_type: The type of threat the adversarial attack is targeting.
        :param to_attack_list: A list of atk_steps
        :param kwargs: Additional keyword arguments.
        """
        task_info = \
            f"You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. " \
            f"Complete the 'equation' function below, considering the physical meaning and relationships of inputs. "\
            f"Find the mathematical function skeleton that represents E. Coli bacterial growth rate, "\
            f"given data on population density, substrate concentration, temperature, and pH level. "
        program_template = \
            f'```python\n' \
            f'import numpy as np\n' \
            f'def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:\n'\
            f'    """ Mathematical function for bacterial growth rate\n\n'\
            f'    Args:\n'\
            f'        b: A numpy array representing observations of population density of the bacterial species.\n'\
            f'        s: A numpy array representing observations of substrate concentration.\n'\
            f'        temp: A numpy array representing observations of temperature.\n'\
            f'        pH: A numpy array representing observations of pH level.\n'\
            f'        params: Array of numeric constants or parameters to be optimized, there are 10 params.\n\n'\
            f'    Return:\n'\
            f'        A numpy array representing bacterial growth rate as the result of applying the mathematical function to the inputs.\n'\
            f'    """\n' \
            f'    grow_rate = params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]  # Example equation\n' \
            f'    return grow_rate\n'\
            f'```\n'
        super().__init__(
            task_info=task_info,
            program_template=program_template,
            **kwargs
        )