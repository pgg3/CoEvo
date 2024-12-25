from coevo.tasks.program_design import ProgramTaskInfo

class StressStrainTaskInfo(ProgramTaskInfo):
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
            f"Find the mathematical function skeleton that represents stress, given data on strain and temperature "\
            f"in an Aluminium rod for both elastic and plastic regions."
        program_template = \
            f'```python\n' \
            f'import numpy as np\n' \
            f'def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:\n'\
            f'    """ Mathematical function for stress in Aluminium rod\n\n'\
            f'    Args:\n' \
            f'        strain: A numpy array representing observations of strain.\n' \
            f'        temp: A numpy array representing observations of temperature.\n'\
            f'        params: Array of numeric constants or parameters to be optimized, there are 10 params.\n\n'\
            f'    Return:\n'\
            f'        A numpy array representing stress as the result of applying the mathematical function to the inputs.\n'\
            f'    """\n' \
            f'    stress = params[0] * strain  +  params[1] * temp  # Example equation\n' \
            f'    return stress\n'\
            f'```\n'
        super().__init__(
            task_info=task_info,
            program_template=program_template,
            **kwargs
        )