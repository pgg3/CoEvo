from coevo.tasks.program_design import ProgramTaskInfo

class Oscillation2TaskInfo(ProgramTaskInfo):
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
            f"Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator "\
            f"system with driving force, given data on time, position, and velocity. "
        program_template = \
            f'```python\n' \
            f'import numpy as np\n' \
            f'def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n'\
            f'    """ Mathematical function for acceleration in a damped nonlinear oscillator\n\n'\
            f'    Args:\n' \
            f'        t: A numpy array representing time.\n' \
            f'        x: A numpy array representing observations of current position.\n'\
            f'        v: A numpy array representing observations of velocity.\n'\
            f'        params: Array of numeric constants or parameters to be optimized, there are 10 params.\n'\
            f'    Return:\n'\
            f'        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n'\
            f'    """\n' \
            f'    dv = params[0] * t + params[1] * x  +  params[2] * v +  + params[3]  # Example equation\n' \
            f'    return dv\n'\
            f'```\n'
        super().__init__(
            task_info=task_info,
            program_template=program_template,
            **kwargs
        )