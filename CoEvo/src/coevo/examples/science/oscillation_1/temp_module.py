import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    restoring_force = params[0] * x
    damping_force = -params[1] * v
    nonlinear_restoring = params[2] * x**3
    periodic_force = params[3] * np.cos(x)
    nonlinear_damping = params[4] * np.sin(v)
    coupling = params[5] * x * v
    exponential_damping = params[6] * np.exp(-params[7] * x)
    nonlinear_velocity = params[8] * v**3
    constant_driving = params[9]
    
    # Calculate the total acceleration by summing all components
    acceleration = (
        restoring_force +
        damping_force +
        nonlinear_restoring +
        periodic_force +
        nonlinear_damping +
        coupling +
        exponential_damping +
        nonlinear_velocity +
        constant_driving
    )
    
    return acceleration  # Return the calculated acceleration for the given system.

