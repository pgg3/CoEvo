import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        t: A numpy array representing time.
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized, there are 10 params.
    
    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    # Unpack parameters
    c_d = params[0]  # Cubic damping coefficient
    k_0 = params[1]  # Base stiffness constant
    k_v = params[2]  # Velocity dependent stiffness
    k_x = params[3]  # Position dependent stiffness
    alpha = params[4]  # Feedback coefficient
    F_0 = params[5]  # Amplitude of the driving force
    omega = params[6]  # Frequency of the driving force
    phi = params[7]   # Phase of the driving force
    v_scale = params[8]  # Scaling factor for acceleration
  
    # Damping force with cubic term
    damping_force = -c_d * v**3  # Nonlinear damping
    
    # Restorative force with adaptive stiffness
    restoring_force = - (k_0 + k_v * v + k_x * x) * x  # Adaptive stiffness

    # Feedback control mechanism
    feedback_control = -alpha * (v**2 + x**2)  # Feedback term

    # Driving force
    driving_force = F_0 * np.sin(omega * t + phi)  # External driving force
    
    # Total acceleration
    acceleration = (damping_force + restoring_force + feedback_control + driving_force) / v_scale
    return acceleration