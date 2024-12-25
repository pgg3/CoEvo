import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for stress in Aluminium rod. """
    
    # Temperature-dependent Young's modulus
    E = params[0] + params[1] * temp + params[2] * temp**2  # E(T) = E0 + k1 * T + k2 * T^2
    
    # Temperature-dependent yield strength
    sigma_y = params[3] + params[4] * temp + params[5] * temp**2  # sigma_y(T)
    
    # Define yield strain
    yield_strain = sigma_y / E
    
    # Calculate stress based on whether we are in the elastic or plastic region
    stress = np.where(
        strain <= yield_strain,
        E * strain,  # Elastic region
        sigma_y + params[6] * (strain - yield_strain)**params[7]  # Plastic region with hardening
    )
    
    return stress