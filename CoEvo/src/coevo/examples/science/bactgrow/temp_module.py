import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    density_effect = params[5] * b * s
    substrate_utilization = s / (params[1] + s)
    metabolic_response = np.exp(-((temp - 37) ** 2) / (2 * (params[2] ** 2))) * np.exp(-((pH - 7) ** 2) / (2 * (params[3] ** 2)))
    
    grow_rate = density_effect + substrate_utilization * density_effect + metabolic_response
    
    return grow_rate