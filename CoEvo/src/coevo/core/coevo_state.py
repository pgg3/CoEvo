from dataclasses import dataclass, field
from evotoolkit.core.state import PopulationState


@dataclass
class CoEvoState(PopulationState):
    current_layer: int = 0
    layer_solutions: dict = field(default_factory=dict)
    pareto_front: list = field(default_factory=list)
    num_idea: list = field(default_factory=lambda: [6, 3, 3])
