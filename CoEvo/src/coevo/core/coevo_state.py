from dataclasses import dataclass, field

from evotoolkit.core import PopulationState, Solution


@dataclass
class CoEvoState(PopulationState):
    """CoEvo algorithm runtime state.

    population: list of the best Solution from each chain (used by base class).
    layer_chains: chain_id -> [Solution_layer0, Solution_layer1, ...] full chain.
    next_chain_id: counter for assigning chain IDs.
    idea_pool: serializable summarizer state (list of idea dicts).
    """

    layer_chains: dict[int, list[Solution]] = field(default_factory=dict)
    next_chain_id: int = 0
    idea_pool: list[dict] = field(default_factory=list)
