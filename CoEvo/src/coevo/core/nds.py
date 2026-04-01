import numpy as np

from evotoolkit.core import Solution


def nds_select(solutions: list[Solution], size: int) -> list[Solution]:
    """NDS non-dominated sorting selection operating on Solution objects.

    Mirrors the original nds_evoda logic but operates on Solution objects.
    Fitness objectives come from additional_info["fitness_list"].
    """
    valid = [s for s in solutions if s.evaluation_res and s.evaluation_res.valid]
    invalid = [s for s in solutions if not (s.evaluation_res and s.evaluation_res.valid)]

    if len(valid) < size:
        return valid + invalid[: size - len(valid)]

    # Build multi-objective matrix.
    # Original: importance order is last attack, second to last, ..., first (excluding time at index -1)
    # fitness_list = [mse, time] so atk_len = 1, we sort on fitness_list[0] only
    atk_len = len(valid[0].evaluation_res.additional_info.get("fitness_list", [1.0])) - 1
    if atk_len <= 0:
        atk_len = 1

    all_obj_list = []
    for reverse_atk_step in range(atk_len - 1, -1, -1):
        this_obj_list = [
            sol.evaluation_res.additional_info.get("fitness_list", [float("inf")])[reverse_atk_step]
            for sol in valid
        ]
        all_obj_list.append(this_obj_list)

    all_obj_array = np.array(all_obj_list).T  # shape (n_solutions, n_objectives)

    fronts = fast_non_dominated_sort(all_obj_array)
    all_sorted_indices: list[int] = []
    for front in fronts:
        all_sorted_indices.extend(front)

    selected_indices = all_sorted_indices[:size]
    return [valid[i] for i in selected_indices]


class Dominator:
    @staticmethod
    def get_relation(a, b, cva=None, cvb=None):
        if cva is not None and cvb is not None:
            if cva < cvb:
                return 1
            elif cvb < cva:
                return -1

        val = 0
        for i in range(len(a)):
            if a[i] < b[i]:
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                if val == 1:
                    return 0
                val = -1
        return val

    @staticmethod
    def calc_domination_matrix(F, _F=None, epsilon=0.0):
        if _F is None:
            _F = F

        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = (
            np.logical_and(smaller, np.logical_not(larger)) * 1
            + np.logical_and(larger, np.logical_not(smaller)) * -1
        )
        return M


def fast_non_dominated_sort(F: np.ndarray, dominator: Dominator = Dominator()) -> list[list[int]]:
    M = dominator.calc_domination_matrix(F)

    n = M.shape[0]
    fronts: list[list[int]] = []

    if n == 0:
        return fronts

    n_ranked = 0
    ranked = np.zeros(n, dtype=int)
    is_dominating: list[list[int]] = [[] for _ in range(n)]
    n_dominated = np.zeros(n)
    current_front: list[int] = []

    for i in range(n):
        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1
            n_ranked += 1

    fronts.append(current_front)

    while n_ranked < n:
        next_front: list[int] = []
        for i in current_front:
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1
                    n_ranked += 1
        fronts.append(next_front)
        current_front = next_front

    return fronts
