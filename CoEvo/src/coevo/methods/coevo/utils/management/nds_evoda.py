import numpy as np
def nds_evoda(population, size):

    # Check their error message and Filter out error population
    no_error_population = []
    error_population = []
    for single_pop in population:
        if single_pop[-1]['error_msg'] is None:
            no_error_population.append(single_pop)
        else:
            error_population.append(single_pop)

    if len(no_error_population) < size:
        return_population = no_error_population + error_population[:size - len(no_error_population)]
        return return_population
    else:
        return_population = no_error_population

    # Non-dominated sorting, Importance: Last attack, second to last attack,..., first attack, time
    atk_len = len(return_population[0][-1]['fitness_list']) - 1
    all_obj_list = []
    for reverse_atk_step in range(atk_len-1, -1, -1):
        this_obj_list = []
        for each_indiv in return_population:
            this_obj_list.append(each_indiv[-1]['fitness_list'][reverse_atk_step])
        all_obj_list.append(this_obj_list)

    # time_obj_list = []
    # for each_indiv in population:
    #     time_obj_list.append(each_indiv[-1]['fitness_list'][-1])
    #
    # all_obj_list.append(time_obj_list)

    all_obj_list = np.array(all_obj_list).T

    fronts = fast_non_dominated_sort(all_obj_list)
    all_sorted = []
    for front in fronts:
        all_sorted.extend(front)

    all_sorted = all_sorted[:size]

    pop_new = [return_population[i] for i in all_sorted]

    return pop_new

def get_relation(ind_a, ind_b):
    return Dominator.get_relation(ind_a.F, ind_b.F, ind_a.CV[0], ind_b.CV[0])


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
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1
        return val

    @staticmethod
    def calc_domination_matrix_loop(F, G):
        n = F.shape[0]
        CV = np.sum(G * (G > 0).astype(float), axis=1)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = Dominator.get_relation(F[i, :], F[j, :], CV[i], CV[j])
                M[j, i] = -M[i, j]

        return M

    @staticmethod
    def calc_domination_matrix(F, _F=None, epsilon=0.0):

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1

        # if cv equal then look at dom
        # M = constr + (constr == 0) * dom

        return M
def fast_non_dominated_sort(F, dominator=Dominator(), **kwargs):
    if "dominator" in kwargs:
        M = Dominator.calc_domination_matrix(F)
    else:
        M = dominator.calc_domination_matrix(F)

    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

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
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts