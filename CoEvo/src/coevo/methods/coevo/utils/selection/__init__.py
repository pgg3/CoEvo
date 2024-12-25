import random


def probabilistic_sampling(lst, m, p):
    sampled = [item for item, prob in zip(lst, p) if random.random() < prob]

    while len(sampled) < m:
        remaining = [item for item in lst if item not in sampled]
        additional = random.sample(remaining, m - len(sampled))
        sampled.extend(additional)

    if len(sampled) > m:
        sampled = random.sample(sampled, m)

    return sampled


def prob_rank(pop,m):
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    # Normalize probabilities
    probs = [p / sum(probs) for p in probs]
    parents = probabilistic_sampling(pop, m, p=probs)
    return parents

class Selection:
    selection_dict = {
        "prob_rank": prob_rank
    }