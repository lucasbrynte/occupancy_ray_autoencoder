import numpy as np
from scipy.special import comb
# import itertools

def k_random_pairs(n_individuals, n_pairs):
    max_pairs_possible = comb(n_individuals, 2, exact=True) # n_individuals choose 2 = n_individuals*(n_individuals-1)/2
    assert not n_pairs > max_pairs_possible
    indices_selected_pairs = np.random.choice(max_pairs_possible, size=(n_pairs,), replace=False)
    # NOTE: If the number of pairs sampled are very few in relation to all possible pairs, it would be much faster to sample a number of pairs with replacement (e.g. incrementally), and prune potential duplicates afterwards.
    all_pairs_possible = np.array([ [i, j] for i in range(n_individuals) for j in range(i+1, n_individuals) ])
    # all_pairs_possible = np.array(list(itertools.combinations(range(n_individuals), 2)))
    assert all_pairs_possible.shape == (max_pairs_possible, 2)
    return all_pairs_possible[indices_selected_pairs, :]
