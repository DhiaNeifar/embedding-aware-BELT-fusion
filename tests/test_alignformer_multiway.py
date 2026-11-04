"""alignformer_ext.multiway: the multi-marginal Sinkhorn reduces to the pairwise plan and uses third-agent evidence."""

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import multiway
from embedding_aware_belt_fusion.alignformer_pva.head_match import log_sinkhorn


def _pairwise(scores, alpha=1.0):
    return log_sinkhorn(torch.as_tensor(scores, dtype=torch.float32)[None], torch.tensor(alpha), 100)[0].numpy()


def test_two_agents_reproduce_the_pairwise_plan():
    rng = np.random.default_rng(0)
    plan = _pairwise(rng.normal(0, 2, (4, 6)))
    joint = multiway.multimarginal_sinkhorn({(0, 1): plan}, [4, 6], iterations=200)
    np.testing.assert_allclose(np.exp(joint), np.exp(plan), atol=1e-4)


def test_marginals_and_pairwise_consistency():
    rng = np.random.default_rng(1)
    sizes = [3, 4, 2]
    pots = {(0, 1): _pairwise(rng.normal(0, 2, (3, 4))), (0, 2): _pairwise(rng.normal(0, 2, (3, 2))),
            (1, 2): _pairwise(rng.normal(0, 2, (4, 2)))}
    joint = np.exp(multiway.multimarginal_sinkhorn(pots, sizes, iterations=500, tolerance=1e-8))
    total = sum(sizes)
    for k, n in enumerate(sizes):
        marginal = joint.sum(axis=tuple(a for a in range(3) if a != k))
        np.testing.assert_allclose(marginal[:n], 1.0, atol=1e-5)
        np.testing.assert_allclose(marginal[n], total - n, atol=1e-4)


def test_sparse_solver_matches_the_dense_one():
    rng = np.random.default_rng(2)
    sizes = [4, 3, 5, 2]
    pots = {}
    for k in range(4):
        for l in range(k + 1, 4):
            if (k, l) != (1, 3):                       # one missing CAV<->CAV link
                pots[(k, l)] = _pairwise(rng.normal(0, 3, (sizes[k], sizes[l])))
    dense = multiway.multimarginal_sinkhorn(pots, sizes, iterations=400, tolerance=1e-9)
    tuples, log_mass = multiway.sparse_multimarginal_sinkhorn(pots, sizes, threshold=0.0, iterations=400, tolerance=1e-9)
    assert len(tuples) == np.prod([n + 1 for n in sizes])
    for k in range(1, 4):
        np.testing.assert_allclose(np.exp(multiway.sparse_pairwise_marginal(tuples, log_mass, sizes, 0, k)),
                                   np.exp(multiway.pairwise_marginal(dense, 0, k)), atol=1e-6)
    tuples, log_mass = multiway.sparse_multimarginal_sinkhorn(pots, sizes, threshold=1e-3, iterations=400)
    assert len(tuples) < np.prod([n + 1 for n in sizes])
    np.testing.assert_allclose(np.exp(multiway.sparse_pairwise_marginal(tuples, log_mass, sizes, 0, 1)),
                               np.exp(multiway.pairwise_marginal(dense, 0, 1)), atol=0.05)


def test_a_third_agent_resolves_an_ambiguous_pair():
    # ego box 0 is equally similar to A's boxes 0 and 1; B sees the object and links ego 0 <-> B 0 <-> A 1
    s_ea = np.array([[2.0, 2.0], [-4.0, -4.0]])
    s_eb = np.array([[4.0, -4.0], [-4.0, -4.0]])
    s_ab = np.array([[-4.0, -4.0], [4.0, -4.0]])
    pots = {(0, 1): _pairwise(s_ea), (0, 2): _pairwise(s_eb), (1, 2): _pairwise(s_ab)}
    pairwise = np.exp(pots[(0, 1)])[:2, :2]
    joint = np.exp(multiway.pairwise_marginal(multiway.multimarginal_sinkhorn(pots, [2, 2, 2], iterations=300), 0, 1))[:2, :2]
    assert abs(pairwise[0, 0] - pairwise[0, 1]) < 0.05
    assert joint[0, 1] > pairwise[0, 1] + 0.15 and joint[0, 1] > 3 * joint[0, 0]
