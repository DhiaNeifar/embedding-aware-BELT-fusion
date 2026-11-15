import torch

from embedding_aware_belt_fusion.alignformer_pva.diagnostics import (
    assignment_diagnostics,
    centre_barycentric_costs,
    marginalized_information,
)


def test_centre_barycentric_identity_holds_for_arbitrary_pose():
    torch.manual_seed(7)
    ego = torch.randn(3, 4, 2, dtype=torch.float64)
    cav = torch.randn(3, 5, 2, dtype=torch.float64)
    weights = torch.rand(3, 4, 5, dtype=torch.float64)
    weights[0, 2] = 0.0
    psi = torch.tensor([0.3, -1.2, 2.4], dtype=torch.float64)
    translation = torch.randn(3, 2, dtype=torch.float64)
    expanded, virtual, dispersion = centre_barycentric_costs(ego, cav, weights, psi, translation)
    torch.testing.assert_close(expanded, virtual + dispersion, rtol=1e-12, atol=1e-12)
    other_expanded, other_virtual, other_dispersion = centre_barycentric_costs(
        ego, cav, weights, psi + 0.7, translation - 3.0
    )
    torch.testing.assert_close(dispersion, other_dispersion, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        expanded - virtual, other_expanded - other_virtual, rtol=1e-12, atol=1e-12
    )


def test_assignment_diagnostics_separate_mass_from_conditional_ambiguity():
    centres = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])
    weights = torch.tensor([[[0.5, 0.5], [0.2, 0.0], [0.0, 0.0]]])
    result = assignment_diagnostics(weights, centres)
    assert result.effective_rows.item() == 2
    assert torch.isclose(result.matched_mass, torch.tensor([1.2])).all()
    assert torch.isclose(result.mean_row_confidence, torch.tensor([0.75])).all()
    assert torch.isclose(result.mean_top2_margin, torch.tensor([0.5])).all()
    assert result.mean_row_entropy.item() > 0.0
    assert torch.isclose(result.mean_spatial_dispersion, torch.tensor([12.5])).all()


def test_missing_information_is_zero_for_identical_score_directions():
    probability = torch.tensor([[0.4, 0.6]])
    gradients = torch.tensor([[[1.0, -2.0], [1.0, -2.0]]])
    hessians = torch.eye(2).reshape(1, 1, 2, 2).repeat(1, 2, 1, 1) * 3.0
    observed, complete, missing = marginalized_information(probability, gradients, hessians)
    torch.testing.assert_close(missing, torch.zeros_like(missing))
    torch.testing.assert_close(observed, complete)


def test_competing_correspondences_remove_information_in_disagreement_axis():
    probability = torch.tensor([[0.5, 0.5]])
    gradients = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    hessians = torch.eye(2).reshape(1, 1, 2, 2).repeat(1, 2, 1, 1) * 2.0
    observed, complete, missing = marginalized_information(probability, gradients, hessians)
    torch.testing.assert_close(complete, torch.diag_embed(torch.tensor([[2.0, 2.0]])))
    torch.testing.assert_close(missing, torch.diag_embed(torch.tensor([[1.0, 0.0]])))
    torch.testing.assert_close(observed, torch.diag_embed(torch.tensor([[1.0, 2.0]])))
