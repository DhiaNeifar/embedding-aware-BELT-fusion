"""Parsing/construction tests for the Phase 1 refine/abstain CLI knobs added
to ``evaluation/belt_fusion.py`` (the Verify-stage gate schedule and the
Abstain-stage agreement rule's tolerance/consensus-floor). These are the
small, additive new-flag plumbing this phase adds on top of the already-
tuned (but previously hardcoded) ``alignformer_pva.refine``/``abstain``
dataclasses -- see those modules' own test files for the math itself.
"""

import pytest

from embedding_aware_belt_fusion.alignformer_pva.refine import (
    AgreementConfig,
    DEFAULT_GATES_M,
    RefineConfig,
)
from embedding_aware_belt_fusion.evaluation.belt_fusion import (
    parse_alignformer_pva_gates_m,
)


def test_parse_alignformer_pva_gates_m_parses_the_tuned_schedule():
    # Chenyi's own tuned schedule (sec/3_method.tex's "Verify" paragraph).
    assert parse_alignformer_pva_gates_m("6,3,1.5,0.75,0.5") == (6.0, 3.0, 1.5, 0.75, 0.5)


def test_parse_alignformer_pva_gates_m_tolerates_surrounding_whitespace():
    assert parse_alignformer_pva_gates_m(" 2.0, 1.0 , 0.5 ") == (2.0, 1.0, 0.5)


def test_parse_alignformer_pva_gates_m_default_string_reproduces_DEFAULT_GATES_M():
    default_spec = ",".join(str(gate) for gate in DEFAULT_GATES_M)
    parsed = parse_alignformer_pva_gates_m(default_spec)
    assert parsed == DEFAULT_GATES_M
    # The CLI default must feed into RefineConfig and reproduce today's
    # config bit for bit -- the additive contract --alignformer-pva-refine-
    # gates-m promises when the flag is omitted.
    assert RefineConfig(gates_m=parsed) == RefineConfig()


@pytest.mark.parametrize("spec", ["", "6,,0.5", "6,abc,0.5", ",", "6, ,0.5"])
def test_parse_alignformer_pva_gates_m_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_alignformer_pva_gates_m(spec)


def test_agreement_config_defaults_match_todays_effective_off_behavior():
    # --alignformer-pva-consensus-floor's default (0) and the dataclass's own
    # default agree, so a caller who never turns on the agreement rule (by
    # leaving --alignformer-pva-agreement-tolerance-m unset, so this type is
    # never constructed at all -- see _load_alignformer_pva) changes nothing.
    assert AgreementConfig(tolerance_m=1.0).consensus_floor == 0
