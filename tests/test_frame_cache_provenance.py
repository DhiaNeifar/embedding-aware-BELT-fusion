from embedding_aware_belt_fusion.evaluation.frame_cache import provenance_matches

_BASE = {
    "cache_version": 1,
    "opencood_config": {"path": "/old/repo/config.yaml", "bytes": 100, "sha256": "abc"},
    "detector_checkpoint": {"path": "/old/repo/net.pth", "bytes": 200, "sha256": "def"},
    "uncertainty_checkpoint": {"path": "/old/repo/unc.pth", "bytes": 300, "sha256": "ghi"},
    "data_root": "/mnt/external/workspace/public/dataset/opv2v",
    "split": "test",
    "noise": {"position_std_m": 0.2, "heading_std_deg": 0.0, "time_delay_ms": 0, "seed": 20},
    "scenario_subset": None,
}


def test_matches_despite_relocated_repo_path():
    # Same checkpoint/config content, different absolute path -- e.g. after
    # the repo was renamed/moved. The cache is still exactly the same data.
    current = {
        **_BASE,
        "opencood_config": {"path": "/new/repo/config.yaml", "bytes": 100, "sha256": "abc"},
        "detector_checkpoint": {"path": "/new/repo/net.pth", "bytes": 200, "sha256": "def"},
        "uncertainty_checkpoint": {"path": "/new/repo/unc.pth", "bytes": 300, "sha256": "ghi"},
    }
    assert provenance_matches(_BASE, current) is True


def test_rejects_genuinely_different_checkpoint_content():
    current = {
        **_BASE,
        "detector_checkpoint": {"path": "/new/repo/net.pth", "bytes": 999, "sha256": "different"},
    }
    assert provenance_matches(_BASE, current) is False


def test_rejects_different_noise_level():
    current = {**_BASE, "noise": {**_BASE["noise"], "position_std_m": 0.6}}
    assert provenance_matches(_BASE, current) is False


def test_rejects_different_split():
    current = {**_BASE, "split": "train"}
    assert provenance_matches(_BASE, current) is False


def test_rejects_missing_saved_manifest():
    assert provenance_matches(None, _BASE) is False
