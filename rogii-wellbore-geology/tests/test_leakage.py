"""Leakage mutants for the trackers: suffix TVT must never influence a prediction; suffix GR must."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wells import eligible_ids, load_wells  # noqa: E402
import tracker as lt  # noqa: E402

WELLS = load_wells("data/stage")
IDS = eligible_ids(WELLS)[:3]
PRIOR = lt.fit_prior([WELLS[i] for i in eligible_ids(WELLS)[10:60]], 10)
P = lt.transition(PRIOR)


def _pred(w, variant):
    logE, bl = lt.variant_emission(w, variant, 10, 1.0)
    return lt.predict_well(w, logE, bl, PRIOR, P)["pred"]


def test_suffix_truth_mutant_has_no_effect():
    for variant in ("v1", "v2"):
        for wid in IDS:
            w = WELLS[wid]; m = copy.deepcopy(w)
            m["tvt"][m["p"]:] += np.random.default_rng(0).normal(0, 50, m["n"] - m["p"])
            assert np.array_equal(_pred(w, variant), _pred(m, variant))


def test_suffix_gr_mutant_changes_prediction():
    w = WELLS[IDS[0]]; m = copy.deepcopy(w)
    m["gr"][m["p"]:] = m["gr"][m["p"]:][::-1]
    assert not np.allclose(_pred(w, "v1"), _pred(m, "v1"))


def test_prefix_anchor_is_last_known_tvt():
    w = WELLS[IDS[0]]
    assert abs(_pred(w, "v1")[0] - w["tvt_input"][w["p"] - 1]) < 2.0
