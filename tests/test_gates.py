"""Gate-integrity tests.

These exist because the project's core rule is that a gate must never quietly get
easier. Each test encodes one way that could happen.

Run:  python tests/test_gates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc2mesh.common import load_config  # noqa: E402
from pc2mesh.eval_gates import GATE_SPEC, PER_SHAPE_FIELDS, aggregate, verdict  # noqa: E402

CFG = load_config()
GATES = dict(CFG.eval.gates)


def _row(**kw):
    r = {k: "" for k in PER_SHAPE_FIELDS}
    r.update(kw)
    return r


def _good():
    return _row(watertight=1, winding_consistent=1, self_intersection_free=1,
                iou128=0.95, chamfer_l2_x1e3=0.2, normal_consistency=0.95,
                euler_match=1)


def test_verdict_bands():
    g = {"pass": 0.98, "partial": 0.90, "direction": "higher"}
    assert verdict("x", 0.99, g) == "PASS"
    assert verdict("x", 0.95, g) == "PARTIAL"
    assert verdict("x", 0.80, g) == "FAIL"
    lo = {"pass": 1.0, "partial": 2.0, "direction": "lower"}
    assert verdict("x", 0.5, lo) == "PASS"
    assert verdict("x", 1.5, lo) == "PARTIAL"
    assert verdict("x", 3.0, lo) == "FAIL"
    # a missing value is a FAIL, never a free pass
    assert verdict("x", float("nan"), g) == "FAIL"
    print("  ok  verdict bands, and NaN is FAIL")


def test_unscorable_shape_counts_as_failure():
    """The regression this guards: an errored shape must not leave the denominator.

    A crashed shape used to keep every field blank, and aggregate() skipped blanks
    — so it vanished from the mean instead of pulling it down.
    """
    rows = [_good() for _ in range(9)]
    crashed = _row(error="RuntimeError: boom", watertight=0, winding_consistent=0,
                   self_intersection_free=0, iou128=0.0, normal_consistency=0.0,
                   euler_match=0, chamfer_l2_x1e3=float("nan"))
    agg = aggregate(rows + [crashed], GATES)
    assert abs(agg["watertight_rate"]["value"] - 0.9) < 1e-9, agg["watertight_rate"]
    assert agg["watertight_rate"]["n"] == 10, "errored shape left the denominator"
    assert abs(agg["iou128_mean"]["value"] - 0.855) < 1e-9, agg["iou128_mean"]
    # chamfer has no meaningful value for a crash, so it is averaged over 9 and says so
    assert agg["chamfer_l2_x1e3_mean"]["n"] == 9
    assert agg["chamfer_l2_x1e3_mean"]["n_missing"] == 1
    print("  ok  unscorable shape counts as a failure in every rate gate")


def test_blank_row_would_be_caught():
    """Negative control: a fully blank row DOES shrink the denominator.

    This is what the fix prevents; if this assertion ever fails, blanks have
    stopped being distinguishable and the test above proves nothing.
    """
    rows = [_good() for _ in range(9)] + [_row(error="X")]
    agg = aggregate(rows, GATES)
    assert agg["watertight_rate"]["n"] == 9, "blank row no longer shrinks n"
    assert agg["watertight_rate"]["value"] == 1.0, "blank row no longer inflates the gate"
    assert agg["watertight_rate"]["n_missing"] == 1, "n_missing does not report the gap"
    print("  ok  negative control: blank rows shrink n and are flagged by n_missing")


def test_gate_keys_match_config():
    assert set(GATE_SPEC) == set(GATES), (
        f"GATE_SPEC and config.eval.gates disagree: "
        f"{set(GATE_SPEC) ^ set(GATES)}")
    assert GATE_SPEC["iou128_mean"][0] == f"iou{int(CFG.eval.iou_resolution)}", (
        "the gate key names a resolution the config does not score at")
    print("  ok  every registered gate exists in config and the IoU key matches "
          "eval.iou_resolution")


def test_thresholds_are_the_preregistered_ones():
    """The published thresholds, hard-coded here so a silent edit fails a test."""
    expected = {
        "watertight_rate": (0.98, 0.90, "higher"),
        "winding_consistent_rate": (0.98, 0.90, "higher"),
        "self_intersection_free_rate": (1.00, 0.90, "higher"),
        "iou128_mean": (0.90, 0.80, "higher"),
        "chamfer_l2_x1e3_mean": (1.00, 2.00, "lower"),
        "normal_consistency_mean": (0.90, 0.80, "higher"),
        "euler_match_rate": (0.85, 0.70, "higher"),
    }
    for k, (p, pa, d) in expected.items():
        g = GATES[k]
        assert float(g["pass"]) == p, f"{k} pass threshold changed: {g['pass']} != {p}"
        assert float(g["partial"]) == pa, f"{k} partial band changed: {g['partial']} != {pa}"
        assert str(g["direction"]) == d, f"{k} direction changed"
    assert int(CFG.eval.iou_resolution) == 128
    assert int(CFG.meshify.resolution) == 128, (
        "meshify.resolution must stay 128; the R=256 ablation uses a CLI override")
    print("  ok  all 7 pre-registered thresholds unchanged; resolutions still 128")


if __name__ == "__main__":
    fails = 0
    for fn in (test_verdict_bands, test_unscorable_shape_counts_as_failure,
               test_blank_row_would_be_caught, test_gate_keys_match_config,
               test_thresholds_are_the_preregistered_ones):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print("\nGATE TESTS:", "ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
