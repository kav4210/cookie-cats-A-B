import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportions_ztest

import ab_analysis as ab


def synth(n=40000, p1c=0.45, p1t=0.45, p7c=0.19, p7t=0.19, seed=1):
    rng = np.random.default_rng(seed)
    half = n // 2
    rows = []
    for ver, p1, p7 in ((ab.CONTROL, p1c, p7c), (ab.TREATMENT, p1t, p7t)):
        rows.append(pd.DataFrame({
            "userid": np.arange(len(rows) * half, len(rows) * half + half),
            "version": ver,
            "sum_gamerounds": rng.geometric(0.02, half),
            "retention_1": rng.random(half) < p1,
            "retention_7": rng.random(half) < p7,
        }))
    return pd.concat(rows, ignore_index=True)


def test_two_prop_matches_statsmodels():
    r = ab.two_proportion_test(8_500, 44_700, 8_300, 45_489)
    z, p = proportions_ztest([8_300, 8_500], [45_489, 44_700])
    assert r["z"] == pytest.approx(z, rel=1e-6)
    assert r["p_value"] == pytest.approx(p, rel=1e-6)
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_srm_flags_bad_split_and_passes_good_one():
    assert ab.sample_ratio_mismatch(50_000, 50_100)["srm_flag"] is False
    assert ab.sample_ratio_mismatch(60_000, 40_000)["srm_flag"] is True


def test_bayes_symmetric_and_decisive_cases():
    same = ab.bayesian_beta_binomial(1900, 10_000, 1900, 10_000)
    assert same["prob_treatment_better"] == pytest.approx(0.5, abs=0.02)
    clear = ab.bayesian_beta_binomial(1500, 10_000, 2000, 10_000)
    assert clear["prob_treatment_better"] > 0.999
    assert clear["expected_loss_ship_treatment"] < 1e-4


def test_power_n_for_1pp_close_to_textbook_formula():
    got = ab.power_analysis(0.19, 45_000)["n_for_1pp_per_group"]
    textbook = (1.959964 + 0.841621) ** 2 * (0.19 * 0.81 + 0.20 * 0.80) / 0.01 ** 2
    assert got == pytest.approx(textbook, rel=0.03)


def test_mde_shrinks_as_sample_grows():
    assert ab.power_analysis(0.19, 90_000)["mde_abs"] < ab.power_analysis(0.19, 10_000)["mde_abs"]


def test_planted_effect_is_detected():
    res = ab.analyze(synth(n=90_000, p7c=0.19, p7t=0.175))
    assert res["retention_7"]["freq"]["p_holm"] < 0.05
    assert res["retention_7"]["freq"]["diff"] < 0
    assert "Keep the gate at level 30" in ab.recommendation(res)


def test_null_effect_is_not_called_significant():
    res = ab.analyze(synth(n=90_000, seed=7))
    assert "No reliable difference" in ab.recommendation(res)


def test_report_and_plots_build(tmp_path):
    res = ab.analyze(synth())
    ab.plots(res, tmp_path)
    assert (tmp_path / "retention_ci.png").exists()
    assert "Recommendation" in ab.report(res)


def test_missing_column_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("userid,version\n1,gate_30\n")
    with pytest.raises(ValueError, match="missing columns"):
        ab.load(p)
