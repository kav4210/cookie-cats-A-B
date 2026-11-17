"""A/B test analysis for the Cookie Cats gate experiment.

Question: does moving the first progression gate from level 30 (control) to level 40
(treatment) change player retention? Metrics: 1-day retention, 7-day retention, and
game rounds played (guardrail / engagement).

Run:  python ab_analysis.py data/cookie_cats.csv
Outputs: REPORT.md, retention_ci.png, posterior_retention_7.png
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

CONTROL, TREATMENT = "gate_30", "gate_40"
REQUIRED = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
ALPHA = 0.05
RNG = np.random.default_rng(42)


# ---------- loading and validation ----------
def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for c in ("retention_1", "retention_7"):
        df[c] = df[c].astype(bool)
    return df


def data_quality(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "duplicate_userids": int(df["userid"].duplicated().sum()),
        "unknown_versions": sorted(set(df["version"]) - {CONTROL, TREATMENT}),
        "null_cells": int(df[REQUIRED].isna().sum().sum()),
        "max_gamerounds": int(df["sum_gamerounds"].max()),
        "zero_round_players": int((df["sum_gamerounds"] == 0).sum()),
    }


def sample_ratio_mismatch(n_control: int, n_treat: int) -> dict:
    """Chi-square goodness of fit against an intended 50/50 split. p < 0.001 means the
    assignment itself is suspect and results should not be trusted."""
    total = n_control + n_treat
    chi2, p = stats.chisquare([n_control, n_treat], f_exp=[total / 2, total / 2])
    return {"n_control": n_control, "n_treatment": n_treat, "chi2": chi2, "p": p, "srm_flag": bool(p < 0.001)}


# ---------- frequentist tests ----------
def two_proportion_test(x_c: int, n_c: int, x_t: int, n_t: int) -> dict:
    """Pooled two-proportion z-test plus a 95% CI on the difference (treatment - control)."""
    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    lo, hi = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="wald", compare="diff")
    return {"p_control": p_c, "p_treatment": p_t, "diff": p_t - p_c, "ci_low": lo, "ci_high": hi,
            "z": z, "p_value": p_value, "rel_lift": (p_t - p_c) / p_c if p_c else float("nan")}


def power_analysis(baseline: float, n_per_group: int, alpha: float = ALPHA, power: float = 0.8) -> dict:
    """Minimum detectable absolute lift at this n, and n needed per group for a 1 pp lift."""
    solver = NormalIndPower()
    mde_h = solver.solve_power(nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0)
    # convert Cohen's h back to an absolute proportion difference
    phi = 2 * math.asin(math.sqrt(baseline)) + mde_h
    mde_abs = math.sin(phi / 2) ** 2 - baseline
    h_1pp = proportion_effectsize(baseline + 0.01, baseline)
    n_for_1pp = solver.solve_power(effect_size=h_1pp, alpha=alpha, power=power, ratio=1.0)
    return {"mde_abs": mde_abs, "n_for_1pp_per_group": math.ceil(n_for_1pp)}


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, reps: int = 5000) -> dict:
    """Bootstrap CI on the difference in mean game rounds (heavy-tailed, so no t-test)."""
    diffs = np.empty(reps)
    for i in range(reps):
        diffs[i] = RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean()
    return {"diff": b.mean() - a.mean(), "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5)}



# ---------- regression ----------
def regression_models(df: pd.DataFrame) -> dict:
    """Logistic regression for each retention outcome and OLS on log(1 + rounds), each on the
    treatment indicator alone. The file has no pre-experiment covariates, and game rounds are
    measured after assignment, so adjusting for rounds would bias the treatment effect
    (post-treatment adjustment). With one binary predictor the logistic odds ratio equals the
    2x2 cross-product ratio, so these models double as a consistency check on the z-tests."""
    X = sm.add_constant((df["version"] == TREATMENT).astype(float).rename("gate_40"))
    out = {}
    for metric in ("retention_1", "retention_7"):
        fit = sm.Logit(df[metric].astype(float), X).fit(disp=0)
        lo, hi = np.exp(fit.conf_int().loc["gate_40"])
        out[metric] = {"odds_ratio": float(np.exp(fit.params["gate_40"])), "or_low": float(lo),
                       "or_high": float(hi), "p": float(fit.pvalues["gate_40"])}
    y = np.log1p(df["sum_gamerounds"].astype(float))
    ols = sm.OLS(y, X).fit(cov_type="HC3")
    lo, hi = ols.conf_int().loc["gate_40"]
    out["log_rounds"] = {"pct_change": float(np.expm1(ols.params["gate_40"]) * 100),
                         "low": float(np.expm1(lo) * 100), "high": float(np.expm1(hi) * 100),
                         "p": float(ols.pvalues["gate_40"])}
    return out

# ---------- Bayesian ----------
def bayesian_beta_binomial(x_c: int, n_c: int, x_t: int, n_t: int, draws: int = 200_000) -> dict:
    """Beta(1,1) priors. Returns P(treatment > control), expected loss of shipping treatment
    if it is actually worse, and a 95% credible interval on the difference."""
    pc = RNG.beta(1 + x_c, 1 + n_c - x_c, draws)
    pt = RNG.beta(1 + x_t, 1 + n_t - x_t, draws)
    diff = pt - pc
    return {
        "prob_treatment_better": float((diff > 0).mean()),
        "expected_loss_ship_treatment": float(np.maximum(-diff, 0).mean()),
        "cred_low": float(np.percentile(diff, 2.5)),
        "cred_high": float(np.percentile(diff, 97.5)),
        "_pc": pc, "_pt": pt,
    }


# ---------- orchestration ----------
def analyze(df: pd.DataFrame) -> dict:
    c = df[df.version == CONTROL]
    t = df[df.version == TREATMENT]
    res = {"quality": data_quality(df), "srm": sample_ratio_mismatch(len(c), len(t))}
    for metric in ("retention_1", "retention_7"):
        xc, xt = int(c[metric].sum()), int(t[metric].sum())
        res[metric] = {
            "freq": two_proportion_test(xc, len(c), xt, len(t)),
            "bayes": bayesian_beta_binomial(xc, len(c), xt, len(t)),
            "power": power_analysis(xc / len(c), min(len(c), len(t))),
        }
    # Holm correction across the two retention tests (same experiment, two primary metrics)
    raw = [res[m]["freq"]["p_value"] for m in ("retention_1", "retention_7")]
    adj = multipletests(raw, alpha=ALPHA, method="holm")[1]
    res["retention_1"]["freq"]["p_holm"], res["retention_7"]["freq"]["p_holm"] = adj

    res["regression"] = regression_models(df)
    res["aa"] = aa_simulation(df)
    ca, ta = c["sum_gamerounds"].to_numpy(float), t["sum_gamerounds"].to_numpy(float)
    u, p_u = stats.mannwhitneyu(ca, ta, alternative="two-sided")
    res["rounds"] = {
        "median_control": float(np.median(ca)), "median_treatment": float(np.median(ta)),
        "mean_control": float(ca.mean()), "mean_treatment": float(ta.mean()),
        "mannwhitney_p": float(p_u),
        "bootstrap_mean_diff": bootstrap_mean_diff(ca, ta),
        "mean_diff_excluding_top_0.1pct": float(
            ta[ta <= np.percentile(df.sum_gamerounds, 99.9)].mean()
            - ca[ca <= np.percentile(df.sum_gamerounds, 99.9)].mean()
        ),
    }
    return res


def aa_simulation(df: pd.DataFrame, reps: int = 2000, seed: int = 0) -> dict:
    """A/A check: shuffle players into two fake arms of the real group sizes and test 7-day
    retention. A calibrated pipeline rejects about 5% of the time and gives uniform p-values."""
    rng = np.random.default_rng(seed)
    y = df["retention_7"].to_numpy(bool)
    n, n_c = len(y), int((df.version == CONTROL).sum())
    n_t = n - n_c
    pvals = np.empty(reps)
    for i in range(reps):
        idx = rng.permutation(n)
        xc, xt = y[idx[:n_c]].sum(), y[idx[n_c:]].sum()
        pooled = (xc + xt) / n
        se = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
        z = (xt / n_t - xc / n_c) / se
        pvals[i] = 2 * stats.norm.sf(abs(z))
    fp = float((pvals < ALPHA).mean())
    half = 1.96 * math.sqrt(ALPHA * (1 - ALPHA) / reps)
    return {"reps": reps, "false_positive_rate": fp, "expected": ALPHA,
            "within_binomial_band": bool(abs(fp - ALPHA) <= half),
            "ks_p_uniform": float(stats.kstest(pvals, "uniform").pvalue)}


def recommendation(res: dict) -> str:
    r7 = res["retention_7"]
    sig = r7["freq"]["p_holm"] < ALPHA
    worse = r7["freq"]["diff"] < 0
    if res["srm"]["srm_flag"]:
        return "Do not decide yet. The group sizes fail the sample ratio check, so the assignment may be broken."
    if sig and worse:
        return ("Keep the gate at level 30. Moving it to 40 lowered 7-day retention by a statistically "
                "reliable amount.")
    if sig and not worse:
        return "Ship gate_40. It raised 7-day retention by a statistically reliable amount."
    return ("No reliable difference at the 5% level after correction. The data cannot support shipping gate_40 "
            "for a retention gain. Treat it as unproven, and see the power section for how large a change this "
            "test could actually detect.")


def plots(res: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, m in enumerate(("retention_1", "retention_7")):
        f = res[m]["freq"]
        ax.errorbar(f["diff"] * 100, i, xerr=[[(f["diff"] - f["ci_low"]) * 100], [(f["ci_high"] - f["diff"]) * 100]],
                    fmt="o", capsize=5, color="#4c78a8")
    ax.axvline(0, color="grey", ls="--")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["1-day retention", "7-day retention"])
    ax.set_xlabel("gate_40 minus gate_30 (percentage points, 95% CI)")
    ax.set_title("Effect of moving the gate to level 40")
    fig.tight_layout(); fig.savefig(outdir / "retention_ci.png", dpi=200); plt.close(fig)

    b = res["retention_7"]["bayes"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist((b["_pt"] - b["_pc"]) * 100, bins=80, color="#e45756", alpha=0.8)
    ax.axvline(0, color="black", ls="--")
    ax.set_xlabel("posterior of 7-day retention difference (percentage points)")
    ax.set_title(f"P(gate_40 better) = {b['prob_treatment_better']:.1%}")
    fig.tight_layout(); fig.savefig(outdir / "posterior_retention_7.png", dpi=200); plt.close(fig)


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def report(res: dict) -> str:
    q, s = res["quality"], res["srm"]
    r1, r7, rr = res["retention_1"], res["retention_7"], res["rounds"]
    f1, f7, b7 = r1["freq"], r7["freq"], r7["bayes"]
    aa = res["aa"]
    g1, g7, lr = res["regression"]["retention_1"], res["regression"]["retention_7"], res["regression"]["log_rounds"]
    return f"""# Cookie Cats gate experiment: results

**Recommendation:** {recommendation(res)}

## Design
Players were split between a control with the first gate at level 30 (`gate_30`) and a treatment with it at level 40 (`gate_40`). Primary metrics: 1-day and 7-day retention. Guardrail: game rounds played. Significance level 5%, with Holm correction across the two retention tests.

## Data checks
- Rows: {q['rows']:,}; duplicate user ids: {q['duplicate_userids']}; unexpected versions: {q['unknown_versions'] or 'none'}; null cells: {q['null_cells']}
- Group sizes: {s['n_control']:,} vs {s['n_treatment']:,}. Sample ratio check (chi-square vs 50/50): p = {s['p']:.3f} ({'FAILED' if s['srm_flag'] else 'BORDERLINE (p < 0.05): group sizes are slightly off 50/50, so treat results with some caution' if s['p'] < 0.05 else 'passed'})
- Game rounds are heavy-tailed (most rounds by one player: {q['max_gamerounds']:,}; {q['zero_round_players']:,} players never played), so rounds are compared with medians, a rank test and a bootstrap, not just means.

## Retention
| Metric | gate_30 | gate_40 | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|---|
| 1-day | {pct(f1['p_control'])} | {pct(f1['p_treatment'])} | {f1['diff']*100:+.2f} pp ({f1['ci_low']*100:+.2f} to {f1['ci_high']*100:+.2f}) | {f1['p_value']:.3f} | {f1['p_holm']:.3f} |
| 7-day | {pct(f7['p_control'])} | {pct(f7['p_treatment'])} | {f7['diff']*100:+.2f} pp ({f7['ci_low']*100:+.2f} to {f7['ci_high']*100:+.2f}) | {f7['p_value']:.3f} | {f7['p_holm']:.3f} |

## Bayesian view (7-day retention, flat priors)
- Probability gate_40 is better than gate_30: **{b7['prob_treatment_better']:.1%}**
- 95% credible interval on the difference: {b7['cred_low']*100:+.2f} to {b7['cred_high']*100:+.2f} pp
- Expected loss from shipping gate_40: {b7['expected_loss_ship_treatment']*100:.3f} pp of retention

## Game rounds (guardrail)
Median rounds: {rr['median_control']:.0f} vs {rr['median_treatment']:.0f}. Mean: {rr['mean_control']:.1f} vs {rr['mean_treatment']:.1f}. Mann-Whitney p = {rr['mannwhitney_p']:.3f}. Bootstrap 95% CI on the mean difference: {rr['bootstrap_mean_diff']['ci_low']:+.2f} to {rr['bootstrap_mean_diff']['ci_high']:+.2f}. Mean difference with the top 0.1% of players removed: {rr['mean_diff_excluding_top_0.1pct']:+.2f}.

## Regression
Logistic regression of each retention outcome on the treatment indicator (odds ratio for gate_40 vs gate_30), and OLS with robust (HC3) errors on log(1 + game rounds).
- 1-day retention: odds ratio {g1['odds_ratio']:.3f} (95% CI {g1['or_low']:.3f} to {g1['or_high']:.3f}), p = {g1['p']:.3f}
- 7-day retention: odds ratio {g7['odds_ratio']:.3f} (95% CI {g7['or_low']:.3f} to {g7['or_high']:.3f}), p = {g7['p']:.4f}
- Game rounds: {lr['pct_change']:+.1f}% (95% CI {lr['low']:+.1f}% to {lr['high']:+.1f}%), p = {lr['p']:.3f}
- No covariates are added on purpose. The data has no pre-experiment player attributes, and adjusting for game rounds would be post-treatment adjustment and would bias the effect. With one binary predictor the odds ratio equals the 2x2 table's cross-product ratio, so the regression agrees with the z-tests by construction.

## A/A simulation (pipeline check)
Before trusting any A/B result, the test pipeline should not find "effects" where none exist. Players were randomly shuffled into two fake arms with the real group sizes, and 7-day retention was tested {aa['reps']:,} times. False positive rate: **{aa['false_positive_rate']:.1%}** (target 5.0%; {'inside' if aa['within_binomial_band'] else 'OUTSIDE'} the sampling band). Uniformity of the p-values (Kolmogorov-Smirnov) p = {aa['ks_p_uniform']:.2f}. {'The test is well calibrated, so the real gate_30 vs gate_40 result is not an artifact of the method.' if aa['within_binomial_band'] else 'The test is miscalibrated; do not trust the results above until this is understood.'}

## Power
With {min(s['n_control'], s['n_treatment']):,} players per group, 80% power and 5% significance, this test could reliably detect a 7-day retention change of about **{r7['power']['mde_abs']*100:.2f} pp**. Detecting a 1 pp change needs about **{r7['power']['n_for_1pp_per_group']:,}** players per group. A non-significant result means the true effect is probably smaller than that, not that it is zero.

## Limits
- The data has no record of how players were assigned. The analysis assumes random assignment, which the sample ratio check can only partly test.
- Retention is binary and measured at two fixed days, so it says nothing about longer-term behavior or revenue.
- Two primary metrics were tested; Holm correction is applied, and the rounds guardrail is descriptive.

![CI](retention_ci.png)
![Posterior](posterior_retention_7.png)
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="data/cookie_cats.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    df = load(a.csv)
    res = analyze(df)
    out = Path(a.out)
    plots(res, out)
    (out / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(recommendation(res))
    print(f"Wrote {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
