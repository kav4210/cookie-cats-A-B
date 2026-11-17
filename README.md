# Cookie Cats A/B Test Analysis

An end-to-end analysis of a mobile-game experiment: does moving the first progression gate from level 30 to level 40 change player retention? The script validates the data, checks the experiment itself, runs frequentist and Bayesian tests, and writes a plain-English readout with a recommendation.

**Results: see [REPORT.md](REPORT.md)** (generated from the real data by the script below).

## Data
The public Cookie Cats dataset (about 90,000 players, columns `userid, version, sum_gamerounds, retention_1, retention_7`). Download it from Kaggle (search "Mobile Games A/B Testing - Cookie Cats") and save it as `data/cookie_cats.csv`. The CSV is not committed.

## Run it
```bash
pip install -r requirements.txt
python ab_analysis.py data/cookie_cats.csv   # writes REPORT.md and two PNG charts
pytest                                       # 11 tests
```

## What it does
1. **Data checks:** duplicate ids, unexpected versions, nulls, extreme values.
2. **Sample ratio mismatch test:** chi-square against the intended 50/50 split, to catch a broken assignment before trusting any result.
3. **Retention tests:** two-proportion z-test with a 95% confidence interval on the difference, Holm correction across the two primary metrics.
4. **Bayesian view:** Beta-Binomial posterior giving the probability the treatment is better, a credible interval, and the expected loss of shipping it.
5. **Guardrail (game rounds):** medians, Mann-Whitney test, and a bootstrap CI, because rounds are heavy-tailed and a t-test on means would be misleading.
6. **Regression:** logistic regression (odds ratios with CIs) for both retention metrics and robust OLS on log game rounds, deliberately with no covariates, because the data has no pre-experiment attributes and game rounds are post-treatment.
7. **Power analysis:** the smallest retention change this test could reliably detect, and the sample size needed for a 1 percentage point change.
8. **Recommendation:** a decision rule that refuses to call a winner if the sample ratio check fails.

## Tests
The tests check the z-test against statsmodels, the power calculation against the textbook formula, that a planted effect is detected, and that a null experiment is not called significant.

## Limits
Two fixed retention days only, no revenue data, and assignment is assumed random.
