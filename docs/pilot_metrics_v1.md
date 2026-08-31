# TCP-CI Pilot Metrics v1

## Purpose

This document defines the minimal evaluation metric set used
for the five-project TCP-CI probabilistic pilot.

The metric stage evaluates predictions already produced on the
chronological outer test period.

It performs no model fitting, model selection, probability
calibration, or threshold optimization.

## Evaluation unit

Metrics are computed independently for each project and model
using outer-test observations.

Per-project values are treated as the primary pilot evidence.

Macro summaries assign equal weight to each project.

## Data context

### Failure prevalence

`failure_prevalence` is the observed fraction of failed test
executions in the outer test period.

It is reported because the TCP-CI pilot projects have highly
different and, in several cases, extremely low failure rates.

## Discrimination metrics

### ROC-AUC

ROC-AUC evaluates whether failed observations tend to receive
higher risk scores than non-failed observations.

A constant-score dummy model has ROC-AUC approximately 0.5
when both classes are present.

ROC-AUC evaluates ranking and does not by itself establish
probability calibration.

### Average Precision

Average Precision summarizes the precision-recall ranking
behavior and is particularly useful under strong class
imbalance.

Its natural constant-score reference is the failure
prevalence.

### AP lift

`ap_lift` is defined as:

`Average Precision / failure prevalence`

A dummy-prior constant-score model therefore has AP lift
approximately 1.

This quantity is used only as a descriptive pilot diagnostic
for interpreting Average Precision across projects with
different failure prevalence.

AP lift is interpreted only at the per-project level because
it is normalized by project-specific failure prevalence.
Its macro arithmetic mean is highly sensitive to projects
with extremely rare failures and is therefore not treated as
a primary cross-project summary.

## Probability-quality metrics

### Brier score

The Brier score measures squared error between the predicted
failure probability and the binary observed outcome.

Lower values are better.

Because the Brier score is probability-sensitive, a model may
have strong ranking performance while still having a worse
Brier score than a simple prevalence baseline.

### Log Loss

Log Loss evaluates probabilistic predictions and penalizes
confident incorrect predictions strongly.

Lower values are better.

## Calibration diagnostics

### Expected Calibration Error

Pilot ECE is computed using 10 equal-width probability bins
over `[0, 1]`.

For each non-empty bin, the absolute difference between the
mean predicted probability and observed failure frequency is
weighted by the fraction of observations in that bin.

Lower ECE values indicate smaller bin-level calibration gaps.

ECE is bin-dependent and is therefore treated as one
calibration diagnostic rather than a complete proof of
calibration quality.

### Mean predicted probability

The arithmetic mean of predicted failure probabilities over
the outer test observations.

### Calibration bias

Pilot calibration bias is defined as:

`mean predicted probability - observed failure prevalence`

A positive value indicates average over-estimation of failure
risk, while a negative value indicates average
under-estimation.

A value near zero indicates agreement in the mean only and
does not imply that the model is calibrated across the
probability range.

### Absolute calibration bias

`absolute_calibration_bias` is defined as:

`|mean predicted probability - observed failure prevalence|`

Unlike signed calibration bias, this diagnostic measures the
magnitude of average probability bias without preserving its
direction.

It is useful for cross-project descriptive summaries because
positive and negative signed calibration biases cannot cancel
each other when averaged.

A lower value indicates closer agreement between the mean
predicted probability and observed failure prevalence.

Like signed calibration bias, however, a small absolute bias
only establishes agreement in the mean and does not imply
calibration across the full probability range.

## Calibration-bin artifact

The metric pipeline stores non-empty calibration bins with:

- probability-bin boundaries;
- number of observations;
- bin weight;
- mean predicted probability;
- observed failure rate;
- absolute calibration gap.

These diagnostics can later be used to construct reliability
diagrams without re-running baseline model training.

## Metrics intentionally deferred

Pilot metric v1 does not report threshold-dependent metrics
such as:

- accuracy;
- thresholded precision;
- thresholded recall;
- F1 score.

No probability threshold or test-selection decision rule has
yet been frozen for this stage.

Operational selection metrics such as:

- selected-test ratio;
- execution-cost ratio;
- cost reduction;
- failure retention;
- risk coverage

are also deferred until the risk/cost selector is introduced.

## Relation to probability calibration

The probabilities evaluated at this stage are raw,
uncalibrated baseline probabilities.

Observed Brier, Log Loss, ECE, and calibration-bias values
provide baseline measurements against which later calibrated
probabilities can be compared.

The current stage does not establish that calibration improves
performance.

## Pilot results

The metric pipeline was executed successfully on all five
frozen TCP-CI pilot projects for both the dummy-prior and
logistic-regression baselines.

### Per-project calibration-oriented results

| Project key | Model | Failure prevalence | AP lift | Brier Score | Log Loss | ECE | Calibration Bias | Absolute Calibration Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| datasets/camunda@camunda-bpm-platform | Dummy prior | 0.005100 | 1.000000 | 0.005075 | 0.032091 | 0.000868 | -0.000868 | 0.000868 |
| datasets/camunda@camunda-bpm-platform | Logistic regression | 0.005100 | 8.959551 | 0.004963 | 0.024080 | 0.001120 | -0.000916 | 0.000916 |
| datasets/apache@shardingsphere | Dummy prior | 0.000434 | 1.000000 | 0.000436 | 0.004681 | 0.001544 | 0.001544 | 0.001544 |
| datasets/apache@shardingsphere | Logistic regression | 0.000434 | 294.291653 | 0.000462 | 0.002140 | 0.000269 | 0.000050 | 0.000050 |
| datasets/facebook@buck | Dummy prior | 0.000063 | 1.000000 | 0.000069 | 0.002959 | 0.002519 | 0.002519 | 0.002519 |
| datasets/facebook@buck | Logistic regression | 0.000063 | 16.422879 | 0.000099 | 0.000744 | 0.000080 | 0.000013 | 0.000013 |
| datasets/Angel-ML@angel | Dummy prior | 0.096960 | 1.000000 | 0.091877 | 0.364717 | 0.065714 | -0.065714 | 0.065714 |
| datasets/Angel-ML@angel | Logistic regression | 0.096960 | 9.849862 | 0.023399 | 0.103535 | 0.038954 | -0.037610 | 0.037610 |
| datasets/zolyfarkas@spf4j | Dummy prior | 0.006308 | 1.000000 | 0.006269 | 0.038280 | 0.000644 | -0.000644 | 0.000644 |
| datasets/zolyfarkas@spf4j | Logistic regression | 0.006308 | 90.757677 | 0.003691 | 0.020820 | 0.002938 | 0.001268 | 0.001268 |


### Macro mean across projects

| Model | ROC-AUC | Average Precision | Brier Score | Log Loss | ECE | Absolute Calibration Bias |
|---|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500000 | 0.021773 | 0.020745 | 0.088546 | 0.014258 | 0.014258 |
| Logistic regression | 0.912658 | 0.340399 | 0.006523 | 0.030264 | 0.008672 | 0.007971 |


The logistic baseline retained substantially stronger
discrimination than the dummy-prior baseline across all five
pilot projects.

For calibration-oriented performance, the pattern was
project-dependent. Logistic regression produced lower ECE
than the dummy-prior baseline for
`datasets/apache@shardingsphere`,
`datasets/facebook@buck`, and
`datasets/Angel-ML@angel`, but slightly higher ECE for
`datasets/camunda@camunda-bpm-platform` and
`datasets/zolyfarkas@spf4j`.

The results therefore do not establish that the raw logistic
probabilities are consistently calibrated across projects.
They provide the uncalibrated reference measurements for the
later explicit probability-calibration experiments.

AP lift is interpreted primarily at the per-project level.
Because it divides Average Precision by project-specific
failure prevalence, its arithmetic macro mean can be strongly
influenced by projects with extremely rare failures.

Signed calibration bias is also interpreted primarily at the
per-project level because positive and negative values may
cancel when averaged. Absolute calibration bias is therefore
reported as an additional magnitude-only diagnostic for
cross-project summaries.

Neither a low ECE nor a low absolute calibration bias alone
is treated as proof of calibration quality.


## Outputs

Generated metric artifacts are stored under:

`results/pilot/metrics/`

The main files are:

- `pilot_metrics.csv`
- `pilot_metrics_macro.csv`
- `calibration_bins.csv`

## Scope

These metrics support validation and characterization of the
five-project pilot pipeline.

They are not the final RQ1 statistical analysis and should not
be interpreted as the final cross-project calibration results
of the thesis.