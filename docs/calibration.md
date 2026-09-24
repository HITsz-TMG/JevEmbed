# Scoring and calibration

Choice and Score use independent temperatures, both defaulting to 0.1. Each Noul path has independent slope and intercept parameters, defaulting to 10 and 0. CLM sets temperature 0.01 and Noul slope 100 to match its checkpoint. The Noul slope is the inverse-temperature equivalent in its logistic mapping. These settings are uncalibrated.

Vectors are L2-normalized before cosine scoring. Choice and Score use a numerically stable softmax:

```text
p_i = softmax(cosine(query, candidate_i) / temperature)
choice = argmax(p_i)
score = sum(i * p_i)
confidence = 1 - entropy(p) / log(K)  # default
```

Temperature must be finite and positive. Lower temperatures concentrate probability on higher similarities; they do not change their ranking. Score can change as probability shifts between levels. There is no rounding or ordinal smoothing. CLM configures confidence as the top probability minus the mean of the others. Confidence measures distribution concentration, not correctness; for K=1 it is defined as 1. It is not claimed to match Jev's confidence formula.

The default embedding configurations use retrieval queries for Noul:

```text
with criteria:    sigmoid(a_with * (cos(q_question_state, q_true_criterion) - cos(q_question_state, q_false_criterion)) + b_with)
without criteria: sigmoid(a_without * cos(q_question, q_state) + b_without)
```

In the default criteria path, `q_question_state` joins the original instructions and state with a newline; each criterion query begins with `true: ` or `false: `. The fixed retrieval instruction prefixes all three queries. CLM uses a state/question query and true/false candidate documents for both Noul forms; both use the difference formula above. Choice/Score temperatures do not affect Noul. With the default Noul parameters, the true/false difference maps to approximately [0.000000002, 0.999999998], while a single cosine maps to [0.000045, 0.999955]. These are theoretical bounds; real embeddings may occupy a narrower range. A positive similarity producing an output above 0.5 is not evidence of understanding a proposition. Negation, mention, and actual requests need separate labeled evaluation.

Configurations can load externally fitted parameters:

```yaml
scoring:
  choice_temperature: 0.15
  score_temperature: 0.2
  noul_with_criteria: {slope: 5.0, intercept: -0.2}
  noul_without_criteria: {slope: 4.0, intercept: -2.0}
  calibration_status: externally_fitted
  calibration_record:
    model_revision: exact-commit-or-artifact-sha256
    template_version: v1
    serialization_version: configurable-text-v2
    scoring_method: temperature-and-logistic
    validation_dataset: your-independent-held-out-dataset
```

These numbers illustrate the configuration format; they are not recommended parameters. Fit each model, primitive, and Noul path independently. Record the model revision, template, serialization, precision, evaluation dataset, and scoring method. Do not fit the three official examples and report performance on the same examples as generalization.

Future fitting tools should evaluate NLL, Brier score, reliability curves, and class metrics on independent held-out data. Such tools are outside this MVP. Scalar calibration cannot recover negation or logical distinctions already lost by the embeddings.
