# Official examples and scoring

## Official Jev examples with kalm-embedding-v2.5 results

These Jev documentation examples use `model: kalm-embedding-v2.5`. The saved Jev outputs appear alongside KaLM-embedding-multilingual-mini-instruct-v2.5 predictions for reference.

### Evaluation settings

Inference used CPU FP32 with caching disabled. The [KaLM-embedding-multilingual-mini-instruct-v2.5 configuration](../configs/kalm-embedding-v2.5.yaml) sets Choice/Score temperature **0.1** and Noul slope **10**, intercept **0**, for both Noul paths. Choice/Score probabilities use softmax over cosine similarities; Noul applies a sigmoid to one similarity or a true-minus-false difference. Normalized-entropy confidence measures distribution concentration. **These predictions are uncalibrated** and may vary slightly by hardware, precision, or revision. See [input mapping and scoring](#input-mapping-and-scoring) and [calibration](calibration.md).

### Example results

| Example | Jev reference output | JevEmbed with KaLM-embedding-multilingual-mini-instruct-v2.5 |
| --- | --- | --- |
| Choice: exchange routing | `returns`, probability 1.0 | `returns`, probability 0.793768 |
| Score: Safari bug severity | 1.43 | 1.256149 |
| Noul: human escalation | 0.99 | 0.999815 |
| Noul: repeat contact | 0.93 | 0.520574 |

The reference outputs are saved Jev documentation predictions. Neither column is ground truth; these examples illustrate behavior, not accuracy.

The same requests were run on the six original base models with CUDA BF16; see the [example results](../examples/README.md).

### Choice: route an exchange request

Source: [Jev Choice example](https://docs.typesafe.ai/primitives/choice). Saved [official reference response](../tests/fixtures/official_choice_exchange.reference.json).

Request ([file](../examples/official_choice_exchange.json)):

```json
{
  "model": "kalm-embedding-v2.5",
  "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    }
  }
}
```

KaLM-embedding-multilingual-mini-instruct-v2.5 response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "department": {
      "type": "choice",
      "probabilities": {
        "returns": 0.7937679922689964,
        "shipping": 0.131790116809968,
        "billing": 0.07444189092103566
      },
      "confidence": 0.4139963162182463,
      "choice": "returns"
    }
  },
  "usage": {
    "input_tokens": 61,
    "output_tokens": 0
  }
}
```

### Score: assess a Safari bug

Source: [Jev Score example](https://docs.typesafe.ai/primitives/score). Saved [official reference response](../tests/fixtures/official_score_safari.reference.json).

Request ([file](../examples/official_score_safari.json)):

```json
{
  "model": "kalm-embedding-v2.5",
  "state": "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.",
  "questions": {
    "bug_severity": {
      "type": "score",
      "instructions": "How severe is the reported issue?",
      "criteria": [
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but workaround exists",
        "Blocking issue; no workaround exists"
      ]
    }
  }
}
```

KaLM-embedding-multilingual-mini-instruct-v2.5 response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "bug_severity": {
      "type": "score",
      "probabilities": {
        "0": 0.15376904118628926,
        "1": 0.43631340883021275,
        "2": 0.40991754998349794
      },
      "confidence": 0.07579551491122716,
      "score": 1.2561485087972086,
      "legend": {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists"
      }
    }
  },
  "usage": {
    "input_tokens": 62,
    "output_tokens": 0
  }
}
```

### Noul: human escalation and repeat contact

Source: [Jev Noul example](https://docs.typesafe.ai/primitives/noul). Saved [official reference response](../tests/fixtures/official_noul_escalation.reference.json).

Request ([file](../examples/official_noul_escalation.json)):

```json
{
  "model": "kalm-embedding-v2.5",
  "state": "I have asked three times now. Can I please just talk to a real person?",
  "questions": {
    "is_human_escalation": {
      "type": "noul",
      "instructions": "Is the customer asking for a human agent?"
    },
    "is_repeat_contact": {
      "type": "noul",
      "instructions": "Has the customer contacted support about this before?",
      "criteria": {
        "true": "Mentions a prior attempt, ticket, or that they have asked before",
        "false": "No sign of any previous contact"
      }
    }
  }
}
```

KaLM-embedding-multilingual-mini-instruct-v2.5 response:

```json
{
  "model": "kalm-embedding-v2.5",
  "answers": {
    "is_human_escalation": {
      "type": "noul",
      "noul": 0.9998150635615807
    },
    "is_repeat_contact": {
      "type": "noul",
      "noul": 0.5205740521193529
    }
  },
  "usage": {
    "input_tokens": 136,
    "output_tokens": 0
  }
}
```

The criteria-free path compares the question with the state; the other compares true and false criteria. Both outputs exceed 0.5 here, but high state/question similarity is not a calibrated probability of agreement.

## Input mapping and scoring

JevEmbed converts each request into embedding inputs, computes cosine similarities, and applies the scoring rule for the selected primitive. The following cases show the model inputs for the [official examples](#official-jev-examples-with-kalm-embedding-v25-results) using the default templates.

### Choice — select a candidate

**Case: route a shoe exchange request.** Combine the original instructions and state into one query:

```text
Instruct: Which team should handle this?
Query: My running shoes arrived in the wrong size. Can I swap them for a size 10?
```

Encode each candidate line separately, including its name:

```text
returns: Exchanges, wrong or damaged items
shipping: Delivery status, delays, lost packages
billing: Charges, invoices, payment problems
```

**Scoring:** compare the query with each candidate, apply `softmax(similarities / temperature)`, and select the candidate with the highest probability. A null description uses the candidate name alone.

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** `returns`, probability **0.793768** at temperature **0.1**.

### Score — estimate an ordinal value

**Case: rate a Safari bug's severity.** Encode the question:

```text
Instruct: How severe is the reported issue?
Query: The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.
```

Encode each level description separately, without adding level numbers:

```text
Cosmetic; no impact to functionality
Broken or degraded feature, but workaround exists
Blocking issue; no workaround exists
```

**Scoring:** apply softmax and compute the expected level index: `score = 0*p0 + 1*p1 + 2*p2`. For K levels, the prediction ranges from 0 to K−1.

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** probabilities approximately **[0.153769, 0.436313, 0.409918]**, giving a score of **1.256149** at temperature **0.1**. The response also retains the descriptions as its legend.

### Noul with criteria — compare the question and state with each criterion

**Case: detect repeat contact.** Encode the question and state together under the retrieval instruction:

```text
Instruct: Retrieve semantically similar text.
Query: Has the customer contacted support about this before?
I have asked three times now. Can I please just talk to a real person?
```

Encode each criterion as a separate query, prefixed with its true/false label:

```text
Instruct: Retrieve semantically similar text.
Query: true: Mentions a prior attempt, ticket, or that they have asked before
```

```text
Instruct: Retrieve semantically similar text.
Query: false: No sign of any previous contact
```

**Scoring:** compare the question-plus-state query with each criterion query, subtract the false similarity from the true similarity, and apply `sigmoid(difference / 0.1)`.

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** **0.520574** with slope **10.0** and intercept **0.0**. The true and false similarities were **0.756465** and **0.748231**; the margin is small, so this single result does not establish reliable repeat-contact detection.

### Noul without criteria — compare two queries

**Case: detect a request for a human agent.** Encode the original question and state separately under the same fixed retrieval instruction:

```text
Instruct: Retrieve semantically similar text.
Query: Is the customer asking for a human agent?
```

```text
Instruct: Retrieve semantically similar text.
Query: I have asked three times now. Can I please just talk to a real person?
```

**Scoring:** use `sigmoid(cosine(question, state) / 0.1)`. There is no negative comparison in this path, so semantic relevance alone can produce a high score.

**KaLM-embedding-multilingual-mini-instruct-v2.5 result:** **0.999815** with slope **10.0** and intercept **0.0**. The high value reflects semantic similarity between the state and question; this path has no negative comparison and can also score related negative cases highly.

### Interpreting outputs

Temperature **0.1** sharpens Choice/Score probabilities without changing their ranking. Both Noul paths use slope **10** and intercept **0**, with separate settings for a similarity difference and a single similarity. Confidence measures concentration, not correctness; Noul has no confidence field. See [scoring and calibration](calibration.md) for details, `--explain` for rendered inputs, and the [runtime guide](runtime.md) for model and cache behavior.
