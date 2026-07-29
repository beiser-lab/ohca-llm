# Validating the sequential Qwen classifier

Before any production rerun, validate the production classifier
(`classify_note_qwen`) against a small set of **reviewed** cases inside the
HIPAA-compliant environment (Randi). The goal is not a headline accuracy number
but to confirm the Step 1 arrest-existence guardrail holds on the hard negatives
that broke the exploratory single-call classifier.

## What to include

Assemble a CSV with, at minimum, a note-text column and a binary ground-truth
column. A category column is strongly recommended so errors are reported per
subgroup rather than as a single specificity figure.

| column         | meaning                                    |
| -------------- | ------------------------------------------ |
| `note_text`    | the clinical note text                     |
| `manual_label` | reviewer ground truth: `1` OHCA, `0` not   |
| `category`     | subgroup label (see below)                 |
| `id`           | optional stable identifier                 |

Deliberately over-sample these **hard negatives** — critical illness with **no
documented arrest**:

- respiratory failure / hypoxemia on BiPAP or intubation
- intubation for airway protection without arrest
- altered mental status / unresponsiveness without arrest
- nursing-home / SNF / assisted-living origin without arrest

and include genuine positives, especially **out-of-hospital arrests that
originate at a nursing home / SNF**, to confirm the residential-origin rescue
does not swing too far the other way.

## Running

Point the model server (vLLM) at your Qwen instruct model, then:

```bash
ohca-llm validate \
    --input reviewed_validation.csv \
    --text-col note_text \
    --label-col manual_label \
    --category-col category \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen2.5-7B-Instruct \
    --out validation_scored.csv
```

The report prints overall sensitivity/specificity/PPV/NPV/F1 and a per-category
breakdown flagging any category with a false positive. `validation_scored.csv`
carries the per-case `llm_label`, `llm_step_detail`, `steps_passed`,
`residential_rescue`, and `llm_rationale` so each disagreement can be traced to
the step that produced it.

## Smoke-testing the harness (no PHI)

To exercise the harness and report formatting without touching patient data:

```bash
ohca-llm validate --synthetic --base-url http://127.0.0.1:8000/v1
```

This uses `ohca_llm.validation_fixtures.load_synthetic_validation_set()`, a set
of fabricated notes covering each hard-negative category. It is for plumbing and
guardrail checks only — it is not a substitute for validation on reviewed data.

## The keyword gate and `--score-all-nonblank`

Review of a production run surfaced false positives on notes that were
**keyword-negative**: respiratory arrest with a maintained pulse, opioid
overdose reversed with naloxone, AMS / stroke / seizure, mechanical falls, and
near-empty ("erroneous documentation") notes. These notes contain no
cardiac-arrest keyword, so the deterministic keyword gate would have labeled
them Not OHCA with no LLM call. They received positive LLM labels only because
the run used `--score-all-nonblank`, which bypasses the gate and sends every
nonblank note to the model.

The lesson is not to add a new LLM rule-out stage — the two-stage design already
exists (keyword gate -> sequential chain) and was simply being bypassed. The
fix is to run with the gate on (drop `--score-all-nonblank`), provided the gate
does not drop true OHCA.

`validate` runs a **keyword-gate audit** before any LLM calls and prints it. The
number that governs the decision is `True OHCA WRONGLY dropped`:

- **Zero** -> the gate rules out only true negatives; safe to run gated. Drop
  `--score-all-nonblank`, and the keyword-negative false positives disappear.
- **Non-zero** -> the gate would miss real arrests. The audit lists them so
  `CARDIAC_KEYWORDS` can be widened deliberately. Do **not** widen it with terms
  like "respiratory arrest" or "bagged" — those re-admit exactly the
  false-positive population above.

Note the gate is a plain substring match with no negation handling: a note that
explicitly says "no CPR, never pulseless" still trips the keyword and is
forwarded to the LLM. Real keyword-negative notes do not contain the terms at
all, so this does not affect the population above, but it is a reason to keep
Step 1 conservative for the notes that do reach the model.

`ohca_llm.validation_fixtures.load_keyword_negative_set()` provides PHI-free
synthetic versions of this false-positive population for regression testing; the
suite asserts the gate rules out every one of them without an LLM call.



- **Any false positive in a `hard_negative_*` category** is the failure mode
  this harness exists to catch. Inspect `llm_step_detail`: a Step 1 `YES` on a
  note with no arrest evidence means the guardrail was bypassed; a Step 4
  flip via `residential_rescue` means the regex rescue over-fired.
- **False negatives on `positive_snf_arrest`** suggest the residential-origin
  rescue or an over-eager transfer call is excluding genuine nursing-home
  arrests. Re-check the `residential_rescue` reason on those rows.
