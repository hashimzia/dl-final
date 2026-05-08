# Deep Learning Final - Spring 2026

This repository contains the Kaggle notebooks and supporting files for my Deep
Learning final project on fine-tuning a vision-language model for scientific
multiple-choice question answering.

## Overview

The project uses the official `HuggingFaceTB/SmolVLM-500M-Instruct` checkpoint for
the Pixels to Predictions ScienceQA-style vision challenge. The system predicts a
single 0-indexed answer choice for each image-grounded science question.

The final leaderboard-oriented notebook combines:

- QLoRA fine-tuning under the 5M trainable-parameter cap
- next-token letter-logit multiple-choice scoring
- choice-order test-time augmentation
- validation-gated parser overrides for repeated visual reasoning templates

The notebooks support two modes:

1. **Load saved adapter weights directly** for fast reproducible inference
2. **Retrain from the pretrained base model** using only the provided competition data

The main submission notebook is currently configured to load saved weights by default.
To retrain, set `LOAD_WEIGHTS = False` near the top of the notebook.

## Main Notebooks

- `final-override-parser.ipynb`: final Kaggle-facing notebook for the direct
  parser-override system.
- `starter_notebook_0827_visual_facts.ipynb`: visual-facts variant where parsers add
  extracted image evidence to the prompt, while the VLM still makes the final answer
  decision.

## Model Weights

To save time, the final override-parser notebook can load the trained LoRA adapter
directly instead of re-running fine-tuning.

### Public Kaggle model link

[Final best weights](https://www.kaggle.com/models/hashimzia1/final-best-weights/)

### Kaggle notebook path

Inside Kaggle, the final override-parser notebook loads the adapter from:

```text
/kaggle/input/models/hashimzia1/final-best-weights/pytorch/default/1/best
```

This path is used when:

```python
LOAD_WEIGHTS = True
```

If `LOAD_WEIGHTS = False`, the notebook fine-tunes a fresh LoRA adapter from the
official pretrained SmolVLM checkpoint and saves the trained adapter under
`/kaggle/working/override_parser/best`.

## Data

The notebooks expect the competition data to be attached in Kaggle at:

```text
/kaggle/input/competitions/pixels-to-predictions
```

The submission file is written to:

```text
/kaggle/working/submission.csv
```

with the required schema:

```text
id,answer
```

where `answer` is the predicted 0-indexed answer choice.
