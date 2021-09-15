# Question Answering

## Open Domain Question Answering

We use the outputs from the DPR model.

## Extractive Question Answering

We use the SQuAD 2.0 dataset (available [here](https://rajpurkar.github.io/SQuAD-explorer/)). We repartition the dev set into `CP-dev` and `CP-test` splits, as the official test set is hidden. Our modeling code relies on the [transformers](https://github.com/huggingface/transformers) library (we use the standard QA model).