# Efficient Conformal Prediction via Cascaded Inference with Expanded Admission
Code for [Efficient Conformal Prediction via Cascaded Inference with Expanded Admission](https://arxiv.org/abs/2007.03114).

## Abstract
Providing a small set of promising candidates in place of a single prediction is well-suited for many open-ended classification tasks. <ins>Conformal Prediction</ins> (CP) is a technique for creating classifiers that produce a valid set of predictions that contains the true answer with arbitrarily high probability. In practice, however, standard CP can suffer from both low *predictive* and *computational* efficiency during inference—i.e., the predicted set is both unusably large and costly to obtain. This is particularly pervasive in the considered setting, where the correct answer is not unique and the number of total possible answers is high. In this work, we develop two simple and complementary techniques for improving both types of efficiencies. We first expand the CP correctness criterion to allow for additional, inferred "admissible" answers, which can substantially reduce the size of the predicted set while still providing valid performance guarantees. Second, we amortize costs by conformalizing prediction cascades, in which we aggressively prune implausible labels early on by using progressively stronger classifiers -- again, while still providing valid performance guarantees. We demonstrate the empirical effectiveness of our approach for multiple applications in natural language processing and computational chemistry for drug discovery.

## Setup

Running `python setup.py develop` will add the `cpcascade` to your python path. Installing `requirements.txt` via `pip` will give most of the required packages, although installing `chemprop` (and associated RDKit, etc) will still be necessary to run `chembl` experiments. Please see the [chemprop](https://github.com/chemprop/chemprop) repository for installation instructions.

## Conformal prediction

All of the code for analyzing cascaded conformal predictions is in the [cpcascades](cpcascades) directory. Examples for how to call into it are given in the tasks sub-directories. Note that in this repository, the functions exposed in `cpcascades` are for analysis only, i.e. they are not implemented as a real-time predictors.

Efficient implementations of *real-time* cascaded conformal predictors for the tasks considered here might be included later. As of now, this (experimental) repository mainly operates over offline, precomputed scores (using whatever task-specific model implementation).

## Tasks

Code for experiments on open-domain question answering ([qa](qa)), information retrieval for fact verification ([ir](ir)), and in-silico screening for drug discovery ([chembl](chembl)) can be found in their respective sub-directories (which also contain further instuctions).

The outputs we used in our experiments are available in the `data` directory after downloading and untarring the data, which one can do per task. First, make the data dir using `mkdir data`. Then run the following:

**ChEMBL**:
```
pushd data
wget https://cpcascade.s3.us-east-2.amazonaws.com/chembl.tar.gz && tar -xvf chembl.tar.gz
popd
```

**QA**:
```
pushd data
wget https://cpcascade.s3.us-east-2.amazonaws.com/open-qa.tar.gz && tar -xvf open-qa.tar.gz
popd
```

**IR**:
```
pushd data
wget https://cpcascade.s3.us-east-2.amazonaws.com/ir.tar.gz && tar -xvf ir.tar.gz
popd
```

The results in our paper (i.e., Table 1) can then be reproduced via the following commands:

```
./run_chembl.sh test
./run_qa.sh test
./run_ir.sh test
```

These commands only run a subset of the commands necessary to generate all results; see the individual sub-directory files for more details and options. Rows of Table 1 can be found in the "conformal_table.txt" file of the created results directories.

## Running an experiment

The main access point for running a conformal experiment using our library is via importing:

```python
from cpcascade import run_experiment
```

Then `run_experiment` takes in:

```python
run_experiment(
    # A Numpy array of shape <float>[num_examples, max_labels, num_metrics].
    # It should be preprocessed to be the values and ordering of the cascade.
    # Value (i, j, k) = kth nonconformity metric of jth label for ith example.
    examples,

    # A Numpy array of shape <float>[num_examples, max_labels]
    # This is a binary (0/1) indicator for whether label (i, j) is correct.
    answers,

    # A Numpy array of shape <float>[num_examples, max_labels]
    # Simply keeps track (0/1) as to whether a label is a real label or just padding.
    mask,

    # If N is the number of total examples, N is partitioned into separate calibration/test
    # splits (multiple times for multiple trials). Each is a list of i \in [0, M - 1].

    # calibration_ids are the example ids used for calibration.
    calibration_ids,

    # test_ids are the example ids used for testing.
    test_ids,

    # Baseline metrics are which nonconformal algorithm to test out (i.e., heuristic).
    baseline_metrics,

    # If comparing to normal CP, we also have to provide the original dataset references.
    # These are randomly sampled if more than one annotation was provided.
    # Note: This does *not* apply to the expanded admission CP criteria.
    references
)
```

The main work is to transform the raw data into Numpy arrays. See the individual experiments for examples of how this is done. The key conformal cascades logic can be found in the [conformal.py](cpcascade/conformal.py) file.


## Citation

If you use this in your work please cite:

```
@inproceedings{fisch2021efficient,
    title={Efficient Conformal Prediction via Cascaded Inference with Expanded Admission},
    author={Adam Fisch and Tal Schuster and Tommi Jaakkola and Regina Barzilay},
    booktitle={Proceedings of The Tenth International Conference on Learning Representations},
    year={2021},
}
```
