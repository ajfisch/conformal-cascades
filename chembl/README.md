# ChEMBL Experiment

In-silico screening of chemical compounds is a common task in drug discovery/repurposing, where the goal is to identify possibly effective drugs to manufacture and test ([Stokes et al., 2020](https://www.sciencedirect.com/science/article/pii/S0092867420301021)). Using the ChEMBL database benchmark ([Mayr et al., 2018)](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c8sc00148k)), we consider the task of screening molecules for combinatorial constraint satisfaction, where given a specified constraint such as *“has property A but not property B”*, we want to identify at least one molecule from a given set of candidates that has the desired attributes. Our cascade consists of (1) the score of a fast, non-neural Random Forest (RF) applied to binary Morgan fingerprints ([Rogers and Hahn, 2010](https://pubs.acs.org/doi/10.1021/ci100050t)), and (2) the score of a directed Message Passing NN ensemble ([Yang et al., 2019](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237)).

The pretrained files that give the outputs used in our conformal experiments are included in the root directory's data download instructions, and can be used  as is. Directly running the conformal evaluation can be done using:

```
python conformal_chembl.py
```

The following set of instructions can also be used to derive the input files (from the download) from scratch.

## Data preparation

We use data from the ChEMBL dataset. The raw data can be accessed by downloading:

```
wget https://cpcascade.s3.us-east-2.amazonaws.com/chembl.csv.gz && gunzip chembl.csv.gz
```

Once this is downloaded, our preprocessed splits (i.e., creating train, dev, and test with some frequency filtering, etc)  can be created by running:

```
python create_chembl_dataset.py --dataset_file chembl.csv
```

The output files will be stored in `../data/chembl`. Finally, as our MPN models will be using assorted auxiliary RDKit features, we can pre-generate these using `save_features.py`.

*N.B. these features can also be generated on the fly using the `--features_generator` option on `mpn.py`, though this is slow.*

## Training single property predictors

We start by training *single property predictors*. Note that ultimately we are interested in identifying the presence of property *combinations*, but these target combinations are unknown (and combinatorial data is sparse). Therefore, the property predictors are initially trained independently (and then assembled with conformal prediction).

### MPN
The MPN model is trained using `mpn.py` (which uses code adapted from the `chemprop` repository). See `python mpn.py --help` for options. Given an ensemble of any number of trained checkpoints (models will be saved to `--checkpoint_dir`), predictions can be made using `mpn_inference.py`.

### Random Forest
The RF model on Morgan fingerprints is trained using `random_forest.py`. See `python random_forest.py --help` for options. This file can be used for both training and predicting (see options `--do_train` and `--do_predict`, respectively).

## Combinatorial data
We produce the combinatorial property constraints using `create_chembl_combination.py`. Molecules are kept to their original train, dev, and test splits. Then, we further partition all property combinations (of order `k`) that appear in the dataset to property combination train, dev, and test splits. This implies that at test time we are making predictions for property combinations we have never tested on before, on molecules that we have never seen before. All molecules (up to a limit) within a *molecule* split that are labeled with a particular combination are kept as "candidates" for retrieval. 

Conformal scores for a particular combination are assembled using `combine_predictions.py`. This creates three main files:
- `examples.npy`: Matrix of size [num_examples, num_candidates, 2] where the last dimension are the conformal scores derived for combining `rf` and `mpn` independent property predictions, respectively. 
- `answers.npy`: Binary matrix of size [num_examples, num_candidates] indicating which candidates satisfy the target constraint.
- `mask.npy`: Binary matrix of size [num_examples, num_candidates] to indicate padding.
- `references.py`: Vector of size [num_examples] indicating the index of a correct candidate that is selected (arbitrarily) to be the "gold" reference.
