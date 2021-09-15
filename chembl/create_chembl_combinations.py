"""Prepare ChEMBL dataset for combinatoric property prediction.

1) Filter valid smiles.
2) Split molecules randomly to train/dev/test.
3) Split property combinations to train/dev/test.
"""

import argparse
import collections
import csv
import itertools
import json
import numpy as np
import os
import subprocess
import tqdm

Molecule = collections.namedtuple(
    "Molecule", ["smiles", "targets"])


def load_dataset(path):
    """Return list of molecules --> attributes."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        smiles_column = columns[0]
        target_columns = columns[1:]

        # Read in all the dataset smiles.
        dataset = []
        num_lines = int(subprocess.check_output(["wc", "-l", path], encoding="utf8").split()[0])
        for row in tqdm.tqdm(reader, total=num_lines, desc="reading smiles"):
            smiles = row[smiles_column]
            datapoint = Molecule(smiles, {t: int(row[t]) for t in target_columns if row[t]})
            dataset.append(datapoint)

        return dataset


def get_labeled_combinations(molecules, valid_targets, combinations, K):
    """For each combination, return set of molecules that have it annotated."""
    combination_to_molecules = collections.defaultdict(list)
    for molecule in tqdm.tqdm(molecules, desc="checking combinations"):
        targets = [t for t in molecule.targets.keys() if t in valid_targets]
        if len(targets) < K:
            continue
        for combination in itertools.combinations(targets, K):
            if tuple(combination) in combinations:
                combination_to_molecules[combination].append(molecule)

    return combination_to_molecules


def make_combinations(targets, train_molecules, val_molecules, test_molecules, args):
    """Make property combinations for each split of the data."""
    # Take all n choose k combinations.
    combinations = [tuple(c) for c in itertools.combinations(targets, args.K)]
    np.random.shuffle(combinations)
    combinations = set(combinations)

    # Gather molecules that are annotated for each combination.
    labeled_train_combinations = get_labeled_combinations(train_molecules, targets, combinations, args.K)
    labeled_val_combinations = get_labeled_combinations(val_molecules, targets, combinations, args.K)
    labeled_test_combinations = get_labeled_combinations(test_molecules, targets, combinations, args.K)

    # Filter combinations to those that have at least N total molecules
    # at least one of which is positive.
    train_combinations = []
    val_combinations = []
    test_combinations = []

    def _check(split, combination, assignment):
        """Check if a combination is valid for a given split."""
        # Check if split has enough examples.
        if split == "train":
            if len(train_combinations) >= args.max_train:
                return False
            labeled_set = labeled_train_combinations
        elif split == "val":
            if len(val_combinations) >= args.max_val:
                return False
            labeled_set = labeled_val_combinations
        else:
            if len(test_combinations) >= args.max_test:
                return False
            labeled_set = labeled_test_combinations

        # Check if combination in set.
        if combination not in labeled_set:
            return False

        # Check if combination has enough candidates.
        if len(labeled_set[combination]) < args.min_molecules_per_combination:
            return False

        # Check if at least one candidate matches the constraint.
        # Check that positive to negative ratio is low enough.
        num_positive = 0
        total = 0
        for molecule in labeled_set[combination]:
            if all([molecule.targets[combination[i]] == assignment[i] for i in range(len(combination))]):
                num_positive += 1
            total += 1
        if num_positive < 1:
            return False

        total = min(total, args.max_molecules_per_combination)
        if num_positive / total > args.max_positive_ratio_per_combination:
            return False

        return True

    def _make_combination(combinations, combination, assignment):
        """Create the combination example data."""
        molecules = combinations[combination]
        negatives = []
        positives = []
        for molecule in molecules:
            if all([molecule.targets[combination[i]] == assignment[i] for i in range(len(combination))]):
                positives.append(molecule.smiles)
            else:
                negatives.append(molecule.smiles)

        if len(positives) > args.max_molecules_per_combination / 2:
            indices = np.random.permutation(len(positives))[:args.max_molecules_per_combination // 2]
            positives = [positives[i] for i in indices]
        if len(negatives) > args.max_molecules_per_combination - len(positives):
            target_num = args.max_molecules_per_combination - len(positives)
            indices = np.random.permutation(len(negatives))[:target_num]
            negatives = [negatives[i] for i in indices]

        return dict(combination=combination, assignment=assignment, positives=positives, negatives=negatives)

    def _powerset(sets, cur, n):
        if n == 1:
            sets.append(cur + [0])
            sets.append(cur + [1])
        else:
            _powerset(sets, cur + [0], n - 1)
            _powerset(sets, cur + [1], n - 1)
        return sets

    for combination in tqdm.tqdm(combinations, desc="allocating combinations"):
        # Allocate a random assignment to the combination.
        # Assign the combination to a split.
        assignments = _powerset([], [], len(combination))
        np.random.shuffle(assignments)
        for assignment in assignments:
            if _check("test", combination, assignment):
                test_combinations.append(_make_combination(labeled_test_combinations, combination, assignment))
                break
            elif _check("val", combination, assignment):
                val_combinations.append(_make_combination(labeled_val_combinations, combination, assignment))
                break
            elif _check("train", combination, assignment):
                train_combinations.append(_make_combination(labeled_train_combinations, combination, assignment))
                break

    print("=" * 50)
    avg_positive = sum([len(c["positives"]) for c in train_combinations]) / len(train_combinations)
    avg_negative = sum([len(c["negatives"]) for c in train_combinations]) / len(train_combinations)
    print("Num train combinations: %d (%d / %d)" % (len(train_combinations), avg_positive, avg_negative))

    avg_positive = sum([len(c["positives"]) for c in val_combinations]) / len(val_combinations)
    avg_negative = sum([len(c["negatives"]) for c in val_combinations]) / len(val_combinations)
    print("Num val combinations: %d (%d / %d)" % (len(val_combinations), avg_positive, avg_negative))

    avg_positive = sum([len(c["positives"]) for c in test_combinations]) / len(test_combinations)
    avg_negative = sum([len(c["negatives"]) for c in test_combinations]) / len(test_combinations)
    print("Num test combinations: %d (%d / %d)" % (len(test_combinations), avg_positive, avg_negative))
    print("=" * 50)

    return train_combinations, val_combinations, test_combinations


def main(args):
    np.random.seed(args.seed)

    # Split dataset molecules into train/val/test.
    molecule_splits = [load_dataset(args.train_file),
                       load_dataset(args.val_file),
                       load_dataset(args.test_file)]

    # Take only attributes that appear in the train split with over N molecules.
    train_molecules = molecule_splits[0]
    all_targets = set([t for mol in train_molecules for t in mol.targets.keys()])
    positive_counts = collections.defaultdict(int)
    negative_counts = collections.defaultdict(int)
    for molecule in train_molecules:
        for target in all_targets:
            if target in molecule.targets:
                if molecule.targets[target] == 0:
                    negative_counts[target] += 1
                else:
                    positive_counts[target] += 1

    # Make sure not all one type.
    targets = []
    for t in all_targets:
        if positive_counts[t] < args.min_positive_molecules_per_target:
            continue
        if negative_counts[t] < args.min_negative_molecules_per_target:
            continue
        if positive_counts[t] + negative_counts[t] < args.min_molecules_per_target:
            continue
        targets.append(t)
    print("Number of targets: %d" % len(targets))

    # Split dataset property constraints into train/val/test.
    combination_splits = make_combinations(targets, *molecule_splits, args)

    # For each split write:
    # 1) smiles --> targets CSV
    # 2) combination --> smiles (positive/negative)
    os.makedirs(args.output_dir, exist_ok=True)
    for i, split in enumerate(["train", "val", "test"]):
        combination_file = os.path.join(args.output_dir, "%s_combinations.jsonl" % split)
        with open(combination_file, "w") as f:
            for combination in combination_splits[i]:
                f.write(json.dumps(combination) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--max_train", type=int, default=100000)
    parser.add_argument("--max_val", type=int, default=5000)
    parser.add_argument("--max_test", type=int, default=5000)
    parser.add_argument("--min_positive_molecules_per_target", type=int, default=100)
    parser.add_argument("--min_negative_molecules_per_target", type=int, default=100)
    parser.add_argument("--min_molecules_per_target", type=int, default=1000)
    parser.add_argument("--max_molecules_per_combination", type=int, default=1000)
    parser.add_argument("--min_molecules_per_combination", type=int, default=100)
    parser.add_argument("--max_positive_ratio_per_combination", type=float, default=0.5)
    parser.add_argument("--train_file", default="../data/chembl/train_molecules.csv")
    parser.add_argument("--val_file", default="../data/chembl/val_molecules.csv")
    parser.add_argument("--test_file", default="../data/chembl/test_molecules.csv")
    parser.add_argument("--output_dir", default="../data/chembl/easy", type=str,
                        help="Path to directory where tasks will be written.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducibility.")
    main(parser.parse_args())
