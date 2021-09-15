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
from rdkit import Chem

CHEMBL_PATH = "/data/scratch/fisch/third_party/chemprop/data/chembl.csv"

Molecule = collections.namedtuple(
    "Molecule", ["smiles", "targets"])


def filter_invalid_smiles(smiles):
    if not smiles:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumHeavyAtoms() == 0:
        return True
    return False


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
            if filter_invalid_smiles(smiles):
                continue
            datapoint = Molecule(smiles, {t: int(row[t]) for t in target_columns if row[t]})
            dataset.append(datapoint)

        return dataset


def split_molecules(dataset, args):
    """Partition molecules into splits."""
    train_size = int(args.train_p * len(dataset))
    val_size = int(args.val_p * len(dataset))

    np.random.shuffle(dataset)
    train = dataset[:train_size]
    val = dataset[train_size:train_size + val_size]
    test = dataset[train_size + val_size:]

    print("=" * 50)
    print("Num train molecules: %d" % len(train))
    print("Num val molecules: %d" % len(val))
    print("Num test molecules: %d" % len(test))
    print("=" * 50)

    return train, val, test


def main(args):
    np.random.seed(args.seed)

    # Load chembl dataset.
    dataset = load_dataset(args.dataset_file)

    # Split dataset molecules into train/val/test.
    molecule_splits = split_molecules(dataset, args)

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

    # For each split write:
    # 1) smiles --> targets CSV
    # 2) combination --> smiles (positive/negative)
    os.makedirs(args.output_dir, exist_ok=True)
    for i, split in enumerate(["train", "val", "test"]):
        molecule_file = os.path.join(args.output_dir, "%s_molecules.csv" % split)
        with open(molecule_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles"] + targets)
            for molecule in molecule_splits[i]:
                row = [molecule.smiles]
                for target in targets:
                    row.append(str(molecule.targets.get(target, "")))
                writer.writerow(row)

    with open(os.path.join(args.output_dir, "targets.json"), "w") as f:
        json.dump(targets, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_p", type=float, default=0.6)
    parser.add_argument("--val_p", type=float, default=0.2)
    parser.add_argument("--min_positive_molecules_per_target", type=int, default=1)
    parser.add_argument("--min_negative_molecules_per_target", type=int, default=1)
    parser.add_argument("--min_molecules_per_target", type=int, default=100)
    parser.add_argument("--dataset_file", default=CHEMBL_PATH, type=str,
                        help="Path to ChEMBL dataset.")
    parser.add_argument("--output_dir", default="../data/chembl", type=str,
                        help="Path to directory where tasks will be written.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducibility.")
    main(parser.parse_args())
