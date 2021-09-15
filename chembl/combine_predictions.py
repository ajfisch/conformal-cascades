"""Combine metrics to make large conformal prediction matrices."""

import argparse
import os
import subprocess
import json
import tqdm
import numpy as np


def load_dataset(path):
    dataset = []
    num_lines = int(subprocess.check_output(["wc", "-l", path], encoding="utf8").split()[0])
    with open(path, "r") as f:
        for line in tqdm.tqdm(f, total=num_lines, desc="reading dataset"):
            dataset.append(json.loads(line))
    return dataset


def main(args):
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_file)

    print("Loading MPN predictions...")
    with open(args.mpn_file, "r") as f:
        mpn = json.load(f)

    print("Loading RF predictions...")
    with open(args.rf_file, "r") as f:
        rf = json.load(f)

    print("Loading targets...")
    with open(args.targets_file, "r") as f:
        targets = json.load(f)
        target2index = {t: i for i, t in enumerate(targets)}

    num_examples = len(dataset)
    num_labels = max([len(ex["positives"]) + len(ex["negatives"]) for ex in dataset])
    num_metrics = 2

    # Create matrices.
    print("Matrix size: %d x %d x % d" % (num_examples, num_labels, num_metrics))
    example_matrix = np.ones((num_examples, num_labels, num_metrics)) * 1e12
    answer_matrix = np.zeros((num_examples, num_labels))
    label_mask = np.zeros((num_examples, num_labels))
    references = np.zeros(num_examples)

    # Iterate examples.
    for i, example in enumerate(tqdm.tqdm(dataset, desc="creating matrices")):
        combination = example["combination"]
        assignment = example["assignment"]
        positives = example["positives"]
        negatives = example["negatives"]

        # Random correct answer for reference.
        random_correct = np.random.randint(len(positives))
        references[i] = random_correct

        # Add labels.
        offset = 0
        for label_type, labels in zip([1, 0], [positives, negatives]):
            for j, smiles in enumerate(labels):
                j = j + offset
                answer_matrix[i, j] = label_type
                label_mask[i, j] = 1
                for k, metric in enumerate([rf, mpn]):
                    scores = metric[smiles]
                    combination_score = 0
                    for target, setting in zip(combination, assignment):
                        idx = target2index[target]
                        if setting == 1:
                            # Setting == 1 means that this property is ACTIVE.
                            score = scores[idx]
                        else:
                            # Setting == 0 means that this property is INACTIVE.
                            score = 1 - scores[idx]
                        # Naively combine individual property predictions (ignoring correlation).
                        combination_score += np.log(score + 1e-12)
                    example_matrix[i, j, k] = -combination_score
            offset += len(labels)

    print("Saving to disk")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "examples.npy"), "wb") as f:
        np.save(f, example_matrix)
    with open(os.path.join(args.output_dir, "answers.npy"), "wb") as f:
        np.save(f, answer_matrix)
    with open(os.path.join(args.output_dir, "mask.npy"), "wb") as f:
        np.save(f, label_mask)
    with open(os.path.join(args.output_dir, "references.npy"), "wb") as f:
        np.save(f, references)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_file", type=str, default="../data/chembl/targets.json")
    parser.add_argument("--dataset_file", type=str, default="../data/chembl/easy/val_combinations.jsonl")
    parser.add_argument("--mpn_file", type=str, default="../ckpts/chembl/independent_mpn/pred_val.json")
    parser.add_argument("--rf_file", type=str, default="../ckpts/chembl/random_forest/preds_val.json")
    parser.add_argument("--output_dir", type=str, default="../data/chembl/combinations/val")
    args = parser.parse_args()
    main(args)
