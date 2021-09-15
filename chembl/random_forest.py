"""Train separate random forest for each property."""

import argparse
import json
import chemprop
import csv
import logging
import multiprocessing.pool
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import sklearn
import subprocess
import tqdm


def load_data(dataset_file, args):
    """Load molecules from file with morgan fingerprints."""
    mols = []
    num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
    with open(dataset_file, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in tqdm.tqdm(reader, total=num_lines - 1, desc="reading dataset"):
            smiles = row[columns[0]]
            targets = [float(row[c]) if row[c] else None for c in columns[1:]]
            mol = chemprop.data.MoleculeDatapoint(smiles=smiles, targets=targets)
            mols.append(mol)
    dataset = chemprop.data.MoleculeDataset(mols)

    morgan_fingerprint = chemprop.features.get_features_generator("morgan")
    for datapoint in tqdm.tqdm(dataset, desc="generating features"):
        datapoint.set_features(morgan_fingerprint(mol=datapoint.smiles,
                                                  radius=args.radius,
                                                  num_bits=args.num_bits))
    return dataset


def train_single_task(task_num):
    """Train random forest, one for each target property."""
    global train_data, val_data, args

    # Only get features and targets for molecules where target is not None
    if not any([targets[task_num] is not None for targets in train_data.targets()]):
        return task_num, None, np.nan

    train_features, train_targets = zip(
        *[(features, targets[task_num])
          for features, targets in zip(train_data.features(), train_data.targets())
          if targets[task_num] is not None])
    train_features = np.stack(train_features)

    if any([targets[task_num] is not None for targets in val_data.targets()]):
        val_features, val_targets = zip(
            *[(features, targets[task_num])
              for features, targets in zip(val_data.features(), val_data.targets())
              if targets[task_num] is not None])
        val_features = np.stack(val_features)
    else:
        val_features, val_targets = None, None

    # Train single model.
    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=args.num_trees,
        max_depth=args.max_depth,
        n_jobs=1)
    model.fit(train_features, train_targets)

    # Evaluate.
    if val_features is not None:
        preds = model.predict_proba(val_features)
        if preds.shape[1] == 2:
            preds = preds[:, 1:]
        val_targets = [[target] for target in val_targets]

        score = chemprop.train.evaluate_predictions(
            preds=preds,
            targets=val_targets,
            num_tasks=1,
            metric_func=chemprop.utils.get_metric_func("auc"),
            dataset_type="classification",
            logger=logging.getLogger(__name__))
    else:
        score = np.nan

    return task_num, model, score


def train_init_fn(_train_data, _val_data, _args):
    global train_data, val_data, args
    train_data = _train_data
    val_data = _val_data
    args = _args


def train_forest(train_data, val_data, args):
    num_tasks = train_data.num_tasks()
    models = [None for _ in range(num_tasks)]
    scores = [np.nan for _ in range(num_tasks)]

    if args.threads > 0:
        workers = multiprocessing.pool.Pool(
            args.threads,
            initializer=train_init_fn,
            initargs=(train_data, val_data, args))
        map_fn = workers.imap_unordered
    else:
        train_init_fn(train_data, val_data, args)
        map_fn = map

    with tqdm.tqdm(total=num_tasks, desc="fitting individual tasks") as pbar:
        for task_num, model, score in map_fn(train_single_task, range(num_tasks)):
            models[task_num] = model
            scores[task_num] = score
            pbar.update()

    print("Average validation auc: %2.2f" % np.mean([v for v in scores if not np.isnan(v)]))
    return models


def predict_single_task(model):
    global data
    features = data.features()
    preds = model.predict_proba(features)
    if preds.shape[1] == 2:
        preds = preds[:, 1:]
    return preds[:, 0].tolist()


def predict_init_fn(_data):
    global data
    data = _data


def predict_forest(models, data, args):
    num_tasks = data.num_tasks()
    all_smiles = data.smiles()
    smiles_to_scores = {}

    if args.threads > 0:
        workers = multiprocessing.pool.Pool(
            args.threads,
            initializer=predict_init_fn,
            initargs=(data,))
        map_fn = workers.imap_unordered
    else:
        predict_init_fn(data)
        map_fn = map

    with tqdm.tqdm(total=num_tasks, desc="predicting individual tasks") as pbar:
        for task_num, results in enumerate(map_fn(predict_single_task, models)):
            for smiles, score in zip(all_smiles, results):
                if smiles not in smiles_to_scores:
                    smiles_to_scores[smiles] = [0 for _ in range(num_tasks)]
                smiles_to_scores[smiles][task_num] = score
            pbar.update()

    return smiles_to_scores


def main(args):
    if args.do_train:
        train_data = load_data(args.train_data, args)
        val_data = load_data(args.val_data, args)
        models = train_forest(train_data, val_data, args)
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(models, f)

    if args.do_predict:
        print("Loading model")
        with open(os.path.join(args.model_dir, "model.pkl"), "rb") as f:
            models = pickle.load(f)
        predict_data = load_data(args.predict_data, args)
        smiles_to_scores = predict_forest(models, predict_data, args)
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(smiles_to_scores, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", pl.utilities.parsing.str_to_bool)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--num_bits", type=int, default=2048)
    parser.add_argument("--num_trees", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--train_data", type=str, default="../data/chembl/train_molecules.csv")
    parser.add_argument("--val_data", type=str, default="../data/chembl/val_molecules.csv")
    parser.add_argument("--predict_data", type=str, default="../data/chembl/train_molecules.csv")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_dir", type=str, default="../ckpts/chembl/random_forest")
    parser.add_argument("--output_file", type=str, default="../ckpts/chembl/random_forest/preds_train.json")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count() - 1)
    args = parser.parse_args()
    main(args)
