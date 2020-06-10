"""Evaluation for chem."""

import collections
import copy
import json
import os
import tqdm

from absl import app
from absl import flags
from absl import logging

import numpy as np
from cpcascade import run_experiment

DEFAULT_EVAL = None

flags.DEFINE_string("eval_file", DEFAULT_EVAL, "Path to scored molecules for development.")

flags.DEFINE_integer("num_positives", 3, "Mean number of positives per screen.")

flags.DEFINE_integer("sample_size", 100, "Number of molecules per screen.")

flags.DEFINE_integer("num_trials", 5000, "Number of experiments for testing.")

flags.DEFINE_string("trials_file", None,
                    "The output file where the trials qids will be written.")

flags.DEFINE_boolean("overwrite_trials", False,
                     "Recompute and overwrite the CV folds if they exist.")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to evaluate.")

flags.DEFINE_list("metrics", "chemprop_optimized",
                  "The conformal scores to use (comma separated).")

FLAGS = flags.FLAGS


def create_trials(positives, negatives, num_positives, sample_size, num_trials, percent=0.8):
    """Split molecules into multiple experiments."""
    global_qid = 0
    calibration_qids = []
    test_qids = []

    # Repeat the experiment N times.
    for _ in tqdm.tqdm(range(num_trials)):
        # Group positives.
        np.random.shuffle(positives)
        groups = []
        offset = 0
        while offset < len(positives):
            n = 0
            while n == 0:
                n = np.random.poisson(num_positives)
            groups.append(positives[offset:offset + n])
            offset += n

        split = int(percent * len(groups))
        calibration = {}
        for sample in groups[:split]:
            n = sample_size - len(sample)
            negative_sample = np.random.choice(negatives, n).tolist()
            calibration[global_qid] = sample + negative_sample
            global_qid += 1
        calibration_qids.append(calibration)

        testing = {}
        for sample in groups[split:]:
            n = sample_size - len(sample)
            negative_sample = np.random.choice(negatives, n).tolist()
            testing[global_qid] = sample + negative_sample
            global_qid += 1
        test_qids.append(testing)

    return calibration_qids, test_qids


def load_molecules(file_path):
    """Load molecule file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    positives = data["positives"]
    negatives = data["negatives"]
    for smiles, values in positives.items():
        for k, v in values.items():
            values[k] = float(v)
        values["text"] = smiles
    for smiles, values in negatives.items():
        for k, v in values.items():
            values[k] = float(v)
        values["text"] = smiles
    return positives, negatives


def get_metric(label, metric):
    """Select nonconformal score for label. Lower is better."""
    if metric == "random_forest":
        score = 1 - label["random_forest"]
    elif metric.startswith("rank") and metric in label:
        score = label[metric]
    elif metric == "svm":
        score = -1 * label["svm"]
    elif metric == "chemprop_optimized":
        score = 1 - label["chemprop_optimized"]
    else:
        raise ValueError("Unrecognized metric: %s" % metric)
    return score


def main(_):
    logging.set_verbosity(logging.INFO)
    np.random.seed(FLAGS.seed)

    FLAGS.epsilons = [float(f) for f in FLAGS.epsilons or []]

    logging.info("Loading data.")
    trials_qids = dev_qids = test_qids = None

    positives, negatives = load_molecules(FLAGS.eval_file)
    if not FLAGS.trials_file:
        FLAGS.trials_file = os.path.splitext(FLAGS.eval_file)[0]
        FLAGS.trials_file += ("-trials=%d-samples=%d-positives=%s.json" %
                              (FLAGS.num_trials, FLAGS.sample_size, FLAGS.num_positives))
        os.makedirs(os.path.dirname(FLAGS.trials_file), exist_ok=True)
    if os.path.exists(FLAGS.trials_file) and not FLAGS.overwrite_trials:
        logging.info("Loading trials from %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "r") as f:
            trials_qids = json.load(f)
    else:
        trials_qids = create_trials(
            positives=list(positives.keys()),
            negatives=list(negatives.keys()),
            num_positives=FLAGS.num_positives,
            sample_size=FLAGS.sample_size,
            num_trials=FLAGS.num_trials)
        logging.info("Writing qids to %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "w") as f:
            json.dump(trials_qids, f)

    # Create examples for all trials.
    dev_qids, test_qids = trials_qids
    all_examples = collections.defaultdict(list)
    qid2answers = collections.defaultdict(set)
    for i in tqdm.tqdm(range(FLAGS.num_trials), desc="creating trials"):
        for examples in [dev_qids[i], test_qids[i]]:
            for qid, screen_samples in examples.items():
                labels = []
                for smiles in screen_samples:
                    if smiles in positives:
                        labels.append(positives[smiles])
                        qid2answers[qid].add(smiles)
                    else:
                        labels.append(negatives[smiles])

                # Sort to get ranks.
                labels = [copy.deepcopy(y) for y in labels]
                for metric in ["svm", "random_forest", "chemprop_optimized"]:
                    labels = sorted(labels, key=lambda x: get_metric(x, metric))
                    for i, y in enumerate(labels):
                        y["rank_" + metric] = i
                all_examples[qid].extend(labels)

    if FLAGS.skip_conformal:
        suffix = ("/hiv/baselines-trials=%d-samples=%d-positives=%d" %
                  (FLAGS.num_trials, FLAGS.sample_size, FLAGS.num_positives))
    else:
        suffix = ("/hiv/trials=%d-samples=%d-positives=%d-smoothed=%s"
                  "-correction=%s-metrics=%s-equivalence=%s" %
                  (FLAGS.num_trials, FLAGS.sample_size, FLAGS.num_positives,
                   FLAGS.smoothed, FLAGS.correction, ",".join(FLAGS.metrics),
                   FLAGS.equivalence))
    FLAGS.output_dir += suffix
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    run_experiment(
        all_examples=all_examples,
        qid2answers=qid2answers,
        calibration_qids=dev_qids,
        test_qids=test_qids,
        baseline_metrics=["chemprop_optimized", "random_forest"],
        epsilons=FLAGS.epsilons,
        get_metric=get_metric)


if __name__ == "__main__":
    app.run(main)
