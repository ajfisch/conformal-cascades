"""Evaluation for FEVER IR."""

import collections
import json
import os
import jsonlines
from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging

import numpy as np
from cpcascade import run_experiment

DEFAULT_DEV_FILE = None
DEFAULT_TEST_FILE = None

flags.DEFINE_string("dev_predictions_file", None,
                    "Path to retriever's predictions for development set.")

flags.DEFINE_string("test_predictions_file", None,
                    "Path to retriever's predictions for development set.")

flags.DEFINE_integer("num_trials", 10, "Number of trials to run.")

flags.DEFINE_string("trials_file", None,
                    "The output file where the trials qids will be written.")

flags.DEFINE_boolean("overwrite_trials", False,
                     "Recompute and overwrite the trials qids if they exist.")

flags.DEFINE_string("dev_gold_file", DEFAULT_DEV_FILE,
                    "Dev set file with gold answers.")

flags.DEFINE_string("test_gold_file", DEFAULT_TEST_FILE,
                    "Test set file with gold answers.")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to evaluate.")

flags.DEFINE_list("metrics", "rank",
                  "The conformal scores to use (comma separated).")

FLAGS = flags.FLAGS


def load_predictions(file_path, metrics):
    """
    Load predictions file and gather predictions for the same query.
    all_examples[qid] = list(dict(properties of candidate sentence))
    """
    all_examples = {}
    max_labels = 0
    for case in jsonlines.Reader(open(file_path, "r")):
        qid = case["annotation_id"]
        if qid not in all_examples.keys():
            all_examples[qid] = []

        case["text"] = case["evidence"]
        all_examples[qid].append(case)

    max_labels = 0
    for qid, predictions in all_examples.items():
        sorted_predictions_sent = sorted(predictions, key=lambda y: -y["sent_bm25"])
        for rank, y in enumerate(sorted_predictions_sent):
            y["rank"] = rank

        all_examples[qid] = sorted_predictions_sent
        max_labels = max(len(predictions), max_labels)

    logging.info("Maximum labels = %d", max_labels)
    return all_examples


def get_fever_title_to_annotation_ids(file_path):
    """Return mapping of doc title --> annotation_ids of queries that refer to it."""
    title2qids = collections.defaultdict(list)
    for case in jsonlines.Reader(open(file_path, "r")):
        # This dataset has only one gold doc per query.
        title = case["docids"][0]
        qid = case["annotation_id"]
        if qid not in title2qids[title]:
            title2qids[title].append(qid)

    return title2qids


def create_trials(title_splits, num_trials, percent=0.8):
    """Create splits for trials, stratified by article."""
    title_splits = list(title_splits.values())
    calibration_qids = []
    test_qids = []

    # Repeat the experiment N times.
    for _ in tqdm(range(num_trials)):
        np.random.shuffle(title_splits)
        split = int(percent * len(title_splits))
        calibration_qids.append([])
        for qids in title_splits[:split]:
            calibration_qids[-1].extend(qids)

        test_qids.append([])
        for qids in title_splits[split:]:
            test_qids[-1].extend(qids)

    return calibration_qids, test_qids


def get_fever_gold_answers(file_path):
    """Return mapping of query ids to correct text retrievals."""
    qid2golds = {}
    for case in jsonlines.Reader(open(file_path, "r")):
        qid = case["annotation_id"]

        texts = []
        for ev_group in case["evidences"]:
            for ev in ev_group:
                texts.append(ev["text"])

        qid2golds[qid] = texts

    return qid2golds


def get_metric(label, metric):
    """Select nonconformal score for label. Lower is better."""
    if metric == "rank":
        score = label["rank"]
    elif metric == "bm25":
        score = -1 * label["sent_bm25"]
    elif metric == "tfidf":
        score = -1 * label["sent_tfidf"]
    elif metric == "prob":
        score = 1 - label["sent_prob"]
    elif metric == "logit":
        score = -1 * label["sent_logit"]
    else:
        raise ValueError("Unrecognized metric: %s" % metric)
    return score


def main(_):
    logging.set_verbosity(logging.INFO)
    np.random.seed(FLAGS.seed)

    FLAGS.epsilons = [float(f) for f in FLAGS.epsilons or []]

    logging.info("Loading data.")
    all_examples = load_predictions(FLAGS.dev_predictions_file, FLAGS.metrics)
    qid2answers = get_fever_gold_answers(FLAGS.dev_gold_file)

    if FLAGS.test_predictions_file:
        logging.info("Combining dev with data from test file.")
        test_examples = load_predictions(FLAGS.test_predictions_file, FLAGS.metrics)
        all_examples.update(test_examples)

        qid2answers_test = get_fever_gold_answers(FLAGS.test_gold_file)
        qid2answers.update(qid2answers_test)

    if not FLAGS.trials_file:
        FLAGS.trials_file = os.path.splitext(FLAGS.dev_gold_file)[0]
        FLAGS.trials_file += "-trials=%d-wtest=%d.json" % (FLAGS.num_trials,
                                                           FLAGS.test_predictions_file is not None)
    os.makedirs(os.path.dirname(FLAGS.trials_file), exist_ok=True)
    if os.path.exists(FLAGS.trials_file) and not FLAGS.overwrite_trials:
        logging.info("Loading trials from %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "r") as f:
            trial_qids = json.load(f)
    else:
        title2qids = get_fever_title_to_annotation_ids(FLAGS.dev_gold_file)
        if FLAGS.test_predictions_file:
            title2qids.update(get_fever_title_to_annotation_ids(FLAGS.test_gold_file))
        trial_qids = create_trials(title2qids, FLAGS.num_trials)
        logging.info("Writing trials to %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "w") as f:
            json.dump(trial_qids, f)

    calibration_qids, test_qids = trial_qids
    if FLAGS.skip_conformal:
        suffix = "/ir/baselines-trials=%s/" % FLAGS.num_trials
    else:
        suffix = ("/ir/trials=%s-smoothed=%s-correction=%s-metrics=%s"
                  "-equivalence=%s/" %
                  (FLAGS.num_trials, FLAGS.smoothed, FLAGS.correction,
                   ",".join(FLAGS.metrics), FLAGS.equivalence))
    FLAGS.output_dir += suffix
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    run_experiment(
        all_examples=all_examples,
        qid2answers=qid2answers,
        calibration_qids=calibration_qids,
        test_qids=test_qids,
        baseline_metrics=["sent_prob", "sent_bm25"],
        epsilons=FLAGS.epsilons,
        efficiencies=FLAGS.efficiencies,
        get_metric=get_metric)


if __name__ == "__main__":
    app.run(main)
