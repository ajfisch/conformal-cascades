"""Evaluation for SQuAD."""

import collections
import json
import tqdm
import os

from absl import app
from absl import flags
from absl import logging

from transformers.data.metrics import squad_metrics
import numpy as np
import scipy.special

from cpcascade import run_experiment

DEFAULT_DEV_FILE = None
DEFAULT_EVAL_FILE = None

flags.DEFINE_string("eval_file", DEFAULT_EVAL_FILE,
                    "Path to nbest file (output of QA eval script).")

flags.DEFINE_integer("num_trials", 50, "Number of trials to run.")

flags.DEFINE_string("trials_file", None,
                    "The output file where the trials qids will be written.")

flags.DEFINE_boolean("overwrite_trials", False,
                     "Recompute and overwrite the trials qids if they exist.")

flags.DEFINE_string("gold_file", DEFAULT_DEV_FILE,
                    "Dev set file with gold answers.")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to evaluate.")

flags.DEFINE_list("metrics", "sum",
                  "The conformal scores to use (comma separated).")

FLAGS = flags.FLAGS


def load_nbest(file_path, metrics, qid2answers):
    """Load (and clean) nbest file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    all_examples = {}
    max_labels = 0
    for qid, nbest in data.items():
        # Skip if not in qid2answers.
        if qid not in qid2answers:
            continue

        # Resolve spans from multiple overlapping windows.
        text2label = {}
        for label in nbest:
            text = squad_metrics.normalize_answer(label["text"])
            label["text"] = text
            if text in text2label:
                if label["rerank_logit"] > text2label[text]["rerank_logit"]:
                    text2label[text] = label
            else:
                text2label[text] = label

        # Deduplicated
        labels = sorted(text2label.values(), key=lambda y: -y["probability"])

        # Skip if answer is not here.
        if not any([label["text"] in qid2answers[qid] for label in labels]):
            logging.info("Skipping example with no answer")
            continue

        # Renormalize probability over top-k and add rank.
        total_p = sum([y["probability"] for y in labels])
        start_p = scipy.special.softmax([y["start_logit"] for y in labels])
        end_p = scipy.special.softmax([y["end_logit"] for y in labels])
        for rank, y in enumerate(labels):
            y["probability"] /= total_p
            y["rank"] = rank
            y["sum"] = y["start_logit"] + y["end_logit"]
            y["start_prob"] = start_p[rank]
            y["end_prob"] = end_p[rank]
        all_examples[qid] = labels
        max_labels = max(len(labels), max_labels)

    logging.info("Maximum labels = %d", max_labels)
    return all_examples


def get_squad_title_to_qids(file_path):
    """Return mapping of title --> article question ids."""
    title2qids = collections.defaultdict(list)
    with open(file_path, "r") as f:
        data = json.load(f)["data"]
    for article in data:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                qid = question["id"]
                title2qids[title].append(qid)
    return title2qids


def create_trials(title_splits, num_trials, percent=0.8):
    """Create splits for trials, stratified by article."""
    title_splits = list(title_splits.values())
    calibration_qids = []
    test_qids = []

    # Repeat the experiment N times.
    for _ in tqdm.tqdm(range(num_trials)):
        np.random.shuffle(title_splits)
        split = int(percent * len(title_splits))
        calibration_qids.append([])
        for qids in title_splits[:split]:
            calibration_qids[-1].extend(qids)

        test_qids.append([])
        for qids in title_splits[split:]:
            test_qids[-1].extend(qids)

    return calibration_qids, test_qids


def get_squad_gold_answers(file_path):
    """Return mapping of question ids to normalized answers."""
    qid2answers = {}
    with open(file_path, "r") as f:
        data = json.load(f)["data"]
    for article in data:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                qid = question["id"]
                answers = set([squad_metrics.normalize_answer(a["text"])
                               for a in question["answers"]])
                if question.get("is_impossible"):
                    answers.add("")
                qid2answers[qid] = answers
    return qid2answers


def get_metric(label, metric):
    """Select nonconformal score for label. Lower is better."""
    if metric == "rank":
        score = label["rank"]
    elif metric == "prob":
        score = 1 - label["probability"]
    elif metric == "start_prob":
        score = 1 - label["start_prob"]
    elif metric == "start_logit":
        score = -1 * label["start_logit"]
    elif metric == "end_prob":
        score = 1 - label["end_prob"]
    elif metric == "end_logit":
        score = -1 * label["end_logit"]
    elif metric == "sum":
        score = -1 * (label["start_logit"] + label["end_logit"])
    elif metric == "rerank_prob":
        score = 1 - label["rerank_probability"]
    elif metric == "rerank_logit":
        score = -1 * label["rerank_logit"]
    elif metric == "rerank_sigmoid":
        score = 1 - label["rerank_sigmoid"]
    else:
        raise ValueError("Unrecognized metric: %s" % metric)
    return score


def main(_):
    logging.set_verbosity(logging.INFO)
    np.random.seed(FLAGS.seed)

    FLAGS.epsilons = [float(f) for f in FLAGS.epsilons or []]

    logging.info("Loading data.")

    qid2answers = get_squad_gold_answers(FLAGS.gold_file)
    all_examples = load_nbest(FLAGS.eval_file, FLAGS.metrics, qid2answers)

    if not FLAGS.trials_file:
        FLAGS.trials_file = os.path.splitext(FLAGS.gold_file)[0]
        FLAGS.trials_file += "-trials=%d.json" % FLAGS.num_trials
    os.makedirs(os.path.dirname(FLAGS.trials_file), exist_ok=True)
    if os.path.exists(FLAGS.trials_file) and not FLAGS.overwrite_trials:
        logging.info("Loading trials from %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "r") as f:
            trial_qids = json.load(f)
    else:
        title2qids = get_squad_title_to_qids(FLAGS.gold_file)
        title2qids = {k: [qid for qid in v if qid in all_examples]
                      for k, v in title2qids.items()}
        trial_qids = create_trials(title2qids, FLAGS.num_trials)
        logging.info("Writing trials to %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "w") as f:
            json.dump(trial_qids, f)

    calibration_qids, test_qids = trial_qids
    if FLAGS.skip_conformal:
        suffix = "/qa/baselines-trials=%s/" % FLAGS.num_trials
    else:
        suffix = ("/qa/trials=%s-smoothed=%s-correction=%s-metrics=%s"
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
        baseline_metrics=["sum", "rerank_logit"],
        epsilons=FLAGS.epsilons,
        efficiencies=FLAGS.efficiencies,
        get_metric=get_metric)


if __name__ == "__main__":
    app.run(main)
