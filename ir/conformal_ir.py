"""Evaluation for FEVER IR."""

import json
import os
import sys
import jsonlines
from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging

import numpy as np
from cpcascade import run_experiment

DEFAULT_GOLD_FILE = "data/ir/test.jsonl"
DEFAULT_PRED_FILE = "data/ir/preds_test.jsonl"

flags.DEFINE_string("eval_file", DEFAULT_PRED_FILE,
                    "Path to retriever's predictions.")

flags.DEFINE_string("gold_file", DEFAULT_GOLD_FILE,
                    "File with gold answers.")

flags.DEFINE_integer("num_trials", 40, "Number of trials to run.")

flags.DEFINE_string("trials_file", None,
                    "The output file where the trials qids will be written.")

flags.DEFINE_boolean("overwrite_trials", False,
                     "Recompute and overwrite the trials qids if they exist.")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to evaluate.")

flags.DEFINE_list("metrics", "rank",
                  "The conformal scores to use (comma separated).")

flags.DEFINE_boolean("debug", False,
                     "Output debug logs.")

FLAGS = flags.FLAGS


def load_predictions(file_path, qid2answers, metrics):
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

    # We choose one label that's annotated as being correct as the reference.
    references = {}
    filtered = {}
    for qid, predictions in all_examples.items():
        correct_idx = [i for i, y in enumerate(predictions) if y["text"] in qid2answers[qid]]
        if len(correct_idx) == 0:
            continue
        filtered[qid] = predictions
        references[qid] = np.random.choice(correct_idx).item()

    logging.info("Filtered %d with no answer" % (len(all_examples) - len(filtered)))
    all_examples = filtered

    max_labels = 0
    for qid, predictions in all_examples.items():
        sorted_predictions_sent = sorted(predictions, key=lambda y: -y["sent_bm25"])
        for rank, y in enumerate(sorted_predictions_sent):
            y["rank"] = rank

        all_examples[qid] = sorted_predictions_sent
        max_labels = max(len(predictions), max_labels)

    logging.info("Maximum labels = %d", max_labels)
    return all_examples, references


def create_trials(qids, num_trials, percent=0.8):
    """Create splits for trials, stratified by article."""
    calibration_qids = []
    test_qids = []

    # Repeat the experiment N times.
    for _ in tqdm(range(num_trials)):
        np.random.shuffle(qids)
        split = int(percent * len(qids))
        calibration_qids.append(qids[:split])
        test_qids.append(qids[split:])

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
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    np.random.seed(FLAGS.seed)

    if FLAGS.skip_conformal:
        suffix = "/ir/baselines-trials=%s/" % FLAGS.num_trials
    else:
        suffix = ("/ir/trials=%s-smoothed=%s-correction=%s-metrics=%s"
                  "-equivalence=%s/" %
                  (FLAGS.num_trials, FLAGS.smoothed, FLAGS.correction,
                   ",".join(FLAGS.metrics), FLAGS.equivalence))

    FLAGS.output_dir += suffix
    if (not FLAGS.overwrite_output
        and os.path.exists(FLAGS.output_dir)
        and os.path.exists(FLAGS.output_dir + 'results.json')):
        logging.info("Output path exists: '%s'", FLAGS.output_dir)
        sys.exit(0)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    FLAGS.epsilons = [float(f) for f in FLAGS.epsilons or []]

    logging.info("Loading data.")
    qid2answers = get_fever_gold_answers(FLAGS.gold_file)
    all_examples, references = load_predictions(FLAGS.eval_file, qid2answers, FLAGS.metrics)

    if not FLAGS.trials_file:
        FLAGS.trials_file = os.path.splitext(FLAGS.gold_file)[0]
        FLAGS.trials_file += "-trials=%d.json" % FLAGS.num_trials

    os.makedirs(os.path.dirname(FLAGS.trials_file), exist_ok=True)
    if os.path.exists(FLAGS.trials_file) and not FLAGS.overwrite_trials:
        logging.info("Loading trials from %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "r") as f:
            trial_qids = json.load(f)
    else:
        trial_qids = create_trials(list(all_examples.keys()), FLAGS.num_trials)
        logging.info("Writing trials to %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "w") as f:
            json.dump(trial_qids, f)

    calibration_qids, test_qids = trial_qids

    # Make inputs.
    num_examples = len(all_examples)
    max_labels = max([len(v) for v in all_examples.values()])
    num_metrics = len(FLAGS.metrics)
    example_matrix = np.ones((num_examples, max_labels, num_metrics)) * 1e12
    answer_matrix = np.zeros((num_examples, max_labels))
    label_mask = np.zeros((num_examples, max_labels))

    qid2index = {}
    index2qid = []
    for i, qid in enumerate(all_examples.keys()):
        index2qid.append(qid)
        qid2index[qid] = i

    calibration_ids = [[qid2index[qid] for qid in trial] for trial in calibration_qids]
    test_ids = [[qid2index[qid] for qid in trial] for trial in test_qids]

    references = np.array([references[qid] for qid in index2qid]).astype(np.long)
    for i, qid in enumerate(index2qid):
        labels = all_examples[qid]
        for j, y in enumerate(labels):
            for k, m in enumerate(FLAGS.metrics):
                example_matrix[i, j, k] = get_metric(y, m)
            if y["text"] in qid2answers[qid]:
                answer_matrix[i, j] = 1
            label_mask[i, j] = 1

    run_experiment(
        examples=example_matrix,
        answers=answer_matrix,
        references=references,
        mask=label_mask,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
        baseline_metrics=["bm25", "logit"],
        epsilons=FLAGS.epsilons)


if __name__ == "__main__":
    app.run(main)
