"""Evaluation for Natural Questions."""

import collections
import json
import tqdm
import os
import gc
import pickle
import sys

from absl import app
from absl import flags
from absl import logging

from transformers.data.metrics import squad_metrics
import numpy as np

from cpcascade import run_experiment

DEFAULT_EVAL_FILE = None

flags.DEFINE_string("eval_file", DEFAULT_EVAL_FILE,
                    "Path to nbest file (output of modified DPR script).")

flags.DEFINE_integer("num_trials", 50, "Number of trials to run.")

flags.DEFINE_string("trials_file", None,
                    "The output file where the trials qids will be written.")

flags.DEFINE_boolean("overwrite_trials", False,
                     "Recompute and overwrite the trials qids if they exist.")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to evaluate.")

flags.DEFINE_list("metrics", "sum",
                  "The conformal scores to use (comma separated).")

flags.DEFINE_integer("limit_docs", None, "Limit maximum retrieved docs per question.")

flags.DEFINE_integer("limit_labels", 50, "Limit maximum labels per retrieved doc per question.")

flags.DEFINE_boolean("cache_data", True,
                     "pickle the processed data.")

flags.DEFINE_boolean("debug", False,
                     "Output debug logs.")

FLAGS = flags.FLAGS


def load_nbest(file_path, metrics, limit_docs=None, limit_labels=None):
    """Load (and clean) predictions from the NQ file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    references = {}
    all_examples = {}
    q2golds = {}
    max_labels = 0
    for example in tqdm.tqdm(data):
        labels_per_doc = collections.Counter()
        qid = example["question"]
        q2golds[qid] = [squad_metrics.normalize_answer(e) for e in example["gold_answers"]]
        nbest = example["predictions"]
        labels = sorted(nbest, key=lambda y: -y["score"])

        correct_idx = []
        filtered_labels = []
        rank = 0
        for y in labels:
            if limit_docs is not None and y["passage_idx"] >= limit_docs:
                continue
            if limit_labels is not None and labels_per_doc[y["passage_idx"]] >= limit_labels:
                continue

            y["text"] = squad_metrics.normalize_answer(y["text"])
            y["rank"] = rank
            if y["text"] in q2golds[qid]:
                correct_idx.append(rank)

            labels_per_doc[y["passage_idx"]] += 1
            rank += 1
            filtered_labels.append(y)

        if correct_idx:
            all_examples[qid] = filtered_labels
            max_labels = max(len(filtered_labels), max_labels)
            references[qid] = np.random.choice(correct_idx).item()

    gc.collect()
    logging.info("Filtered %d with no answer" % (len(data) - len(all_examples.keys())))
    logging.info("Maximum labels = %d, Num examples = %d", max_labels, len(all_examples.keys()))
    return all_examples, q2golds, references


def create_trials(questions, num_trials, percent=0.8):
    """Create splits for trials, randomly over all questions."""
    calibration_qids = []
    test_qids = []

    # Repeat the experiment N times.
    for _ in tqdm.tqdm(range(num_trials)):
        np.random.shuffle(questions)
        split = int(percent * len(questions))
        calibration_qids.append(questions[:split])
        test_qids.append(questions[split:])

    return calibration_qids, test_qids


def get_metric(label, metric):
    """Select nonconformal score for label. Lower is better."""
    if metric == "rank":
        score = label["rank"]
    elif metric == "start_logit":
        score = -1 * label["start_score"]
    elif metric == "end_logit":
        score = -1 * label["end_score"]
    elif metric == "sum":
        score = -1 * label["score"]
    elif metric == "relevance_plus_score":
        score = -1 * (label["score"] + label["relevance_score"])
    elif metric == "relevance_logit":
        score = -1 * label["relevance_score"]
    elif metric == "psg_rank":
        score = label["passage_idx"]
    elif metric == "psg_score":
        score = -1 * label["passage_score"]
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
        suffix = "/open_qa/baselines-trials=%s/" % FLAGS.num_trials
    else:
        suffix = ("/open_qa/trials=%s-docs=%s-labels=%s-smoothed=%s-correction=%s-metrics=%s"
                  "-equivalence=%s/" %
                  (FLAGS.num_trials, FLAGS.limit_docs, FLAGS.limit_labels, FLAGS.smoothed, FLAGS.correction,
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

    cache_path = os.path.splitext(FLAGS.eval_file)[0]
    cache_path += "-docs=%s-labels=%s.pkl" % (FLAGS.limit_docs, FLAGS.limit_labels)
    if os.path.exists(cache_path) and FLAGS.cache_data:
        logging.info("Loading data from %s", cache_path)
        with open(cache_path, 'rb') as f:
            all_examples, qid2answers, references = pickle.load(f)
        logging.info("Num examples = %d", len(all_examples.keys()))
    else:
        all_examples, qid2answers, references = load_nbest(
            FLAGS.eval_file, FLAGS.metrics, FLAGS.limit_docs, FLAGS.limit_labels)
        if FLAGS.cache_data:
            logging.info("Writing data to %s", cache_path)
            with open(cache_path, 'wb') as f:
                pickle.dump((all_examples, qid2answers, references), f)

    if not FLAGS.trials_file:
        FLAGS.trials_file = os.path.splitext(FLAGS.eval_file)[0]
        FLAGS.trials_file += ("-trials=%d-docs=%s-labels=%s.json" %
                              (FLAGS.num_trials, FLAGS.limit_docs, FLAGS.limit_labels))
    os.makedirs(os.path.dirname(FLAGS.trials_file), exist_ok=True)
    if os.path.exists(FLAGS.trials_file) and not FLAGS.overwrite_trials:
        logging.info("Loading trials from %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "r") as f:
            trial_qids = json.load(f)
    else:
        questions = list(all_examples.keys())
        trial_qids = create_trials(questions, FLAGS.num_trials)
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
        mask=label_mask,
        references=references,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
        baseline_metrics=["sum"],
        epsilons=FLAGS.epsilons,
        )


if __name__ == "__main__":
    app.run(main)
