"""Evaluation for ChEMBL molecule retrieval."""

import json
import tqdm
import os
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np

from cpcascade import run_experiment

DEFAULT_EVAL_FILE = None

flags.DEFINE_string("eval_dir", DEFAULT_EVAL_FILE,
                    "Path to dir with npy files.")

flags.DEFINE_integer("num_trials", 50, "Number of trials to run.")

flags.DEFINE_string("trials_file", None,
                    "The output file where the trials qids will be written.")

flags.DEFINE_boolean("overwrite_trials", False,
                     "Recompute and overwrite the trials qids if they exist.")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to evaluate.")

flags.DEFINE_list("metrics", "MPN",
                  "The conformal scores to use (comma separated).")

flags.DEFINE_boolean("debug", False, "Output debug logs.")

FLAGS = flags.FLAGS


def load_data(dir_path, metrics):
    """Load from npy files."""

    # examples is stored with two metrics in this order.
    stored_metrics = ["RF", "MPN"]

    full_example_matrix = np.load(os.path.join(dir_path, "examples.npy"))
    answer_matrix = np.load(os.path.join(dir_path, "answers.npy"))
    label_mask = np.load(os.path.join(dir_path, "mask.npy"))
    references = np.load(os.path.join(dir_path, "references.npy")).astype(np.long)

    metric_inds = [stored_metrics.index(m) for m in metrics]
    example_matrix = full_example_matrix[:, :, metric_inds]

    logging.info("Maximum labels = %d, Num examples = %d",
                 np.max(label_mask.sum(1)), answer_matrix.shape[0])
    return example_matrix, answer_matrix, label_mask, references


def create_trials(combinations, num_trials, percent=0.8):
    """Create splits for trials, randomly over all combinations."""
    calibration_ids = []
    test_ids = []

    # Repeat the experiment N times.
    for _ in tqdm.tqdm(range(num_trials)):
        np.random.shuffle(combinations)
        split = int(percent * len(combinations))
        calibration_ids.append(combinations[:split])
        test_ids.append(combinations[split:])

    return calibration_ids, test_ids


def main(_):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    np.random.seed(FLAGS.seed)

    if FLAGS.skip_conformal:
        suffix = "/chembl/baselines-trials=%s/" % FLAGS.num_trials
    else:
        suffix = (
            "/chembl/trials=%s-smoothed=%s-correction=%s-metrics=%s-equivalence=%s/"
            % (FLAGS.num_trials, FLAGS.smoothed, FLAGS.correction, ",".join(
                FLAGS.metrics), FLAGS.equivalence))
    FLAGS.output_dir += suffix
    if (not FLAGS.overwrite_output and os.path.exists(FLAGS.output_dir)
            and os.path.exists(FLAGS.output_dir + 'results.json')):
        logging.info("Output path exists: '%s'", FLAGS.output_dir)
        sys.exit(0)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    FLAGS.epsilons = [float(f) for f in FLAGS.epsilons or []]

    # Load inputs.
    logging.info("Loading data from %s.", FLAGS.eval_dir)
    example_matrix, answer_matrix, label_mask, references = load_data(
        FLAGS.eval_dir, FLAGS.metrics)

    if not FLAGS.trials_file:
        FLAGS.trials_file = FLAGS.eval_dir
        FLAGS.trials_file += ("-trials=%d.json" % FLAGS.num_trials)
    os.makedirs(os.path.dirname(FLAGS.trials_file), exist_ok=True)
    if os.path.exists(FLAGS.trials_file) and not FLAGS.overwrite_trials:
        logging.info("Loading trials from %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "r") as f:
            trial_ids = json.load(f)
    else:
        combinations = list(range(len(example_matrix)))
        trial_ids = create_trials(combinations, FLAGS.num_trials)
        logging.info("Writing trials to %s", FLAGS.trials_file)
        with open(FLAGS.trials_file, "w") as f:
            json.dump(trial_ids, f)

    calibration_ids, test_ids = trial_ids

    run_experiment(
        examples=example_matrix,
        answers=answer_matrix,
        mask=label_mask,
        references=references,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
        baseline_metrics=["MPN"],
        epsilons=FLAGS.epsilons,
    )


if __name__ == "__main__":
    app.run(main)
