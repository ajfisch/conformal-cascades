"""Baseline methods (nonconformal)."""

import collections
import functools
import multiprocessing.pool
import tqdm

from absl import flags
from absl import logging
import numpy as np
import scipy.interpolate
import scipy.special

from cpcascade import utils

EPS = 1e-18

FLAGS = flags.FLAGS

MAX_THRESHOLDS = 5000

BaselineResult = collections.namedtuple(
    "BaselineResult",
    ["scores", "thresholds"])


def _evaluate_baseline_trial(examples, baseline_class, target_epsilons, threads=0, cuda=False):
    """Run inner evaluation loop."""
    calibration_examples = examples[0]
    calibration_answers = examples[1]
    calibration_mask = examples[2]
    test_examples = examples[3]
    test_answers = examples[4]
    test_mask = examples[5]

    # Initialize baseline.
    baseline = baseline_class()

    # Only use multiprocessing with threads > 0.
    worker_fn = baseline.predict
    if threads > 0:
        workers = multiprocessing.pool.Pool(processes=threads)
        map_fn = workers.imap
    else:
        map_fn = map

    # ***************************
    # CALIBRATION.
    # ***************************

    # Initialize output matrices (for calibration).
    num_examples, max_labels = calibration_examples.shape[:2]
    threshold_matrix = -np.ones((num_examples, max_labels, 1)) * float("inf")

    # Populate threshold and accuracy matrices.
    logging.debug("Populating threshold matrices")
    for i, result in enumerate(map_fn(worker_fn, zip(calibration_examples, calibration_answers))):
        for j in range(len(result.thresholds)):
            threshold_matrix[i, j] = result.thresholds[j]

    # Evaluate all possible thresholds. Limit number.
    viable_thresholds = np.unique(threshold_matrix - EPS)
    np.random.shuffle(viable_thresholds)
    viable_thresholds = viable_thresholds[:MAX_THRESHOLDS]
    thresholds = sorted(viable_thresholds.tolist() + [np.max(threshold_matrix) + EPS])
    logging.debug("Evaluating threshold scores over %s values.", len(thresholds))
    if cuda:
        thresholds = utils.evaluate_thresholds_cuda(
            thresholds=thresholds,
            threshold_matrix=threshold_matrix,
            scores_matrix=calibration_answers,
            label_mask=calibration_mask)
    else:
        thresholds = utils.evaluate_thresholds(
            thresholds=thresholds,
            threshold_matrix=threshold_matrix,
            scores_matrix=calibration_answers,
            label_mask=calibration_mask,
            threads=threads)

    # Find the thresholds for the desired accuracies. As accuracy is monotic in threshold value,
    # max{t: acc(t) > 1 - epsilon} = t of the next highest accuracy not equal to 1 - epsilon.
    thresholds = sorted(thresholds, key=lambda x: x[2])
    points = list(reversed(np.arange(0, 1.0001, 0.0001).tolist()))
    f = scipy.interpolate.interp1d([e[2] for e in thresholds], [e[0] for e in thresholds], kind="next",
                                   bounds_error=False, fill_value="extrapolate")
    calibration_thresholds_to_use = f(points)

    # ***************************
    # TESTING.
    # ***************************

    # Initialize output matrices (for test).
    num_examples, max_labels = test_examples.shape[:2]
    threshold_matrix = -np.ones((num_examples, max_labels, 1)) * float("inf")

    logging.debug("Populating threshold matrices for test answers.")
    for i, result in enumerate(map_fn(worker_fn, zip(test_examples, test_answers))):
        for j in range(len(result.thresholds)):
            threshold_matrix[i, j, 0] = result.thresholds[j]

    # Evaluate all possible thresholds.
    logging.debug("Evaluating threshold scores on test answers.")
    if FLAGS.cuda:
        thresholds = utils.evaluate_thresholds_cuda(
            thresholds=calibration_thresholds_to_use,
            threshold_matrix=threshold_matrix,
            scores_matrix=test_answers,
            label_mask=test_mask)
    else:
        thresholds = utils.evaluate_thresholds(
            thresholds=calibration_thresholds_to_use,
            threshold_matrix=threshold_matrix,
            scores_matrix=test_answers,
            label_mask=test_mask)

    epsilons = []
    for i, (_, efficiency, accuracy, _) in enumerate(thresholds):
        epsilons.append((1 - points[i], efficiency, accuracy))

    # Collect results.
    trial_results = {}
    trial_results["values"] = epsilons

    # Find target values of epsilon (among unique values).
    results = []
    for target in target_epsilons:
        index = np.argmin(np.abs(target - np.array([eps[0] for eps in epsilons])))
        results.append(epsilons[index])
    trial_results["epsilon"] = results

    return trial_results


def baseline_evaluation(
    examples,
    answers,
    mask,
    calibration_ids,
    test_ids,
    target_epsilons,
    baseline_class,
    references=None,
    inner_threads=None,
    outer_threads=None,
):
    """Compute all baseline metrics.

    Args:
      examples: <float> [num_examples, max_labels, num_metrics]
        Array of label metrics.
      mask: <float> [num_examples, max_labels]
        Indicator for padding or not padding.
      answers: <float> [num_examples, max_labels]
        Indicator of acceptable/not acceptable labels.
      calibration_ids: <list>
        List of ids just for calibration.
      test_ids: <list>
        List of ids just for testing.
      target_epsilons: <list>
        List of target epsilons.
      baseline_class: <class>
        Return baseline predictor function.
      references: <int>[num_examples]
        Indices to use as references.
      inner_threads: <int>
        Number of threads to use during multiprocessing inner loop.
      outer_threads: <int>
        Number of threads to use during multiprocessing outer loop.

    Returns:
      results: Dict of efficiency and accuracy results.
    """
    outer_threads = outer_threads if outer_threads is not None else FLAGS.outer_threads
    inner_threads = inner_threads if inner_threads is not None else FLAGS.inner_threads

    # Store results for all trials.
    trial_results = {"epsilon": []}

    # Store all accuracies per efficiency.
    all_epsilons = collections.defaultdict(list)

    # Only use multiprocessing with threads > 0.
    if outer_threads > 0:
        if inner_threads > 0:
            workers = multiprocessing.pool.ThreadPool(processes=outer_threads)
        else:
            workers = multiprocessing.pool.Pool(processes=outer_threads)
        map_fn = workers.imap
    else:
        map_fn = map

    worker_fn = functools.partial(
        _evaluate_baseline_trial,
        baseline_class=baseline_class,
        target_epsilons=target_epsilons,
        threads=inner_threads,
        cuda=FLAGS.cuda)

    # Gather and evaluate all trials.
    all_trials = []
    for i in range(len(calibration_ids)):
        calibration_examples = examples[calibration_ids[i]]
        calibration_mask = mask[calibration_ids[i]]
        test_examples = examples[test_ids[i]]
        test_answers = answers[test_ids[i]]
        test_mask = mask[test_ids[i]]

        calibration_references = references[calibration_ids[i]]
        calibration_answers = np.zeros_like(calibration_examples)
        calibration_answers[np.arange(len(calibration_references)), calibration_references] = 1

        all_trials.append((calibration_examples,
                           calibration_answers,
                           calibration_mask,
                           test_examples,
                           test_answers,
                           test_mask))

    # Run.
    with tqdm.tqdm(total=len(all_trials), desc="evaluating") as pbar:
        for result in map_fn(worker_fn, all_trials):
            trial_results["epsilon"].append(result["epsilon"])
            for epsilon, efficiency, accuracy in result["values"]:
                all_epsilons[epsilon].append((efficiency, accuracy))
            pbar.update()

    # Average results over all trials.
    avg_all_epsilons = []
    for epsilon, trials in all_epsilons.items():
        efficiencies = utils.stats([trial[0] for trial in trials])
        accuracies = utils.stats([trial[1] for trial in trials])
        avg_all_epsilons.append((epsilon, efficiencies, accuracies))

    avg_target_epsilons = []
    for i in range(len(target_epsilons)):
        epsilon = np.mean([trial[i][0] for trial in trial_results["epsilon"]])
        efficiencies = utils.stats([trial[i][1] for trial in trial_results["epsilon"]])
        accuracies = utils.stats([trial[i][2] for trial in trial_results["epsilon"]])
        avg_target_epsilons.append(dict(
            epsilon=epsilon,
            efficiency=efficiencies,
            accuracy=accuracies))

    results = dict(targets=avg_target_epsilons,
                   data=avg_all_epsilons)

    return results


class TopK(object):
    """Use the top-k ranked predictions."""
    name = "top_k"

    def predict(self, example):
        labels, answers = example
        ranks = np.argsort(labels)
        thresholds = -ranks

        # Hypothetically would take top K here, but unncessary for any of our evaluations.

        return BaselineResult(answers, thresholds)


class Threshold(object):
    """Use a simple threshold for any metric."""
    name = "threshold"

    def predict(self, example):
        labels, answers = example

        # Flip, because we assume higher is better.
        labels = -labels

        # Hypothetically would take any label with score > thresh here, but unnecessary for any
        # of our evaluations.

        return BaselineResult(answers, labels)
