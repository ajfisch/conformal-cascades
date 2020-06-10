"""Baseline methods (nonconformal)."""

import collections
import functools
import multiprocessing
import tqdm

from absl import flags
import numpy as np
import scipy.interpolate
import scipy.special

from cpcascade import utils

EPS = 1e-18

FLAGS = flags.FLAGS

MAX_THRESHOLDS = 10000

BaselineResult = collections.namedtuple(
    "BaselineResult",
    ["qid", "scores", "thresholds"])


def _evaluate_baseline_trial(examples, baseline_class, target_epsilons):
    """Run inner evaluation loop."""
    global qid2answers
    calibration_examples, test_examples = examples

    # Initialize baseline.
    baseline = baseline_class(qid2answers=qid2answers)

    # ***************************
    # CALIBRATION.
    # ***************************

    # Initialize output matrices (for calibration).
    num_examples = len(calibration_examples)
    max_labels = max([len(ex.labels) for ex in calibration_examples])
    label_mask = np.zeros((num_examples, max_labels))
    threshold_matrix = -np.ones((num_examples, max_labels, 1)) * float("inf")
    scores_matrix = np.zeros((num_examples, max_labels))

    # Populate threshold and accuracy matrices.
    worker_fn = baseline.predict
    for i, result in enumerate(map(worker_fn, calibration_examples)):
        for j in range(len(result.thresholds)):
            label_mask[i, j] = 1
            threshold_matrix[i, j, 0] = result.thresholds[j]
            scores_matrix[i, j] = result.scores[j]

    # Evaluate all possible thresholds. Limit number.
    viable_thresholds = np.unique(threshold_matrix - EPS)
    np.random.shuffle(viable_thresholds)
    viable_thresholds = viable_thresholds[:MAX_THRESHOLDS]
    thresholds = utils.evaluate_thresholds(
        thresholds=sorted(viable_thresholds.tolist() + [np.max(threshold_matrix) + EPS]),
        threshold_matrix=threshold_matrix,
        scores_matrix=scores_matrix,
        label_mask=label_mask)

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
    num_examples = len(test_examples)
    max_labels = max([len(ex.labels) for ex in test_examples])
    label_mask = np.zeros((num_examples, max_labels))
    threshold_matrix = -np.ones((num_examples, max_labels, 1)) * float("inf")
    scores_matrix = np.zeros((num_examples, max_labels))

    # Populate threshold and accuracy matrices.
    worker_fn = baseline.predict
    for i, result in enumerate(map(worker_fn, test_examples)):
        for j in range(len(result.thresholds)):
            label_mask[i, j] = 1
            threshold_matrix[i, j, 0] = result.thresholds[j]
            scores_matrix[i, j] = result.scores[j]

    # Evaluate all possible thresholds.
    thresholds = utils.evaluate_thresholds(
        thresholds=calibration_thresholds_to_use,
        threshold_matrix=threshold_matrix,
        scores_matrix=scores_matrix,
        label_mask=label_mask)

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


def _worker_init_fn(_qid2answers):
    """Initialize workers with a copy of answers."""
    global qid2answers
    qid2answers = _qid2answers


def baseline_evaluation(
    target_epsilons,
    examples,
    qid2answers,
    baseline_class,
    calibration_qids=None,
    test_qids=None,
    threads=None,
):
    """Compute all baseline metrics.

    Args:
      target_epsilons: <list>
        List of target epsilons.
      examples: <dict>
        Dict of all qids to labels to evaluate.
      qid2answers: <dict>
        Map of qid to answer set.
      baseline_class: <class>
        Return baseline predictor function.
      calibration_qids: <list>
        List of qids just for calibration.
      test_qids: <list>
        List of qids just for testing.
      threads: <int>
        Number of threads to use during multiprocessing.

    Returns:
      results: Dict of efficiency and accuracy results.
    """
    threads = threads if threads is not None else FLAGS.threads

    # Store results for all trials.
    trial_results = {"epsilon": []}

    # Store all accuracies per efficiency.
    all_epsilons = collections.defaultdict(list)

    # Only use multiprocessing with threads > 0.
    if threads > 0:
        workers = multiprocessing.Pool(
            processes=threads,
            initializer=_worker_init_fn,
            initargs=(qid2answers,))
        map_fn = workers.imap_unordered
    else:
        _worker_init_fn(qid2answers)
        map_fn = map

    worker_fn = functools.partial(
        _evaluate_baseline_trial,
        baseline_class=baseline_class,
        target_epsilons=target_epsilons)

    # Gather and evaluate all trials.
    all_trials = []
    for i in range(len(calibration_qids)):
        calibration_examples = [utils.Example(qid, examples[qid]) for qid in calibration_qids[i]]
        test_examples = [utils.Example(qid, examples[qid]) for qid in test_qids[i]]
        all_trials.append((calibration_examples, test_examples))

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

    def __init__(self, k=None, qid2answers=None):
        self.k = k
        self.qid2answers = qid2answers

    def predict(self, example):
        global qid2answers
        qid, labels = example
        answers = self.qid2answers[qid]
        assert(answers), "Answer does not exist!"

        predictions = sorted(labels, key=lambda y: -y.metrics[0])
        thresholds = [-i for i in range(len(predictions))]
        if self.k:
            predictions = predictions[:self.k]
            thresholds = thresholds[:self.k]
        scores = [float(label.text in answers) for label in predictions]
        return BaselineResult(qid, scores, thresholds)


class Threshold(object):
    """Use a simple threshold for any metric."""
    name = "threshold"

    def __init__(self, threshold=None, qid2answers=None):
        self.threshold = threshold
        self.qid2answers = qid2answers

    def predict(self, example):
        qid, labels = example
        answers = self.qid2answers[qid]
        assert(answers), "Answer does not exist!"

        label_scores = [y.metrics[0] for y in labels]
        predictions = []
        thresholds = []
        for i, y in enumerate(labels):
            score = label_scores[i]
            if self.threshold and score < self.threshold:
                continue
            predictions.append(y)
            thresholds.append(score)
        scores = [float(label.text in answers) for label in predictions]
        return BaselineResult(qid, scores, thresholds)
