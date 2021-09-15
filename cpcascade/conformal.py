"""Utilities for computing conformal metrics."""

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

INF = 1e18

FLAGS = flags.FLAGS

MAX_THRESHOLDS = 5000

ConformalLabel = collections.namedtuple(
    "ConformalLabel",
    ["idx", "pvalues"])

ConformalResult = collections.namedtuple(
    "ConformalResult",
    ["efficiency", "accuracy", "scores", "pvalues", "iterations"])


def compute_pvalue(
    score,
    calibration_scores,
    smoothed=True,
):
    """Estimate P(V >= v_i).
    Args:
      score: <float>[1] Nonconformity score of sample.
      calibration_scores: <float>[N] Nonconformity scores for calibration.
      smoothed: <bool> Apply randomized tie-breaking.

    Returns:
      pvalue: <float>
    """
    greater = np.sum(calibration_scores > score)
    equal = np.sum(calibration_scores == score)
    if smoothed:
        equal = np.random.random() * equal

    # We add +1 to account for n + 1th example in conformalization process.
    pvalue = (greater + equal + 1) / (calibration_scores.shape[0] + 1)
    return pvalue.item()


def compute_correction(pvalues, correction="simes"):
    """Compute a multiple hypothesis testing adjusted pvalue.

    Args:
      pvalues: <float>[m] Multiple hypothesis testing pvalues.
      correction: <string> One of {bonferroni, simes}.

    Returns:
      pvalue: <float> Adjusted overall pvalue.
    """
    n = len(pvalues)
    if correction == "simes":
        adjusted_pvalues = [n * p_i / i for i, p_i in enumerate(sorted(pvalues), 1)]
    elif correction == "bonferroni":
        adjusted_pvalues = [n * p_i for p_i in pvalues]
    elif correction == "none":
        adjusted_pvalues = pvalues
    else:
        raise ValueError("Unknown correction %s" % correction)
    return min(adjusted_pvalues)


def predict_mht(
    labels,
    calibration_scores,
    epsilon,
    smoothed=True,
    correction="simes",
):
    """Return the conformal prediction set using multiple hypothesis testing.

    Args:
      labels: <float>[max_labels, num_metrics]
        Candidate labels to consider.
      calibration_scores: <float>[num_calibration, num_metrics]
        Calibration points for m metrics.
      epsilon: <float>
        Tolerance level.
      smoothed: <bool>
        Whether to apply tie-breaking in pvalue computation.
      correction: <string>
        Type of MHT correction to apply.

    Returns:
      labels: (<Label>, List<pvalue>)[num_retained]
        Conformal label set with associated confidence values at each cascade level.
      iterations: <int>
        Number of iterations (pvalue) computations performed.
    """
    num_metrics = calibration_scores.shape[1]

    # Pair labels with conservative pvalues.
    labels = [(idx, y, [1.0] * num_metrics, []) for idx, y in enumerate(labels)]

    # Evaluate conformal cascade.
    iterations = 0
    for i in range(num_metrics):
        # C(x) for next round.
        kept_labels = []

        for idx, y, pvalues, corrected_pvalues in labels:
            # Update realized pvalue for cascade i.
            pvalues[i] = compute_pvalue(
                score=y[i],
                calibration_scores=calibration_scores[:, i],
                smoothed=smoothed)

            # Correct for MHT, and prune if possible.
            corrected_pvalue = compute_correction(pvalues, correction)
            if corrected_pvalue > epsilon:
                corrected_pvalues.append(corrected_pvalue)
                kept_labels.append((idx, y, pvalues, corrected_pvalues))

            iterations += 1

        # Update current conformal set C(X).
        labels = kept_labels

    # Return remaining labels.
    labels = [ConformalLabel(idx, corrected_pvalues) for idx, _, _, corrected_pvalues in labels]
    return labels, iterations


def predict_ecdf(
    labels,
    calibration_scores,
    ecdf_cache,
    epsilon,
    smoothed=True,
):
    """Return the conformal prediction set using the empirical CDF.

    Args:
      labels: <float>[max_labels, num_metrics]
        Candidate labels to consider.
      calibration_scores: <float>[num_calibration, num_metrics]
        Calibration points for m metrics.
      ecdf_cache: ECDF object.
        Cached empirical CDF for calibration points.
      epsilon: <float>
        Tolerance level.
      smoothed: <bool>
        Whether to apply tie-breaking in pvalue computation.

    Returns:
      labels: List<ConformalLabels>[num_retained]
        Conformal label set with associated confidence values at each cascade level.
      iterations: <int>
        Number of iterations (pvalue) computations performed.
    """
    num_calibration, num_metrics = calibration_scores.shape

    # Pair labels with conservative nonconformity scores.
    labels = [(idx, y, np.array([-INF] * num_metrics), []) for idx, y in enumerate(labels)]

    # Evaluate conformal cascade.
    iterations = 0
    for i in range(num_metrics):
        # C(x) for next round.
        kept_labels = []

        for idx, y, scores, corrected_pvalues in labels:
            # === Overview ===
            # Given an m-dimensional metrics vector, [v^1, ..., v^m], we want to
            # get a single scalar that represents its nonconformity. One such scalar
            # is 1 - P(V^1 ≥ v^1, ..., V^m ≥ v^m) (i.e., derived from the ECDF).
            #
            # If we do this symmetrically to both the calibration and new test point,
            # then the resulting scalar ECDF value will itself act as a nonconformity
            # score. This is essentially the same as doing "full" conformal prediction,
            # just with a very simple nonconformity measure. Higher ECDF values indicate
            # a higher relative degree of nonconformity.
            #
            # Given the new joint ECDF values, we then compute standard p-values.
            # To remain valid, we conservatively set unknown nonconformity measures to
            # -INF (i.e., the most conforming possible). See paper for details.

            # === Step One ===
            # Update scores to use all realized values so far.
            scores[i] = y[i]

            # === Step Two ===
            # Compute p' =  1 - P(V^1 ≥ v^1, ..., V^m ≥ v^m) for standard nonconformity scores.
            # We get this for all n + 1 scores. The ECDF object caches this.
            ecdf_calibration, ecdf_y = ecdf_cache.ecdf(scores, i)

            # === Step Three ===
            # Compute the new p-value for p', P(P ≥ p').
            corrected_pvalue = compute_pvalue(
                score=ecdf_y,
                calibration_scores=ecdf_calibration,
                smoothed=smoothed)

            # === Step Four ===
            # Compare P(P ≥ p') to epsilon. We can safely reject if it is ≤ epsilon.
            if corrected_pvalue > epsilon:
                corrected_pvalues.append(corrected_pvalue)
                kept_labels.append((idx, y, scores, corrected_pvalues))

            iterations += 1

        # Update current conformal set C(X).
        labels = kept_labels

    # Return remaining labels.
    labels = [ConformalLabel(idx, marginal_pvalues) for idx, _, _, marginal_pvalues in labels]
    return labels, iterations


def score_example(
    example,
    epsilon,
    calibration_scores,
    calibration_ecdf=None,
    correction="simes",
    smoothed=True,
):
    """Score an example for conformal prediction.

    Args:
      example: [<float>[max_labels, num_metrics], <float>[max_labels]]
        Input example to predict label set for, together with correct answer indices.
      epsilon: <float>
        Tolerance level.
      calibration_scores: <float>[num_calibration, num_metrics]
        Calibration points for m metrics.
      calibration_ecdf: <float>[num_calibration]
        Empirical CDF for calibration points.
      correction: <string>
        Calibration correction method, one of {simes, bonferroni, ecdf, none}.
      smoothed: <bool>
        Use tie-breaking during pvalue computation.

    Returns:
      result: <ConformalResult>
    """
    labels, answers = example
    answers = np.flatnonzero(answers)
    num_metrics = calibration_scores.shape[1]

    # Get conformal label set.
    if num_metrics == 1:
        prediction, iterations = predict_mht(
            labels=labels,
            calibration_scores=calibration_scores,
            epsilon=epsilon,
            smoothed=smoothed,
            correction="none")
    elif correction in ["simes", "bonferroni", "none"]:
        prediction, iterations = predict_mht(
            labels=labels,
            calibration_scores=calibration_scores,
            epsilon=epsilon,
            smoothed=smoothed,
            correction=correction)
    elif correction in ["ecdf", "ecdf-biased"]:
        prediction, iterations = predict_ecdf(
            labels=labels,
            calibration_scores=calibration_scores,
            ecdf_cache=calibration_ecdf,
            epsilon=epsilon,
            smoothed=smoothed)
    else:
        raise ValueError("Unknown CP method %s" % correction)

    # Compute metrics of the prediction.
    pvalues = [y.pvalues for y in prediction]
    scores = [float(y.idx in answers) for y in prediction]
    efficiency = len(prediction) / len(labels)
    accuracy = float(sum(scores) >= 1)
    result = ConformalResult(efficiency, accuracy, scores, pvalues, iterations)
    return result


class ECDF(object):
    """Class to compute cached ECDF efficiently."""

    def __init__(self, calibration_values):
        """Cache values for estimating the ECDF."""
        num_points, num_metrics = calibration_values.shape

        # [num_points, num_metrics]
        self.calibration_values = calibration_values
        self.num_points = num_points

        # [num_points, num_points, num_metrics]
        less_equal = np.less_equal(np.expand_dims(calibration_values, 0),
                                   np.expand_dims(calibration_values, 1))

        # [num_points, num_points]
        less_equal = np.all(less_equal, axis=2)

        # [num_points]
        self.less_equal = less_equal.sum(1)

    def ecdf(self, values, i):
        """Update ecdf using the new data point."""
        # Compute for the old points.
        # [num_points, num_metrics]
        less_equal_cal = np.less_equal(
            np.expand_dims(values, 0),
            self.calibration_values)

        # [num_points]
        less_equal_cal = self.less_equal + np.all(less_equal_cal, axis=1)
        ecdf_cal = less_equal_cal / (self.num_points + 1)

        # Compute for the new point. We take a conservative estimate,
        # P(V_1 ≤ v_1, ..., V_m ≤ v_m) ≥ 1 - P(not V_1 ≤ v_1, ..., V_i ≤ v_i).
        # [num_points, num_metrics]
        less_equal_val = np.less_equal(
            self.calibration_values,
            np.expand_dims(values, 0))

        # [num_points]
        less_equal_val = np.all(less_equal_val, axis=1)
        ecdf_val = (less_equal_val.sum() + 1) / (self.num_points + 1)

        return ecdf_cal, ecdf_val


def _evaluate_conformal_trial(
    examples,
    correction,
    smoothed,
    target_epsilons,
    equivalence=True,
    threads=0,
    cuda=False
):
    """Run inner evaluation loop."""
    calibration_examples = examples[0]
    calibration_answers = examples[1]
    references = examples[2]
    test_examples = examples[3]
    test_answers = examples[4]
    test_mask = examples[5]

    # Retrieve just the metrics for the calibration examples.
    # [num_calibration, num_metrics]
    calibration_scores = utils.get_calibration_scores(
        examples=calibration_examples,
        answers=calibration_answers,
        references=references if not equivalence else None)

    # If using smoothing, apply random perturbation to break ties.
    if smoothed:
        test_examples += np.random.uniform(0.0, EPS, size=test_examples.shape)
        calibration_examples += np.random.uniform(0.0, EPS, size=calibration_examples.shape)

    # Compute ECDF if necessary.
    if correction == "ecdf":
        calibration_ecdf = ECDF(calibration_scores)
    else:
        calibration_ecdf = None

    # Set the epsilon level to be a no-op, i.e., we never early exit.
    worker_fn = functools.partial(
        score_example,
        epsilon=-float("inf"),
        calibration_scores=calibration_scores,
        calibration_ecdf=calibration_ecdf,
        correction=correction,
        smoothed=smoothed)

    # Only use multiprocessing with threads > 0.
    if threads > 0:
        workers = multiprocessing.pool.Pool(processes=threads)
        map_fn = workers.imap
    else:
        map_fn = map

    # Populate p-value matrix. We compute the p-value for each label.
    logging.debug("Populating p-value matrix.")
    pvalue_matrix = -np.ones_like(test_examples)
    for i, result in enumerate(map_fn(worker_fn, zip(test_examples, test_answers))):
        for j in range(len(result.pvalues)):
            pvalue_matrix[i, j] = result.pvalues[j]

    # Close cleanly.
    if threads > 0:
        workers.close()
        workers.join()

    # Evaluate all possible accuracies/efficiencies/cost per epsilon.
    viable_pvalues = np.unique(pvalue_matrix - EPS)
    np.random.shuffle(viable_pvalues)
    viable_pvalues = viable_pvalues[:MAX_THRESHOLDS]
    thresholds = sorted(viable_pvalues.tolist() + target_epsilons + [0, 1])
    logging.debug("Evaluating per epsilon over %s values.", len(thresholds))
    if cuda:
        epsilons = utils.evaluate_thresholds_cuda(
            thresholds=thresholds,
            threshold_matrix=pvalue_matrix,
            scores_matrix=test_answers,
            label_mask=test_mask)
    else:
        epsilons = utils.evaluate_thresholds(
            thresholds=thresholds,
            threshold_matrix=pvalue_matrix,
            scores_matrix=test_answers,
            label_mask=test_mask,
            threads=threads)

    # Collect results.
    logging.debug("Collecting results.")
    trial_results = {}

    # Find target values of epsilon (among unique values).
    results = []
    for target in target_epsilons:
        index = np.argmin(np.abs(target - np.array([eps[0] for eps in epsilons])))
        results.append(epsilons[index])
    trial_results["epsilon"] = results

    # Then take advantage of the fact that it's a step function to fill in missing epsilon.
    logging.debug("Fill missing epsilon.")
    points = np.arange(0, 1, 0.0001)
    values = [points]
    for i in range(1, 4):
        f = scipy.interpolate.interp1d([e[0] for e in epsilons], [e[i] for e in epsilons], kind="next")
        values.append(f(points))
    epsilons = zip(*values)
    trial_results["values"] = []
    for epsilon, efficiency, accuracy, cost in epsilons:
        trial_results["values"].append((epsilon, efficiency, accuracy, cost))

    return trial_results


def conformal_evaluation(
    examples,
    answers,
    mask,
    calibration_ids,
    test_ids,
    references=None,
    target_epsilons=None,
    correction=None,
    equivalence=None,
    smoothed=None,
    inner_threads=None,
    outer_threads=None,
):
    """Compute all conformal metrics.

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
      references: <int>[num_examples]
        Indices to use as references.
      target_epsilons: <list>
        List of target epsilons.
      correction: <string>
        MHT correction method, one of {simes, bonferroni, ecdf}.
      equivalence: <bool>
        Use min nonconformity score from the acceptable lables.
      smoothed: <bool>
        Use tie-breaking during pvalue computation.
      inner_threads: <int>
        Number of threads to use during multiprocessing inner loop.
      outer_threads: <int>
        Number of threads to use during multiprocessing outer loop.

    Returns:
      results: Dict of efficiency and accuracy results.
    """
    correction = correction if correction is not None else FLAGS.correction
    smoothed = smoothed if smoothed is not None else FLAGS.smoothed
    outer_threads = outer_threads if outer_threads is not None else FLAGS.outer_threads
    inner_threads = inner_threads if inner_threads is not None else FLAGS.inner_threads
    equivalence = equivalence if equivalence is not None else FLAGS.equivalence
    target_epsilons = target_epsilons or []

    # Store results for all trials.
    trial_results = {"efficiency": [], "epsilon": []}

    # Store all efficiencies/accuracies/costs per epsilon.
    all_epsilons = collections.defaultdict(list)

    # Only use multiprocessing with threads > 0.
    if outer_threads > 0:
        if inner_threads > 0:
            workers = multiprocessing.pool.ThreadPool(processes=outer_threads)
        else:
            workers = multiprocessing.pool.Pool(processes=outer_threads)
        map_fn = workers.imap_unordered
    else:
        map_fn = map

    worker_fn = functools.partial(
        _evaluate_conformal_trial,
        correction=correction,
        equivalence=equivalence,
        smoothed=smoothed,
        target_epsilons=target_epsilons,
        threads=inner_threads,
        cuda=FLAGS.cuda)

    # Gather and evaluate all trials.
    all_trials = []
    logging.debug("Gather all trials.")
    for i in range(len(calibration_ids)):
        calibration_examples = examples[calibration_ids[i]]
        calibration_answers = answers[calibration_ids[i]]
        calibration_references = references[calibration_ids[i]]
        test_examples = examples[test_ids[i]]
        test_answers = answers[test_ids[i]]
        test_mask = mask[test_ids[i]]
        all_trials.append((calibration_examples,
                           calibration_answers,
                           calibration_references,
                           test_examples,
                           test_answers,
                           test_mask))

    # Run.
    logging.debug("Running trials.")
    with tqdm.tqdm(total=len(all_trials), desc="evaluating") as pbar:
        for result in map_fn(worker_fn, all_trials):
            trial_results["epsilon"].append(result["epsilon"])
            for epsilon, efficiency, accuracy, cost in result["values"]:
                all_epsilons[epsilon].append((efficiency, accuracy, cost))
            pbar.update()

    # Close cleanly.
    if outer_threads > 0:
        workers.close()
        workers.join()

    # Average results over all trials.
    logging.debug("Average results.")
    avg_all_epsilons = []
    for epsilon, trials in all_epsilons.items():
        efficiencies = utils.stats([trial[0] for trial in trials])
        accuracies = utils.stats([trial[1] for trial in trials])
        costs = utils.stats([trial[2] for trial in trials])
        avg_all_epsilons.append((epsilon, efficiencies, accuracies, costs))

    logging.debug("Compute results for target epsilons.")
    avg_target_epsilons = []
    for i in range(len(target_epsilons)):
        epsilon = np.mean([trial[i][0] for trial in trial_results["epsilon"]])
        efficiencies = utils.stats([trial[i][1] for trial in trial_results["epsilon"]])
        accuracies = utils.stats([trial[i][2] for trial in trial_results["epsilon"]])
        costs = utils.stats([trial[i][3] for trial in trial_results["epsilon"]])
        avg_target_epsilons.append(dict(
            epsilon=epsilon,
            efficiency=efficiencies,
            accuracy=accuracies,
            cost=costs))

    results = dict(targets=avg_target_epsilons,
                   data=avg_all_epsilons)

    return results
