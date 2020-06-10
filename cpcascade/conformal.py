"""Utilities for computing conformal metrics."""

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

MAX_THRESHOLDS = 20000

ConformalResult = collections.namedtuple(
    "Result",
    ["qid", "efficiency", "accuracy", "scores", "pvalues", "iterations"])


def compute_pvalue(
    score,
    calibration_scores,
    calibration_mask=None,
    smoothed=True,
    laplace=True,
):
    """Estimate P(V >= v_i).
    Args:
      score: <float>[1] Nonconformity score of sample.
      calibration_scores: <float>[N] Nonconformity scores for calibration.
      calibration_mask: <byte>[N] Points to include for calibration.
      smoothed: <bool> Apply randomized tie-breaking.
      laplace: <bool> Apply laplace smoothing.

    Returns:
      pvalue: <float>
    """
    if calibration_mask is None:
        calibration_mask = np.ones_like(calibration_scores, dtype=np.bool)
    greater = np.sum((calibration_scores > score) * calibration_mask)
    equal = np.sum((calibration_scores == score) * calibration_mask)
    if smoothed:
        equal = np.random.random() * equal
    pvalue = (greater + equal + laplace) / (calibration_scores.shape[0] + laplace)
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
      labels: <Label>[num_labels]
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
    labels = [(y, [1.0] * num_metrics, []) for y in labels]

    # Evaluate conformal cascade.
    iterations = 0
    for i in range(num_metrics):
        # C(x) for next round.
        kept_labels = []

        for y, pvalues, corrected_pvalues in labels:
            # Update realized pvalue for cascade i.
            pvalues[i] = compute_pvalue(
                score=y.metrics[i],
                calibration_scores=calibration_scores[:, i],
                smoothed=smoothed)

            # Correct for MHT, and prune if possible.
            corrected_pvalue = compute_correction(pvalues, correction)
            if corrected_pvalue > epsilon:
                corrected_pvalues.append(corrected_pvalue)
                kept_labels.append((y, pvalues, corrected_pvalues))

            iterations += 1

        # Update current conformal set C(X).
        labels = kept_labels

    # Return remaining labels.
    labels = [(y, corrected_pvalues) for y, _, corrected_pvalues in labels]
    return labels, iterations


def predict_ecdf(
    labels,
    calibration_scores,
    calibration_ecdf,
    epsilon,
    smoothed=True,
):
    """Return the conformal prediction set using the empirical CDF.

    Args:
      labels: <Label>[num_labels]
        Candidate labels to consider.
      calibration_scores: <float>[num_calibration, num_metrics]
        Calibration points for m metrics.
      calibration_ecdf: <float>[num_calibration]
        Empirical CDF for calibration points.
      epsilon: <float>
        Tolerance level.
      smoothed: <bool>
        Whether to apply tie-breaking in pvalue computation.

    Returns:
      labels: (<Label>, List<marginal ecdf>)[num_retained]
        Conformal label set with associated confidence values at each cascade level.
      iterations: <int>
        Number of iterations (pvalue) computations performed.
    """
    num_calibration, num_metrics = calibration_scores.shape

    # Initialize a marginal pvalue and calibration mask placeholder
    # for each label in the label set.
    labels = [(y, np.ones(num_calibration, dtype=np.bool), []) for y in labels]

    # Evaluate conformal cascade.
    iterations = 0
    for i in range(num_metrics):
        # C(x) for next round.
        kept_labels = []

        for y, calibration_mask, marginal_pvalues in labels:
            marginal_ecdf = compute_pvalue(
                score=y.metrics[i],
                calibration_scores=calibration_scores[:, i],
                calibration_mask=calibration_mask,
                smoothed=smoothed)

            # Update ecdf to use this point.
            calibration_ecdf_y = calibration_ecdf.ecdf(
                value=y.metrics[i],
                index=i,
                mask=calibration_mask)

            # Invert because now smaller is worse.
            marginal_pvalue = compute_pvalue(
                score=1 - marginal_ecdf,
                calibration_scores=1 - calibration_ecdf_y,
                smoothed=smoothed)

            # Compare to epsilon.
            if marginal_pvalue > epsilon:
                # Update conditional mask.
                greater = calibration_scores[:, i] > y.metrics[i]
                equal = calibration_scores[:, i] == y.metrics[i]
                if smoothed:
                    equal *= np.random.randint(0, 2, equal.shape, np.bool)
                calibration_mask *= (greater | equal)
                marginal_pvalues.append(marginal_pvalue)
                kept_labels.append((y, calibration_mask, marginal_pvalues))

            iterations += 1

        # Update current conformal set C(X).
        labels = kept_labels

    # Return remaining labels.
    labels = [(y, marginal_pvalues) for y, _, marginal_pvalues in labels]
    return labels, iterations


def score_example(
    example,
    epsilon,
    qid2answers,
    calibration_scores,
    calibration_ecdf=None,
    correction="simes",
    smoothed=True,
):
    """Score an example for conformal prediction.

    Args:
      example: <Example>
        Input example to predict label set for.
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
    qid, labels = example
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
            calibration_ecdf=calibration_ecdf,
            epsilon=epsilon,
            smoothed=smoothed)
    else:
        raise ValueError("Unknown CP method %s" % correction)

    # Compute metrics of the prediction.
    answers = qid2answers[qid]
    assert(answers), "Missing answers!"
    pvalues = [pvalue for _, pvalue in prediction]
    scores = [float(y.text in answers) for y, _ in prediction]
    efficiency = len(prediction) / len(labels)
    accuracy = float(sum(scores) >= 1)

    result = ConformalResult(qid, efficiency, accuracy, scores, pvalues, iterations)
    return result


class ECDF(object):
    """Class to compute ECDF efficiently (for either full or biased)."""

    def __init__(self, values, smoothed=True, unbiased=True):
        # Compute P(V_1 >= v_1, ..., V_n >= v_n)
        self.smoothed = smoothed
        self.unbiased = unbiased
        num_points, num_metrics = values.shape

        # [num_points, num_points, num_metrics]
        diffs = np.expand_dims(values, 0) - np.expand_dims(values, 1)
        greater = np.greater(diffs, 0).astype(np.float32)
        equal = np.equal(diffs, 0).astype(np.float32)

        # Apply random tie-breaking.
        if smoothed:
            equal *= np.random.randint(0, 2, equal.shape, np.bool)
        greater_equal = greater + equal

        # [num_points, num_points]
        greater_equal = np.all(greater_equal, axis=2)

        # [num_points]
        greater_equal = greater_equal.sum(1)

        # Only save what is necessary.
        if unbiased:
            self.greater_equal = greater_equal
            self.values = values
            self.num_points = num_points
        else:
            self.biased_ecdf = greater_equal / num_points

    def ecdf(self, value, index, mask=None):
        if not self.unbiased:
            return self.biased_ecdf

        # [num_points]
        greater = np.greater(value, self.values[:, index])
        if mask is not None:
            greater *= mask
        greater = greater.astype(np.float32)

        # [num_points]
        equal = np.equal(value, self.values[:, index])
        if mask is not None:
            equal *= mask
        equal = equal.astype(np.float32)

        # Apply random tie-breaking.
        if self.smoothed:
            equal *= np.random.randint(0, 2, equal.shape, np.bool)
        greater_equal = greater + equal

        # [num_points]
        ecdf = (self.greater_equal + greater_equal) / (self.num_points + 1)

        return ecdf


def get_calibration_scores(examples, qid2answers, equivalence):
    """Compute calibration set using correct values.

    Args:
      examples: <Example>[num_examples]
        All examples to use for calibration (positives + negatives).
      qid2answers: <dict>
        Map of qid to answer set.
      equivalence: <string>
        How to sample from the equivalence set.

    Return:
      calibration_scores: <float>[num_examples, num_metrics]
    """
    # Collect full equivalence set.
    calibration_labels = []
    for qid, labels in examples:
        answers = qid2answers[qid]
        correct = []
        for y in labels:
            if y.text in answers:
                correct.append(y)
        if correct:
            calibration_labels.append(correct)

    # Note: when equivalence == "all", we expand all of the equivalence labels
    # into individual (X, Y) pairs to calibrate on. This increases the effective
    # sample size, which can be good in scenarios where the total number of
    # calibration exampls is small (similar to, say, data augmentation).
    # For comparison with min calibration, however, it can be a confounding factor
    # in terms of where the improvements come from---so we leave an option to sample.
    if equivalence == "all":
        expanded = []
        for labels in calibration_labels:
            expanded.extend([[label] for label in labels])
        calibration_labels = expanded

    # Get metrics.
    num_calibration = len(calibration_labels)
    num_metrics = len(calibration_labels[0][0].metrics)
    calibration_scores = np.empty((num_calibration, num_metrics))
    for i, labels in enumerate(calibration_labels):
        for j in range(num_metrics):
            if equivalence == "min":
                calibration_scores[i, j] = min([y.metrics[j] for y in labels])
            elif equivalence == "sample":
                calibration_scores[i, j] = np.random.choice([y.metrics[j] for y in labels])
            else:
                metrics = [y.metrics[j] for y in labels]
                assert(len(metrics) == 1)
                calibration_scores[i, j] = metrics[0]
    return calibration_scores


def _worker_init_fn(_qid2answers):
    """Initialize workers with a copy of answers."""
    global qid2answers
    qid2answers = _qid2answers


def _evaluate_conformal_trial(
    examples,
    correction,
    equivalence,
    smoothed,
    target_epsilons,
):
    """Run inner evaluation loop."""
    global qid2answers
    calibration_examples, test_examples = examples

    # Retrieve just the correct labels for calibration.
    calibration_scores = get_calibration_scores(
        examples=calibration_examples,
        qid2answers=qid2answers,
        equivalence=equivalence)

    # Compute ECDF if necessary.
    if correction == "ecdf":
        calibration_ecdf = ECDF(calibration_scores, smoothed)
    elif correction == "ecdf-biased":
        calibration_ecdf = ECDF(calibration_scores, smoothed, unbiased=False)
    else:
        calibration_ecdf = None

    # Initialize output matrices.
    num_examples = len(test_examples)
    num_metrics = calibration_scores.shape[-1]
    max_labels = max([len(ex.labels) for ex in test_examples])
    label_mask = np.zeros((num_examples, max_labels))
    pvalue_matrix = -np.ones((num_examples, max_labels, num_metrics))
    scores_matrix = np.zeros((num_examples, max_labels))

    # Set the epsilon level to be a no-op.
    worker_fn = functools.partial(
        score_example,
        epsilon=-float("inf"),
        qid2answers=qid2answers,
        calibration_scores=calibration_scores,
        calibration_ecdf=calibration_ecdf,
        correction=correction,
        smoothed=smoothed)

    # Populate pvalue and accuracy matrices.
    for i, result in enumerate(map(worker_fn, test_examples)):
        for j in range(len(result.pvalues)):
            label_mask[i, j] = 1
            pvalue_matrix[i, j] = result.pvalues[j]
            scores_matrix[i, j] = result.scores[j]

    # Evaluate all possible accuracies/efficiencies/cost per epsilon.
    viable_pvalues = np.unique(pvalue_matrix - EPS)
    np.random.shuffle(viable_pvalues)
    viable_pvalues = viable_pvalues[:MAX_THRESHOLDS]
    epsilons = utils.evaluate_thresholds(
        thresholds=sorted(viable_pvalues.tolist() + target_epsilons + [0, 1]),
        threshold_matrix=pvalue_matrix,
        scores_matrix=scores_matrix,
        label_mask=label_mask)

    # Collect results.
    trial_results = {}

    # Find target values of epsilon (among unique values).
    results = []
    for target in target_epsilons:
        index = np.argmin(np.abs(target - np.array([eps[0] for eps in epsilons])))
        results.append(epsilons[index])
    trial_results["epsilon"] = results

    # Then take advantage of the fact that it's a step function to fill in missing epsilon.
    points = np.arange(0, 1, 0.0001)
    values = [points]
    for i in range(1, 4):
        f = scipy.interpolate.interp1d([e[0] for e in epsilons], [e[i] for e in epsilons], kind="previous")
        values.append(f(points))
    epsilons = zip(*values)
    trial_results["values"] = []
    for epsilon, efficiency, accuracy, cost in epsilons:
        trial_results["values"].append((epsilon, efficiency, accuracy, cost))

    return trial_results


def conformal_evaluation(
    examples,
    qid2answers,
    calibration_qids,
    test_qids,
    target_epsilons=None,
    correction=None,
    equivalence=None,
    smoothed=None,
    threads=None,
):
    """Compute all conformal metrics.

    Args:
      examples: <dict>
        Dict of all qids to labels to evaluate.
      qid2answers: <dict>
        Map of qid to answer set.
      calibration_qids: <list>
        List of qids just for calibration.
      test_qids: <list>
        List of qids just for testing.
      target_epsilons: <list>
        List of target epsilons.
      correction: <string>
        MHT correction method, one of {simes, bonferroni, ecdf}.
      equivalence: <string>
        How to sample from the equivalence set.
      smoothed: <bool>
        Use tie-breaking during pvalue computation.
      threads: <int>
        Number of threads to use during multiprocessing.

    Returns:
      results: Dict of efficiency and accuracy results.
    """
    correction = correction if correction is not None else FLAGS.correction
    smoothed = smoothed if smoothed is not None else FLAGS.smoothed
    threads = threads if threads is not None else FLAGS.threads
    equivalence = equivalence if equivalence is not None else FLAGS.equivalence
    target_epsilons = target_epsilons or []

    # Store results for all trials.
    trial_results = {"efficiency": [], "epsilon": []}

    # Store all efficiencies/accuracies/costs per epsilon.
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
        _evaluate_conformal_trial,
        correction=correction,
        equivalence=equivalence,
        smoothed=smoothed,
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
            for epsilon, efficiency, accuracy, cost in result["values"]:
                all_epsilons[epsilon].append((efficiency, accuracy, cost))
            pbar.update()

    # Average results over all trials.
    avg_all_epsilons = []
    for epsilon, trials in all_epsilons.items():
        efficiencies = utils.stats([trial[0] for trial in trials])
        accuracies = utils.stats([trial[1] for trial in trials])
        costs = utils.stats([trial[2] for trial in trials])
        avg_all_epsilons.append((epsilon, efficiencies, accuracies, costs))

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
