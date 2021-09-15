"""General utilities."""

import functools
import multiprocessing.pool
import torch

from absl import flags
import numpy as np

INF = 1e18

FLAGS = flags.FLAGS


def stats(results):
    mean = np.mean(results)
    median = np.median(results)
    p_84 = np.percentile(results, 84)
    p_16 = np.percentile(results, 16)
    return (mean, median, p_84, p_16)


def get_calibration_scores(examples, answers, references=None):
    """Compute calibration set using correct values.

    Args:
      examples: <float>[num_examples, max_labels, num_metrics]
        All examples to use for calibration (positives + negatives).
      answers: <float>[num_examples, max_labels]
        Binary mask of acceptable/not acceptable labels.
      references: <int>[num_examples]
        Index of dataset reference.

    Return:
      calibration_scores: <float>[num_examples, num_metrics]
    """
    # If the dataset reference is already given (i.e., standard conformal prediction),
    # we build the calibration set directly from these indices. Otherwise we construct
    # the calibration set from the minimum nonconformity scores, per the relaxed criterion.

    # ==== Option One ===
    # Use references indices.
    if references is not None:
        calibration_scores = examples[np.arange(len(references)), references]

    # ==== Option Two ===
    # Select minimum nonconformity scores.
    else:
        examples = examples + (1 - np.expand_dims(answers, -1)) * INF
        calibration_scores = examples.min(axis=1)
    return calibration_scores


def _evaluate_threshold(threshold, threshold_matrix, scores_matrix, label_mask):
    """Evaluate the results for a given tolerance threshold (epsilon).

    Args:
      threshold: <float>
        Epsilon threshold to evaluate.
      threshold_matrix: <float>[num_examples, max_labels, num_metrics]
        Conservative p-values at each cascade level (until the final).
        They can also not be p-values (just thresholds); epsilon just won't be valid.
      scores_matrix: <float>[num_examples, max_labels]
        Accuracy scores (0 = incorrect, 1 = correct) for all labels.
      label_mask: <float>[num_examples, max_labels]
        Mask of labels vs. padding (1 = label, 0 = padding).

    Returns:
      result: <tuple>
        tuple with (threshold, efficiency, accuracy, cost) points.
    """
    num_examples = threshold_matrix.shape[0]
    num_metrics = threshold_matrix.shape[-1]

    include = (threshold_matrix.min(axis=-1) > threshold) * label_mask
    corrects = ((include * scores_matrix).max(1) > 0).sum()
    accuracy = corrects / num_examples

    if FLAGS.absolute_efficiency:
        efficiency = (include.sum(axis=1)).sum() / num_examples
    else:
        efficiency = (include.sum(axis=1) / label_mask.sum(axis=1)).sum() / num_examples

    # We can prune if the conservative pvalue is already greater than threshold.
    # Anything that happens *after* this event wasn't evaluated.
    can_prune = threshold_matrix <= threshold
    did_evaluate = (np.cumsum(can_prune, axis=-1) <= 1) * np.expand_dims(label_mask, -1)
    num_evaluated = did_evaluate.sum()
    cost = num_evaluated / (label_mask.sum() * num_metrics)

    return (threshold, efficiency, accuracy, cost)


def evaluate_thresholds(thresholds, threshold_matrix, scores_matrix, label_mask, threads=0):
    """Evaluate all given thresholds.

    Args:
      thresholds: <float>[num_thresholds]
        Epsilon thresholds to evaluate.
      threshold_matrix: <float>[num_examples, max_labels, num_metrics]
        Conservative p-values at each cascade level (until the final).
        They can also not be p-values (just thresholds); epsilon just won't be valid.
      scores_matrix: <float>[num_examples, max_labels]
        Accuracy scores (0 = incorrect, 1 = correct) for all labels.
      label_mask: <float>[num_examples, max_labels]
        Mask of labels vs. padding (1 = label, 0 = padding).
      threads: <int>
        Number of threads to use.

    Returns:
      all_thresholds: <list>
        List of results (threshold, efficiency, accuracy, cost) points.
    """
    worker_fn = functools.partial(
        _evaluate_threshold,
        threshold_matrix=threshold_matrix,
        scores_matrix=scores_matrix,
        label_mask=label_mask)

    # Only use multiprocessing with threads > 0.
    if threads > 0:
        workers = multiprocessing.pool.ThreadPool(processes=threads)
        map_fn = workers.imap
    else:
        map_fn = map

    all_thresholds = []
    for result in map_fn(worker_fn, thresholds):
        all_thresholds.append(result)

    # Close cleanly.
    if threads > 0:
        workers.close()
        workers.join()

    return sorted(all_thresholds, key=lambda x: x[0])


def evaluate_thresholds_cuda(thresholds, threshold_matrix, scores_matrix, label_mask):
    thresholds = torch.Tensor(thresholds).cuda()
    threshold_matrix = torch.Tensor(threshold_matrix).cuda()
    scores_matrix = torch.Tensor(scores_matrix).cuda()
    label_mask = torch.Tensor(label_mask).cuda()

    num_examples = threshold_matrix.shape[0]
    num_metrics = threshold_matrix.shape[-1]
    all_results = []
    for threshold in thresholds:
        include = (threshold_matrix.min(dim=-1)[0] > threshold) * label_mask
        corrects = ((include * scores_matrix).max(dim=1)[0] > 0).sum()
        accuracy = corrects.item() / num_examples

        if FLAGS.absolute_efficiency:
            efficiency = (include.sum(axis=1)).sum().item() / num_examples
        else:
            efficiency = (include.sum(axis=1) / label_mask.sum(axis=1)).sum().item() / num_examples

        # We can prune if the conservative pvalue is already greater than threshold.
        # Anything that happens *after* this event wasn't evaluated.
        can_prune = threshold_matrix <= threshold
        did_evaluate = (torch.cumsum(can_prune, dim=-1) <= 1) * torch.unsqueeze(label_mask, -1)
        num_evaluated = did_evaluate.sum()
        cost = (num_evaluated / (label_mask.sum() * num_metrics)).item()

        all_results.append((threshold.item(), efficiency, accuracy, cost))

    return all_results
