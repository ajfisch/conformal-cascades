"""General utilities."""

import collections
import tqdm

from absl import flags
import numpy as np

FLAGS = flags.FLAGS

Example = collections.namedtuple(
    "Example",
    ["qid", "labels"])

Label = collections.namedtuple(
    "Label",
    ["text", "metrics"])


def stats(results):
    mean = np.mean(results)
    median = np.median(results)
    p_84 = np.percentile(results, 84)
    p_16 = np.percentile(results, 16)
    return (mean, median, p_84, p_16)


def convert_labels(all_examples, metrics, getter=None):
    """Convert labels to Label tuples.

    Passing tuples around through multiprocessing is less overhead than
    passing around heavy dictionaries (takes time to serialize).

    Args:
      all_examples: <dict>
        Dictionary of qid to list of labels (each label is a dictionary).
      metrics: <list>
        List of metric keys to use.
      getter: <function>
        If provided, might do a transform on the metric. Should accept the label and the
        metric string, and return a float.

    Returns:
      converted_examples: <dict>
        Dict of qid to Label objects.
    """
    converted_examples = {}
    for qid, labels in tqdm.tqdm(all_examples.items(), desc="converting labels"):
        # Create Label objects.
        converted_examples[qid] = []
        for y in labels:
            getter = getter if getter is not None else lambda x, k: x[k]
            label = Label(
                text=y["text"],
                metrics=tuple(getter(y, m) for m in metrics))
            converted_examples[qid].append(label)
    return converted_examples


def evaluate_thresholds(thresholds, threshold_matrix, scores_matrix, label_mask):
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

    Returns:
      all_thresholds: <list>
        List of results (threshold, efficiency, accuracy, cost) points.
    """
    num_examples = threshold_matrix.shape[0]
    num_metrics = threshold_matrix.shape[-1]

    all_thresholds = []
    for threshold in thresholds:
        include = threshold_matrix.min(axis=-1) > threshold
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

        all_thresholds.append((threshold, efficiency, accuracy, cost))

    return all_thresholds
