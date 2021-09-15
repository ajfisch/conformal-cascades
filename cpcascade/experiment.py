"""Run full experiment."""

import collections
import functools
import json
import os

from absl import flags
from absl import logging

from cpcascade import conformal
from cpcascade import baselines

flags.DEFINE_enum("correction", "simes",
                  ["simes", "bonferroni", "ecdf", "ecdf-biased", "none"],
                  "MHT correction method.")

flags.DEFINE_boolean("smoothed", True,
                     "Apply tie-breaking for pvalues.")

flags.DEFINE_boolean("absolute_efficiency", False,
                     "Count the size of the cset without dividing by |label space|.")

flags.DEFINE_boolean("skip_conformal", False,
                     "Skip conformal (to run baselines only).")

flags.DEFINE_boolean("skip_baselines", False,
                     "Skip baselines (to run conformal only).")

flags.DEFINE_float("percent_calibration", 0.8,
                   "Percent of data to use for calibration.")

flags.DEFINE_integer("inner_threads", 0,
                     "Number of threads to use in inner loops.")

flags.DEFINE_integer("outer_threads", 20,
                     "Number of threads to use in outer loops.")

flags.DEFINE_boolean("cuda", False,
                     "Compute threshold results on GPU.")

flags.DEFINE_string("output_dir", "/tmp/conformal_outputs",
                    "Directory where results are written to.")

flags.DEFINE_boolean("overwrite_output", False,
                     "Overwrite files in output dir")

flags.DEFINE_boolean("equivalence", True,
                     "Take minimum when calibrating.")

flags.DEFINE_integer("seed", 42, "Random seed.")

delattr(flags.FLAGS, "alsologtostderr")
flags.DEFINE_boolean("alsologtostderr", True, "Log to stderr.")

FLAGS = flags.FLAGS


def run_experiment(
    examples,
    answers,
    mask,
    calibration_ids,
    test_ids,
    baseline_metrics,
    references=None,
    epsilons=None,
):
    """Run all numbers.

    Args:
      examples: <float> [num_examples, max_labels, num_metrics]
        Array of label metrics.
      answers: <float> [num_examples, max_labels]
        Indicator of acceptable/not acceptable labels.
      mask: <float> [num_examples, max_labels]
        Indicator for padding or not padding.
      calibration_ids: <list>
        List of ids just for calibration.
      test_ids: <list>
        List of ids just for testing.
      baseline_metrics: List[int]
        The main metrics the baselines should be using.
      references: <int>[num_examples]
        Indices to use as references.

    Returns:
      all_results: <dict>
        Dictionary of all the evaluation results.
    """
    if not FLAGS.skip_baselines:
        for metric in baseline_metrics:
            assert metric in FLAGS.metrics, metric

    def write_table(metric, results):
        if not epsilons:
            return
        filename = os.path.join(FLAGS.output_dir, "%s_table.txt" % metric)
        with open(filename, "w") as f:
            for result in results["targets"]:
                latex_str = "& %2.2f " % (1 - result["epsilon"])
                for key in ["accuracy", "efficiency"]:
                    latex_str += "& %2.2f (%2.2f-%2.2f)" % (result[key][0], result[key][2], result[key][3])
                if "cost" in result:
                    key = "cost"
                    latex_str += "& %2.2f (%2.2f-%2.2f)" % (result[key][0], result[key][2], result[key][3])
                f.write(latex_str + "\\\\" + "\n")

    all_results = collections.defaultdict(lambda: collections.defaultdict(dict))

    logging.info("=" * 50)
    if not FLAGS.skip_conformal:
        logging.info("Running conformal evaluation.")

        results = conformal.conformal_evaluation(
            examples=examples,
            answers=answers,
            mask=mask,
            calibration_ids=calibration_ids,
            test_ids=test_ids,
            references=references,
            target_epsilons=epsilons)

        all_results["epsilon"]["conformal"] = results
        write_table("conformal", results)

    if not FLAGS.skip_baselines:
        logging.info("=" * 50)
        logging.info("Running baselines.")
        run_baseline = functools.partial(
            baselines.baseline_evaluation,
            answers=answers,
            mask=mask,
            calibration_ids=calibration_ids,
            test_ids=test_ids,
            references=references,
            target_epsilons=epsilons)

        for baseline_metric in baseline_metrics:
            baseline_metric_idx = FLAGS.metrics.index(baseline_metric)
            logging.info("Main metric = %s", baseline_metric)

            results = run_baseline(
                examples=examples[:, :, baseline_metric_idx],
                baseline_class=baselines.TopK)
            all_results["epsilon"]["top_k_%s" % baseline_metric] = results
            write_table("top_k_%s" % baseline_metric, results)

            results = run_baseline(
                examples=examples[:, :, baseline_metric_idx],
                baseline_class=baselines.Threshold)
            all_results["epsilon"]["threshold_%s" % baseline_metric] = results
            write_table("threshold_%s" % baseline_metric, results)

    output_file = os.path.join(FLAGS.output_dir, "results.json")
    logging.info("Writing results to %s", output_file)
    with open(output_file, "w") as f:
        json.dump(all_results, f, sort_keys=True, indent=4)

    flag_file = os.path.join(FLAGS.output_dir, "flags.txt")
    with open(flag_file, "w") as f:
        f.write(FLAGS.flags_into_string())

    return all_results
