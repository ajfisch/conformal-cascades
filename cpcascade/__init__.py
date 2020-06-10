"""Run full experiment."""

import collections
import functools
import json
import os

from absl import flags
from absl import logging

from cpcascade import conformal
from cpcascade import utils
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

flags.DEFINE_integer("threads", 30,
                     "Number of threads to use.")

flags.DEFINE_string("output_dir", "/tmp/conformal_outputs",
                    "Directory where results are written to.")

flags.DEFINE_enum("equivalence", "min", ["min", "sample", "all"],
                  "Take minimum when calibrating.")

flags.DEFINE_integer("seed", 42, "Random seed.")

delattr(flags.FLAGS, "alsologtostderr")
flags.DEFINE_boolean("alsologtostderr", True, "Log to stderr.")

FLAGS = flags.FLAGS


def run_experiment(
    all_examples,
    qid2answers,
    calibration_qids,
    test_qids,
    baseline_metrics,
    epsilons=None,
    efficiencies=None,
    get_metric=lambda arr, k: arr[k],
):
    """Run all numbers.

    Args:
      all_examples: <dict>
        Dict of all qids to their list of labels.
      qid2answers: <dict>
        Dict of qid to answer sets.
      qid2answers: <dict>
        Map of qid to answer set.
      calibration_qids: <list>
        List of qids just for calibration.
      test_qids: <list>
        List of qids just for testing.
      baseline_metrics: List[<string>]
        The main metrics the baselines should be using.
      get_metric: <function>
        Specialized getter from the label dict (i.e., converting to nonconformal).

    Returns:
      all_results: <dict>
        Dictionary of all the evaluation results.
    """
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
        examples = utils.convert_labels(
            all_examples=all_examples,
            metrics=FLAGS.metrics,
            getter=get_metric)

        results = conformal.conformal_evaluation(
            examples=examples,
            calibration_qids=calibration_qids,
            test_qids=test_qids,
            qid2answers=qid2answers,
            target_epsilons=epsilons)

        all_results["epsilon"]["conformal"] = results
        write_table("conformal", results)

    if not FLAGS.skip_baselines:
        logging.info("=" * 50)
        logging.info("Running baselines.")
        run_baseline = functools.partial(
            baselines.baseline_evaluation,
            target_epsilons=epsilons,
            calibration_qids=calibration_qids,
            test_qids=test_qids,
            qid2answers=qid2answers)

        for baseline_metric in baseline_metrics:
            logging.info("Main metric = %s", baseline_metric)
            examples = utils.convert_labels(
                all_examples=all_examples,
                metrics=[baseline_metric])

            results = run_baseline(
                examples=examples,
                baseline_class=baselines.TopK)
            all_results["epsilon"]["top_k_%s" % baseline_metric] = results
            write_table("top_k_%s" % baseline_metric, results)

            results = run_baseline(
                examples=examples,
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
