import matplotlib as mpl
mpl.use("Agg")
import json
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import seaborn as sns
import os
import glob
import scipy
import traceback
from itertools import cycle

from absl import app
from absl import flags

sns.set_context("paper", font_scale=2.0)
sns.set_style("white")
sns.set_style("ticks")
sns.despine()

LINES = ["--", "-.", ":"]
OURS_LINE = "-"
SINGLE_LINE = ":"

COLORS = [u"#1f77b4", u"#2ca02c", u"#9467bd", u"#8c564b",
          u"#e377c2", u"#7f7f7f", u"#bcbd22", u"#17becf"]
OURS_COLOR = u"#ff7f0e"
SINGLE_COLOR = u"#d62728"

flags.DEFINE_string("exp_dir", None,
                    "Path to directory containing the results of the conformal experiments.")

flags.DEFINE_string("baseline_dir", None,
                    "Path to directory containing the results of the baseline experiments.")

flags.DEFINE_string("filter", "",
                    "Run only on experiment names that contain this string")

flags.DEFINE_integer("n_bins", 100,
                     "Number of bins for plot binning")

flags.DEFINE_integer("n_bins_eff_suc", None,
                     "Number of bins for plot binning")

flags.DEFINE_integer("dpi", 100,
                     "DPI for saving")

flags.DEFINE_multi_float("x_lim", [-0.01, 0.21],
                         "X lim for eff_suc plot")

flags.DEFINE_multi_float("y_lim", [0.75, 1.01],
                         "X lim for eff_suc plot")

flags.DEFINE_float("x_lim_eff", None,
                   "X lim for eps_eff plot")

flags.DEFINE_float("y_lim_eff", None,
                   "X lim for eps_eff plot")

flags.DEFINE_float("zoom_factor", None,
                   "Zoom factor for zoom box")

flags.DEFINE_boolean("y_median", False,
                     "Aggregate y by median instead of mean")

flags.DEFINE_boolean("breakdown", False,
                     "Create breakdown plot for the metrics. Requires all breakdown \
                     results to be available in their respective dirs.")

flags.DEFINE_boolean("compare_corrections", False,
                     "Compare corrections (run on simes dir)")

flags.DEFINE_boolean("skip_single", False,
                     "Skip non comparsion/ breakdown plots")

flags.DEFINE_list("epsilons", None,
                  "Target accuracyies of 1 - epsilon to to include in output table.")

FLAGS = flags.FLAGS

#CORRECTIONS = ["none", "bonferroni", "simes", "ecdf"]
CORRECTIONS = ["none", "bonferroni", "simes"]


def get_task(path):
    task = ""
    if "/ir/" in path:
        task = "ir"
    elif "/hiv/" in path:
        task = "hiv"
    elif "/qa/" in path:
        task = "qa"
    elif "/open_qa/" in path:
        task = "open_qa"
    elif "/chembl/" in path:
        task = "chembl"
    else:
        raise Exception
    return task


METRIC_ABBRV = {"ir": {"rank": "BM25 rank", "logit": "CLS logit", "bm25": "BM25"},
                "hiv": {"rank_random_forest": "rank(RF)", "random_forest": "RF",
                        "chemprop_optimized": "Chemprop", "rank_chemprop_optimized": "rank(Chemprop)",
                        "svm": "SVM", "rank_svm": "rank(SVM)"},
                "qa": {"rank": "rank(EXT)", "start_logit": "start logit", "end_logit": "end logit",
                       "sum": "sum logits", "start_prob": "start prob", "end_prob": "end prob",
                       "rerank_logit": "CLS logit", "rerank_sigmoid": "prob(CLS)"},
                "open_qa": {"start_logit": "span st", "end_logit": "span en",
                       "sum": "span sum", "relevance_logit": "parag",
                       "psg_score": "retriever", "psg_rank": "doc rank (ret)"},
                "chembl": {"RF": "RF", "MPN": "MPNN"}}

def get_conformal(data):
    try:
        return zip(*data["epsilon"]["data"])
    except KeyError:
        return zip(*data["epsilon"]["conformal"]["data"])


def get_baselines(baseline_dir):
    task_baselines = {"ir": [
                            #("threshold_bm25", "Threshold BM25"), ("top_k_bm25", "Top-K BM25"),
                             ("threshold_logit", "Threshold CLS"), ("top_k_logit", "Top-K CLS")],
                      "hiv": [("threshold_chemprop_optimized", "Threshold Chemprop"),
                              ("top_k_chemprop_optimized", "Top-K Chemprop"),
                              ("threshold_random_forest", "Threshold RF"),
                              ("top_k_random_forest", "Top-K RF")],
                      "qa": [("threshold_sum", "Threshold EXT"), ("top_k_sum", "Top-K EXT"),
                             ("threshold_rerank_logit", "Threshold CLS"),
                             ("top_k_rerank_logit", "Top-K CLS")],
                      "chembl": [("threshold_MPN", "Threshold MPNN"), ("top_k_MPN", "Top-K MPNN")],
                      "open_qa": [("threshold_sum", "Threshold span score"), ("top_k_sum", "Top-K span score")]}
                      #"open_qa": [("threshold_relevance_plus_score", "Threshold sum scores"), ("top_k_relevance_plus_score", "Top-K sum scores")]}

    task = get_task(baseline_dir)
    metrics = task_baselines[task]
    outlist = []
    baseline_data = json.load(open(os.path.join(baseline_dir, "results.json"), "r"))
    for metric in metrics:
        eps, eff, acc = zip(*baseline_data["epsilon"][metric[0]]["data"])
        eps, eff, acc = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc))
        outlist.append((eps, eff, acc, metric[1]))
    return outlist


def get_res_file(res_dir):
    # If only one metric, use the same correction since it doesn't matter (say simes)
    start = res_dir.find("-metrics=") + len("-metrics=")
    end = res_dir.find("-equivalence=")
    metrics = res_dir[start: end].split(",")
    if len(metrics) == 1:
        start = res_dir.find("-correction=") + len("-correction=")
        end = res_dir.find("-metrics=")
        res_dir = res_dir[:start] + 'simes' + res_dir[end:]
    res_file = os.path.join(res_dir, "results.json")
    return res_file


def area_under_curve(xs, ys, low=0.0, high=1.0, res=1000):
    """Area under the efficiency-accuracy curve."""
    x_inter = np.arange(low, high, (high - low) / res)
    f_interpolate = scipy.interpolate.interp1d(xs, ys, bounds_error=False, fill_value=(0, 1))
    y_inter = np.clip(f_interpolate(x_inter), 0, 1)
    area = np.trapz(y_inter, x_inter) / (max(x_inter) - min(x_inter))
    return area


def plot_smooth(ax, x, y, n_bins=100, name="", lowauc=0.0, highauc=1.0,
                pre_zero=None, post_one=None,
                line_style=OURS_LINE, color=OURS_COLOR):
    """Plot bin-smoothed line plot."""
    # Smooth points by taking the average of a bin (evenly spaced, n_bins of them).
    try:
        x[0][0]
        x = np.array([e[0] for e in x])
    except Exception:
        x = x

    try:
        len(y[0])
    except:
        import pdb; pdb.set_trace()
    y_er_top = np.array([e[2] for e in y])
    y_er_bot = np.array([e[3] for e in y])
    y = np.array([e[0] for e in y])

    bins = np.linspace(0, 1, n_bins)
    digitized = np.digitize(x, bins)
    x_smth = np.array([np.mean(x[digitized == i]) for i in np.unique(digitized)])
    if FLAGS.y_median:
        y_smth = np.array([np.median(y[digitized == i]) for i in np.unique(digitized)])
    else:
        y_smth = np.array([np.mean(y[digitized == i]) for i in np.unique(digitized)])
    y_er_top = np.array([np.mean(y_er_top[digitized == i]) for i in np.unique(digitized)])
    y_er_bot = np.array([np.mean(y_er_bot[digitized == i]) for i in np.unique(digitized)])

    # Add "fill" point before 0.
    if pre_zero is not None:
        x_smth = np.concatenate([np.array([0.]), x_smth])
        x = np.concatenate([np.array([0.]), x])
        y_smth = np.concatenate([np.array([pre_zero]), y_smth])
        y = np.concatenate([np.array([pre_zero]), y])
        y_er_top = np.concatenate([np.array([pre_zero]), y_er_top])
        y_er_bot = np.concatenate([np.array([pre_zero]), y_er_bot])

    # Add "fill" point after 1.
    if post_one is not None:
        x_smth = np.concatenate([x_smth, np.array([1.])])
        x = np.concatenate([x, np.array([1.])])
        y_smth = np.concatenate([y_smth, np.array([post_one])])
        y = np.concatenate([y, np.array([post_one])])
        y_er_top = np.concatenate([y_er_top, np.array([post_one])])
        y_er_bot = np.concatenate([y_er_bot, np.array([post_one])])

    # Plot with error bounds.
    plt.plot(x_smth, y_smth, line_style, color=color, linewidth=2,
             label="{} ({:#.3g})".format(name, area_under_curve(x, y, lowauc, highauc, res=n_bins)))
    plt.fill_between(x_smth, y_er_bot, y_er_top, color=color, alpha=0.2)


def plot_results(res_dir, n_bins=100):
    """Plot results for a single configuration (one results file)."""
    res_file = get_res_file(res_dir)
    data = json.load(open(res_file, "r"))

    eps, eff, acc, cost = get_conformal(data)

    # Write output table.
    if FLAGS.epsilons is not None:
        # Find target values of epsilon (among unique values).
        results = []
        for target in FLAGS.epsilons:
            index = np.argmin(np.abs(target - np.array(eps)))
            results.append((eps[index], acc[index][0], eff[index][0], cost[index][0]))

        filename = os.path.join(res_dir, 'table.txt')
        with open(filename, "w") as f:
            latex_str = " 1 - $\eps$ & Succ. & $|\cset|$ & Amortized Cost"
            f.write(latex_str + "\\\\" + "\n")
            for result in results:
                latex_str = "%2.2f " % (1 - result[0])
                latex_str += "& %2.2f & %2.2f & %2.2f" % (result[1], result[2], result[3])
                f.write(latex_str + "\\\\" + "\n")

    # Reverse results.
    eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
    eps[eps == -1] = 0
    eps = 1 - eps

    # Efficiency - success.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plot_smooth(ax, eff, acc, n_bins=n_bins, name="Conformal", lowauc=0.0, highauc=1.0, pre_zero=0., post_one=1.)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Predictive Efficiency")

    plt.legend()

    name = "eff_suc.png"
    out_path = os.path.join(res_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()

    # Epsilon - efficiency.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plot_smooth(ax, eps, eff, n_bins=n_bins, name="Conformal", lowauc=0.0, highauc=1.0, pre_zero=0., post_one=1.)

    ax.set_ylabel("Predictive Efficiency")
    # ax.set_xlabel("1 - %s" % u"\u03B5")
    ax.set_xlabel("$1 - \epsilon$")

    plt.legend()
    name = "eps_eff.png"
    out_path = os.path.join(res_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()

    # Epsilon - Success.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.plot([0, 1], [0, 1], color="k", linestyle="dashed", label="diagonal")
    plot_smooth(ax, eps, acc, n_bins=n_bins, name="Conformal", lowauc=0.0, highauc=1.0,
                pre_zero=0., post_one=1.)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("$1 - \epsilon$")

    plt.legend(loc="lower right")
    name = "eps_suc.png"
    out_path = os.path.join(res_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()

    # Epsilon - cost.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plot_smooth(ax, eps, cost, n_bins=n_bins, name="Conformal", lowauc=0.0, highauc=1.0,
                pre_zero=min([x[0] for x in cost]), post_one=max([x[0] for x in cost]))

    ax.set_ylabel("Amortized Cost")
    ax.set_xlabel("$1 - \epsilon$")

    plt.legend(loc="lower right")
    name = "eps_cost.png"
    out_path = os.path.join(res_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()


def plot_with_baselines(res_dirs, baseline_dir, n_bins=100, dest_dir=''):
    '''res_dirs = [CP, Min CP]'''
    task = get_task(res_dirs[-1])

    baselines = get_baselines(baseline_dir)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    def plot_eff(a):
        linecycler = cycle(LINES)
        colorcycler = cycle(COLORS)
        for eps, eff, acc, metric in baselines:
            if metric == 'end_logit':
                metric = 'sum'
            eps, eff, acc = np.array(eps), np.array(eff), np.array(acc)
            eps[eps == -1] = 0
            eps = 1 - eps
            metric_str = ",".join([METRIC_ABBRV[task][m] if m in METRIC_ABBRV[task]
                                   else m for m in metric.split(",")])
            plot_smooth(a, eps, eff, n_bins=FLAGS.n_bins_eff_suc, name=metric_str, lowauc=0.0, highauc=1.0,
                        pre_zero=0., post_one=1., line_style=next(linecycler), color=next(colorcycler))

        # Plot only last layer of cascade + single best non-cascade.
        res_dir = res_dirs[0]
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps[eps == -1] = 0
        eps = 1 - eps
        plot_smooth(a, eps, eff, n_bins=n_bins, name="CP", lowauc=0.0, highauc=1.0,
                    pre_zero=0., post_one=1., line_style=SINGLE_LINE, color=SINGLE_COLOR)

        res_dir = res_dirs[1]
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps[eps == -1] = 0
        eps = 1 - eps
        plot_smooth(a, eps, eff, n_bins=n_bins, name="Min CP", lowauc=0.0, highauc=1.0,
                    pre_zero=0., post_one=1., line_style=OURS_LINE, color=OURS_COLOR)

    plot_eff(ax)
    ax.set_ylabel("Predictive Efficiency")
    ax.set_xlabel("$1 - \epsilon$")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc="upper left")

    if FLAGS.x_lim_eff is not None and task == "qa":
        axins = zoomed_inset_axes(ax, 0.8 * FLAGS.zoom_factor, loc='center', bbox_to_anchor=(0.5, 0.3), bbox_transform=ax.transAxes)
        plot_eff(axins)
        x1, x2, y1, y2 = FLAGS.x_lim_eff, 1, 0, FLAGS.y_lim_eff  # specify the limits
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks(np.around(np.arange(FLAGS.x_lim_eff, 1.001, 0.05), 2))
        axins.set_xticklabels(np.around(np.arange(FLAGS.x_lim_eff, 1.001, 0.05), 2), fontsize=12)
        axins.set_yticks(np.around(np.arange(0., FLAGS.y_lim_eff + 0.01, 0.1), 2))
        axins.set_yticklabels(np.around(np.arange(0., FLAGS.y_lim_eff + 0.01, 0.1), 2), fontsize=12)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    name = "eps_eff_break_baseline.png"
    out_path = os.path.join(dest_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    def plot_acc(a):
        linecycler = cycle(LINES)
        colorcycler = cycle(COLORS)
        for eps, eff, acc, metric in baselines:
            eps, eff, acc = np.array(eps), np.array(eff), np.array(acc)
            eps[eps == -1] = 0
            eps = 1 - eps
            metric_str = ",".join([METRIC_ABBRV[task][m] if m in METRIC_ABBRV[task]
                                   else m for m in metric.split(",")])
            plot_smooth(a, eps, acc, n_bins=FLAGS.n_bins_eff_suc, name=metric_str, lowauc=0.0, highauc=1.0,
                        pre_zero=0., post_one=1., line_style=next(linecycler), color=next(colorcycler))

        # Plot only last layer of cascade + single best non-cascade.
        res_dir = res_dirs[0]
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps[eps == -1] = 0
        eps = 1 - eps
        plot_smooth(a, eps, acc, n_bins=n_bins, name="CP", lowauc=0.0, highauc=1.0,
                    pre_zero=0., post_one=1., line_style=SINGLE_LINE, color=SINGLE_COLOR)

        res_dir = res_dirs[1]
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps[eps == -1] = 0
        eps = 1 - eps
        plot_smooth(a, eps, acc, n_bins=n_bins, name="Min CP", lowauc=0.0, highauc=1.0,
                    pre_zero=0., post_one=1., line_style=OURS_LINE, color=OURS_COLOR)

    plot_acc(ax)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("$1 - \epsilon$")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc="lower right")

    name = "eps_suc_break_baseline.png"
    out_path = os.path.join(dest_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()


def plot_cascade_efficiency_breakdown(res_dirs, metrics, n_bins=100):
    """Plot breakdown by cascade configuration."""
    task = get_task(res_dirs[-1])

    # Plot epsilon vs. efficiency.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(-0.01, 1.01)

    linecycler = cycle(LINES)
    colorcycler = cycle(COLORS)
    for res_dir, metric in zip(res_dirs, metrics):
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps[eps == -1] = 0
        eps = 1 - eps
        if metric == metrics[-2]:
            line_style = OURS_LINE
            color = OURS_COLOR
        elif metric == metrics[-1]:
            line_style = SINGLE_LINE
            color = SINGLE_COLOR
        else:
            line_style = next(linecycler)
            color = next(colorcycler)
        metric_str = ",".join([METRIC_ABBRV[task][m] if m in METRIC_ABBRV[task]
                               else m for m in metric.split(",")])
        name = metric_str
        plot_smooth(ax, eps, eff, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0, line_style=line_style,
                    color=color, pre_zero=0., post_one=1.)

    ax.set_ylabel("Predictive Efficiency")
    ax.set_xlabel("$1 - \epsilon$")
    plt.xticks(np.arange(0.5, 1.01, 0.1))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend()

    name = "eps_eff_break.png"
    out_path = os.path.join(res_dirs[-2], name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()
        

def plot_equivalence_breakdown(res_dirs, names, n_bins=100):
    """Plot breakdown by equivalence configuration (true vs. false)."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.00, 1.00)

    def plot_all(a):
        linecycler = cycle(LINES)
        colorcycler = cycle(COLORS)
        for res_dir, name in zip(res_dirs, names):
            res_file = get_res_file(res_dir)
            data = json.load(open(res_file, "r"))
            eps, eff, acc, cost = get_conformal(data)
            eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
            eps[eps == -1] = 0
            eps = 1 - eps
            if name == names[-1]:
                line_style = OURS_LINE
                color = OURS_COLOR
            elif name == names[-2]:
                line_style = SINGLE_LINE
                color = SINGLE_COLOR
            else:
                line_style = next(linecycler)
                color = next(colorcycler)
            plot_smooth(a, eps, eff, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0, line_style=line_style,
                        color=color, pre_zero=0., post_one=1.)

    plot_all(ax)

    ax.set_ylabel("Predictive Efficiency")
    ax.set_xlabel("$1 - \epsilon$")
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc="upper left")

    if FLAGS.x_lim_eff is not None:
        axins = zoomed_inset_axes(ax, FLAGS.zoom_factor, loc='center', bbox_to_anchor=(0.4, 0.35), bbox_transform=ax.transAxes)
        plot_all(axins)
        x1, x2, y1, y2 = FLAGS.x_lim_eff, 1, 0, FLAGS.y_lim_eff  # specify the limits
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks(np.around(np.arange(FLAGS.x_lim_eff, 1.001, 0.05), 2))
        axins.set_xticklabels(np.around(np.arange(FLAGS.x_lim_eff, 1.001, 0.05), 2), fontsize=12)
        axins.set_yticks(np.around(np.arange(0., FLAGS.y_lim_eff + 0.01, 0.1), 2))
        axins.set_yticklabels(np.around(np.arange(0., FLAGS.y_lim_eff + 0.01, 0.1), 2), fontsize=12)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    p_name = "eps_eff_min_break.png"
    out_path = os.path.join(res_dirs[-1], p_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.plot([0, 1], [0, 1], color="k", linestyle="dashed", label="diagonal")
    linecycler = cycle(LINES)
    colorcycler = cycle(COLORS)
    for res_dir, name in zip(res_dirs, names):
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps = np.array(eps)
        eps[eps == -1] = 0
        eps = 1 - eps
        if name == names[-1]:
            line_style = OURS_LINE
            color = OURS_COLOR
        elif name == names[-2]:
            line_style = SINGLE_LINE
            color = SINGLE_COLOR
        else:
            line_style = next(linecycler)
            color = next(colorcycler)
        plot_smooth(ax, eps, acc, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0, line_style=line_style,
                    color=color, pre_zero=0., post_one=1.)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("$1 - \epsilon$")

    plt.legend(loc="lower right")

    p_name = "eps_suc_min_break.png"
    out_path = os.path.join(res_dirs[-1], p_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()


def plot_corrections(res_dirs, names, n_bins=100):
    """Plot success vs epsilon by correction."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.plot([0, 1], [0, 1], color="k", linestyle="dashed", label="diagonal")
    linecycler = cycle(LINES)
    colorcycler = cycle(COLORS)
    for res_dir, name in zip(res_dirs, names):
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, eff, acc, cost = get_conformal(data)
        eps, eff, acc, cost = np.array(list(reversed(eps))), list(reversed(eff)), list(reversed(acc)), list(reversed(cost))
        eps[eps == -1] = 0
        eps = 1 - eps
        if name == names[-1]:
            line_style = OURS_LINE
            color = OURS_COLOR
        else:
            line_style = next(linecycler)
            color = next(colorcycler)
        plot_smooth(ax, eps, acc, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0,
                    line_style=line_style, color=color, pre_zero=0., post_one=1.)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("$1 - \epsilon$")
    plt.legend(loc="lower right")

    p_name = "eps_suc_corrections.png"
    out_path = os.path.join(res_dirs[-1], p_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)

def plot_casacde_cost_short(res_dirs, m, n_bins=100):
    """
    Plot minCP vs. CascadedMinCP amortized cost.
    m = depth of cascade
    res_dirs = minCP, CascadedMinCP
    """

    # Epsilon vs. amortized cost.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    #MinCP
    res_file = get_res_file(res_dirs[0])
    data = json.load(open(res_file, "r"))
    eps, _, _, cost = get_conformal(data)
    eps, cost = np.array(list(reversed(eps))), np.array(list(reversed(cost)))
    cost[:, :] = 1
    eps[eps == -1] = 0
    eps = 1 - eps
    line_style = SINGLE_LINE
    color = SINGLE_COLOR

    name = "Min CP"
    pre_zero = 1.0
    post_one = 1.0
    plot_smooth(ax, eps, cost, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0,
                line_style=line_style, color=color, pre_zero=pre_zero, post_one=post_one)

    #CascadedMinCP
    res_file = get_res_file(res_dirs[1])
    data = json.load(open(res_file, "r"))
    eps, _, _, cost = get_conformal(data)
    eps, cost = np.array(list(reversed(eps))), np.array(list(reversed(cost)))
    eps[eps == -1] = 0
    eps = 1 - eps
    line_style = OURS_LINE
    color = OURS_COLOR

    name = "Cascaded Min CP"
    pre_zero = 1.0 / m
    post_one = 1.0
    plot_smooth(ax, eps, cost, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0,
                line_style=line_style, color=color, pre_zero=pre_zero, post_one=post_one)


    ax.set_ylabel("Amortized Cost")
    ax.set_xlabel("$1 - \epsilon$")
    plt.legend(loc="lower right")

    name = "eps_cost_break.png"
    out_path = os.path.join(res_dirs[-1], name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()


    return

def plot_cascade_cost_breakdown(res_dirs, metrics, n_bins=100):
    """Plot breakdown by cascade level w.r.t. amortized cost."""
    task = get_task(res_dirs[0])

    # Epsilon vs. amortized cost.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    linecycler = cycle(LINES)
    colorcycler = cycle(COLORS)
    for res_dir, metric in zip(res_dirs, metrics):
        res_file = get_res_file(res_dir)
        data = json.load(open(res_file, "r"))
        eps, _, _, cost = get_conformal(data)
        eps, cost = np.array(list(reversed(eps))), np.array(list(reversed(cost)))
        if metric == metrics[0]:
            cost[:, :] = 1
        eps[eps == -1] = 0
        eps = 1 - eps
        if metric == metrics[-1]:
            line_style = OURS_LINE
            color = OURS_COLOR
        elif metric == metrics[0]:
            line_style = SINGLE_LINE
            color = SINGLE_COLOR
        else:
            line_style = next(linecycler)
            color = next(colorcycler)
        metric_str = ", ".join([METRIC_ABBRV[task][m] if m in METRIC_ABBRV[task]
                               else m for m in metric.split(",")])
        name = metric_str
        pre_zero = 1.0 / len(metric.split(","))
        post_one = 1.0
        plot_smooth(ax, eps, cost, n_bins=n_bins, name=name, lowauc=0.0, highauc=1.0,
                    line_style=line_style, color=color, pre_zero=pre_zero, post_one=post_one)

    ax.set_ylabel("Amortized Cost")
    ax.set_xlabel("$1 - \epsilon$")
    plt.legend(loc="lower right")

    name = "eps_cost_break.png"
    out_path = os.path.join(res_dirs[-1], name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FLAGS.dpi)
    plt.close()


def main(_):
    # Use default number of bins for effiency vs success if not otherwise specified.
    if FLAGS.n_bins_eff_suc is None:
        FLAGS.n_bins_eff_suc = FLAGS.n_bins

    FLAGS.epsilons = [float(f) for f in FLAGS.epsilons or []]

    # Collect all of the results files that match the filter string.
    dirs = glob.glob(os.path.join(FLAGS.exp_dir, "*{}*/results.json".format(FLAGS.filter)))
    exp_dirs = [os.path.dirname(os.path.abspath(d)) for d in dirs]

    # All directories, not just the filter.
    all_dirs = glob.glob(os.path.join(FLAGS.exp_dir, "**/results.json"))
    all_dirs = [os.path.dirname(os.path.abspath(d)) for d in all_dirs]

    # Get the baseline dirs seperately.
    #if FLAGS.baseline_dir is None:
    #    FLAGS.baseline_dir = os.path.join(
    #        os.path.dirname(FLAGS.exp_dir.rstrip("/")),
    #        "baselines",
    #        os.path.basename(FLAGS.exp_dir.rstrip("/")))
    #baseline_dirs = glob.glob(os.path.join(FLAGS.baseline_dir, "**/results.json"))
    #baseline_dirs = [os.path.dirname(os.path.abspath(d)) for d in baseline_dirs]
    #if baseline_dirs:
    #    baseline_dir = baseline_dirs[0]
    #else:
    #    baseline_dir = ""
    baseline_dir = FLAGS.baseline_dir

    for res_dir in exp_dirs:
        print("Processing {}".format(res_dir))
        try:
            if not FLAGS.skip_single:
                plot_results(res_dir, n_bins=FLAGS.n_bins)

            if FLAGS.breakdown:
                start = res_dir.find("-metrics=") + len("-metrics=")
                end = res_dir.find("-equivalence=")
                metrics = res_dir[start: end].split(",")

                # 1) Tolerance vs. efficiency, comparison of main methods.
                cp_metric = metrics[-1]
                if cp_metric == "end_logit":
                    cp_metric = "sum"
                cp_dir = res_dir[:start] + cp_metric + res_dir[end:]
                start = cp_dir.find("-equivalence=") + len("-equivalence=")
                cp_dir = cp_dir[:start] + "False"
                cp_dir_min = cp_dir[:start] + "True"
                start = res_dir.find("-equivalence=") + len("-equivalence=")
                eq_val = res_dir[start:]
                if eq_val == "True":
                    break_dirs = [cp_dir, res_dir[:start] + "False", cp_dir_min, res_dir]
                    if all([d in all_dirs for d in break_dirs]):
                        plot_equivalence_breakdown(break_dirs,
                                                   ["CP", "Cascaded CP", "Min CP", "Cascaded Min CP"],
                                                   n_bins=FLAGS.n_bins)

                    cost_break_dirs = break_dirs[-2:]
                    # 1.5) Tolerance vs. efficiency AND accuracy vs. efficiency baselines.
                    break_dirs = [break_dirs[0]] + [break_dirs[2]]
                    if baseline_dir:
                        print("Using baseline dir %s" % baseline_dir)
                        if all([d in all_dirs for d in break_dirs]):
                            plot_with_baselines(break_dirs, baseline_dir, n_bins=FLAGS.n_bins, dest_dir=res_dir)
                    else:
                        print("No baseline dir")

                # 2) Tolerance vs. efficiency AND accuracy, breakdown by cascade.
                start = res_dir.find("-metrics=") + len("-metrics=")
                end = res_dir.find("-equivalence=")
                metrics_break = [",".join(metrics[:i]) for i in range(1, len(metrics) + 1)]
                metrics_break += [metrics[-1]]
                break_dirs = [res_dir[:start] + m + res_dir[end:] for m in metrics_break]
                if len(break_dirs) > 0 and all([d in all_dirs for d in break_dirs]):
                    plot_cascade_efficiency_breakdown(break_dirs, metrics_break,
                                                      n_bins=FLAGS.n_bins)

                # 3) Tolerance vs. amortized cost, breakdown by cascade.

                #metrics_break = list(reversed([",".join(metrics[i:]) for i in range(len(metrics))]))
                #break_dirs = [res_dir[:start] + m + res_dir[end:] for m in metrics_break]
                #if len(break_dirs) > 0 and all([d in all_dirs for d in break_dirs[1:]]):
                #    plot_cascade_cost_breakdown(break_dirs, metrics_break, n_bins=FLAGS.n_bins)
                if all([d in all_dirs for d in cost_break_dirs]):
                    plot_casacde_cost_short(cost_break_dirs, len(metrics), n_bins=FLAGS.n_bins)

            # 4) Finally, compare corrections in terms of tightness.
            if FLAGS.compare_corrections:
                start = res_dir.find("-correction=") + len("-correction=")
                end = res_dir.find("-metrics=")
                corr = res_dir[start:end]
                if not corr.startswith("simes"):
                    continue
                if corr == "ecdf-biased":
                    all_corr = [c if c != "ecdf" else "ecdf-biased" for c in CORRECTIONS]
                else:
                    all_corr = CORRECTIONS
                corr_dirs = [res_dir[:start] + c + res_dir[end:] for c in all_corr]
                if all([d in all_dirs for d in corr_dirs]):
                    print("Comparing corrections")
                    plot_corrections(corr_dirs, CORRECTIONS, n_bins=FLAGS.n_bins)

        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    app.run(main)
