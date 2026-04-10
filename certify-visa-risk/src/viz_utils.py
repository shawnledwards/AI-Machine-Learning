"""
Reusable plotting helpers for the VisaRisk project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined on a shared x-axis.

    data: dataframe
    feature: dataframe column name
    figsize: size of figure (default (12,7))
    kde: whether to show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with count or percentage labels on each bar.

    data: dataframe
    feature: dataframe column name
    perc: display percentages instead of counts (default False)
    n: show only the top n categories (default None = all)
    """
    total = len(data[feature])
    count = data[feature].nunique()
    figsize = (n + 1, 5) if n is not None else (count + 1, 5)
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        hue=feature,
        legend=False,
        order=data[feature].value_counts().index[:n].sort_values(),
    )
    for p in ax.patches:
        label = (
            "{:.1f}%".format(100 * p.get_height() / total) if perc else p.get_height()
        )
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.show()


def stacked_barplot(data, predictor, target):
    """
    Print crosstab counts and plot a normalized stacked bar chart.

    data: dataframe
    predictor: independent variable column name
    target: target variable column name
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(loc="lower left", frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def distribution_plot_wrt_target(data, predictor, target):
    """
    2x2 grid: histplots split by target class (top) and boxplots w/ and w/o outliers (bottom).

    data: dataframe
    predictor: feature column name
    target: target column name
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor, kde=True, ax=axs[0, 0], color="teal", stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor, kde=True, ax=axs[0, 1], color="orange", stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(
        data=data, x=target, y=predictor, ax=axs[1, 0],
        palette="gist_rainbow", hue=target, legend=False,
    )

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data, x=target, y=predictor, ax=axs[1, 1],
        showfliers=False, palette="gist_rainbow", hue=target, legend=False,
    )

    plt.tight_layout()
    plt.show()
