import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import pandas as pd

rule2id = {
    "Rule 1 (Doesn't Challenge OP)": 0,
    "Rule 2 (Rude/Hostile)": 1,
    "Rule 3 (Bad Faith Accusation)": 2,
    "Rule 4 (Delta Abuse)": 3,
    "Rule 5 (Off-Topic/Insubstantial)": 4
}

id2rule = {
    0: "Rule 1 (Doesn't Challenge OP)",
    1: "Rule 2 (Rude/Hostile)",
    2: "Rule 3 (Bad Faith Accusation)",
    3: "Rule 4 (Delta Abuse)",
    4: "Rule 5 (Off-Topic/Insubstantial)",
}

# Produces a point plot visualizing the distribution of likert ratings for each rule
# Does not actually appear in the main paper or supplement
def viz_dist(loc, gt_proportions, predicted_proportions):
    sns.set(style="white", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Likert Ratings by Rule', y=1.05)

    to_plot = {}
    key = 0
    for i in range(0, 5):
        row = i // 3
        column = i % 3
        ax = axes[row][column]
        source = predicted_proportions[i]
        gt = gt_proportions[i]
        to_pd = {}
        count =0
        for ratings in source:
            for j in range(5):
                entry = {"rating": j+1, "proportion": ratings[j], "source": "MRP-Estimated"}
                to_pd[count] = entry
                count+=1
        for j in range(5):
            to_pd[count] = {"rating": j+1, "proportion": gt[j], "source": "Sample"}
            count+=1
        to_pd = pd.DataFrame.from_dict(to_pd, orient="index")
        ax.set_title(id2rule[i])
        g = sns.pointplot(ax= ax, x="rating", y="proportion", hue="source", palette = ["#4ac6ff", "#ff4a7d"],  data=to_pd, errorbar=("pi", 95), linestyles="None")

        majorLocator = MultipleLocator(.25)
        minorLocator = MultipleLocator(.05)
        g.yaxis.set_major_locator(majorLocator)
        g.yaxis.set_minor_locator(minorLocator)

        g.tick_params(direction="out", length=4, width=2, colors="k", which="major", bottom=False, left=True)
        g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", bottom=False, left=True)
        g.minorticks_on()
        sns.despine(bottom=True, left=False)
        g.set_ylim([0, 1])
        if i > 0:
            g.set_xlabel("")
            g.set_ylabel("")
            g.legend().remove()

    plot_df = pd.DataFrame.from_dict(to_plot, orient="index")
    plt.savefig(loc)

# Produces a point plot visualizing the average likert rating for each rule
# Appears in A.2.1 in the main paper.
def viz_avg(loc, gt_proportions, predicted_proportions):
    from matplotlib.ticker import MultipleLocator
    sns.set(style="white", font_scale=1.1)
    fig, ax = plt.subplots(1,1)
    ax.set_title('Avg Likert Ratings by Rule', y=1.05)

    to_plot = {}
    key = 0
    to_pd = {}
    count =0
    for i in range(0, 5):
        source = predicted_proportions[i]
        gt = gt_proportions[i]
        for ratings in source:
            avg = 0
            for j in range(5):
                avg += ((j+1)*ratings[j])
            entry = {"Rule": "Rule {}".format(i+1), "avg": avg, "source": "MRP-Estimated"}
            to_pd[count] = entry
            count+=1
        avg = 0
        for j in range(5):
            avg += ((j+1)*gt[j])
        to_pd[count] = {"Rule": "Rule {}".format(i+1), "avg": avg, "source": "Sample"}
        print(avg)
        count+=1
    to_pd = pd.DataFrame.from_dict(to_pd, orient="index")
    g = sns.pointplot(ax= ax, x="Rule", y="avg", hue="source", palette = ["#4ac6ff", "#ff4a7d"],  data=to_pd, errorbar=("pi", 94), linestyles="None")

    majorLocator = MultipleLocator(.25)
    minorLocator = MultipleLocator(.05)
    g.yaxis.set_major_locator(majorLocator)
    g.yaxis.set_minor_locator(minorLocator)

    g.tick_params(direction="out", length=4, width=2, colors="k", which="major", bottom=False, left=True)
    g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", bottom=False, left=True)
    g.minorticks_on()
    sns.despine(bottom=True, left=False)
    g.set_ylim([3, 5])
    if i > 0:
        g.set_xlabel("")
        g.set_ylabel("")
        g.legend().remove()

    plot_df = pd.DataFrame.from_dict(to_plot, orient="index")
    plt.savefig(loc)
