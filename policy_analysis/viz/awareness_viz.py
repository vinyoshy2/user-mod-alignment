import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import pandas as pd

#Generates the plot in A.1.2 of the main paper
def plot_posterior_predictions(loc, gt_proportions, predicted_proportions, id2rule):
    to_plot = {}
    key = 0
    for i in range(0, len(predicted_proportions)):
        for j in range(0, len(predicted_proportions[0])):
            to_plot[key] = {"rule": id2rule[i], "val": np.array(predicted_proportions[i])[j], "Source": "MRP-Adjusted"}
            key+=1
        to_plot[key] = {"rule": id2rule[i], "val": gt_proportions[i], "Source": "Sample Data"}
        key += 1
    plot_df = pd.DataFrame.from_dict(to_plot, orient="index")

    from matplotlib.ticker import MultipleLocator

    sns.set(style="white", font_scale=1.1)

    plt.figure(figsize=(13,8))
    ax = sns.pointplot(x="rule", y="val", hue="Source", palette = ["#4ac6ff", "#ff4a7d"], data=plot_df, errorbar=("pi", 95), linestyles="None")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)

    majorLocator = MultipleLocator(.25)
    minorLocator = MultipleLocator(.05)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)

    ax.tick_params(direction="out", length=4, width=2, colors="k", which="major", bottom=False, left=True)
    ax.tick_params(direction="out", length=2, width=1, colors="k", which="minor", bottom=False, left=True)
    ax.minorticks_on()
    sns.despine( bottom=True, left=False)
    ax.set_ylim([0, 1.1])
    plt.savefig(loc)

#makes some standard graph style changes for the participation rates plots
def set_graph_style(g, maj_amt, min_amt, remove_legend = True):
    majorLocator = MultipleLocator(maj_amt)
    minorLocator = MultipleLocator(min_amt)
    g.yaxis.set_major_locator(majorLocator)
    g.yaxis.set_minor_locator(minorLocator)

    g.tick_params(direction="out", length=4, width=2, colors="k", which="major", bottom=False, left=True)
    g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", bottom=False, left=True)

    g.set_xlabel("")
    g.minorticks_on()

    if remove_legend:
        g.legend().remove()

#plots the histograms of participation data that appear in figure 4 of the main paper
#outputs combined participation DF
def plot_participation(loc, pop_df, scraped):
    
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(10, 15))

    pop_df["Source"] = "Population"
    scraped["Source"] = "Survey"
    all_participation_df = pd.concat([scraped, pop_df])

    sns.set(style="white", font_scale=1.1)
    sns.set_palette(["#ff4a7d", "#4ac6ff"])

    #Generate com_gen plot
#     multiple="dodge",
#, stat="probability"
    g = sns.kdeplot(ax=axes[0, 0], data=all_participation_df, x="com_gen", hue="Source", alpha=.9, common_norm=False)
    g.set_title("Comments on Reddit, Log-Scaled")
    set_graph_style(g, .05, .01, remove_legend = False)
#    g.set_xlim([0,8])
    g.set_ylim([0,.3501])

    # Generate com_sub plot
    h = sns.kdeplot(ax=axes[0, 1], data=all_participation_df, x="com_sub", hue="Source", alpha=.9, common_norm=False)
    h.set_title("Comments on r/CMV, Log-Scaled")
    set_graph_style(h, .05, .01)
    h.set_ylim([0,.501])

    # Generate rem_gen plot
    i = sns.kdeplot(ax=axes[1, 0], data=all_participation_df, x="rem_gen", hue="Source", alpha=.9, common_norm=False)
    i.set_title("Comments Removed from Reddit, Log-Scaled")
    set_graph_style(i, .1, .02)
    i.set_ylim([0, 1.01])

    # Generate rem_sub plot
    j = sns.kdeplot(ax=axes[1, 1], data=all_participation_df, x="rem_sub", hue="Source", alpha=.9, common_norm=False)
    j.set_title("Comments Removed from r/CMV, Log-Scaled")
    set_graph_style(j, 1, .2)
    j.set_ylim([0, 6.01])

    # Generate age plot
    k = sns.kdeplot(ax=axes[2, 0], data=all_participation_df, x="age", hue="Source", alpha=.9, common_norm=False)
    k.set_title("Account Age, Months")
    set_graph_style(k, .01, .002)
    k.set_ylim([0, .021])

    # Generate is_mod plot
    x="is_mod"
    y="Source"
    new_df = (all_participation_df
              .groupby(y)[x]
              .value_counts(normalize=True)
              .rename("Probability")
              .reset_index())
    l = sns.barplot(ax=axes[2, 1], data=new_df, x="is_mod", y="Probability", hue="Source", alpha=.9)
    l.set_title("Mod Status")
    set_graph_style(l, .2, .04)
    l.set_xticklabels(["Not a Mod", "Is a Mod"])
    l.set_ylim([0, .801])

    sns.despine(trim=True, bottom=True, left=False, offset=5)
    plt.savefig(loc)
    
    return all_participation_df
