import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

rule_map = {
    0: "User Survey Comments",
    1: "Rule 1 (Must Challenge OP)",
    2: "Rule 2 (Rudeness/Hostility)",
    3: "Rule 3 (Bad Faith Accusation)",
    4: "Rule 4 (Delta Abuse)",
    5: "Rule 5 (Off-Topic)"
}

long2short = {
    "user_opinion": "op",
    "user_application": "app",
    "user_perception": "perc"
}

# for a particular rule and decision type, plots of the frequency of the number of user-supplied positive labels
# grouped by the IRL decision label.
def gen_bars(dataset, rule, decision_type = "user_opinion", ax=None, dataset_only=False):

    user_column = decision_type
    seen_comments = {}
    subset = dataset[(dataset["user"] == True)  & (dataset["rule"] == rule)]

    decs = []
    #iterate over user_decisions
    for index, row in subset.iterrows():
        _id = row["_id"]
        
        #skip if we've seen this comment in a different user decision
        if _id in seen_comments:
            continue

        #find IRL mod decisions associated with comment
        rating_type = "mod decision"
        mod_decisions = dataset[(dataset["user"] == False) &  (dataset["survey"] == False) & (dataset["_id"] == _id)]
        mod_decision = 0
        for index, row in mod_decisions.iterrows():
            mod_decision = int(row["mod_decision"])
        user_removal_count = 0
        #get all user decisions associated with a comment
        user_decisions = dataset[(dataset["user"] == True) & (dataset["_id"] == _id)]
        dec = 0
        for index, row in user_decisions.iterrows():
            if int(row[user_column]) == 2:
                user_removal_count += 1
            dec+=1
        decs.append(dec)
        if dec >= 3 or dec == 1: #note that a handful of comments were shown to three or more users or only one user. we avoid visualizing these for simplicity.
            continue        
        #update seen_comments entry appropriately
        seen_comments[_id] = {
            "Removal Status": "Removed" if int(mod_decision) else "Approved",
            "# Users Supporting Removal": user_removal_count, 
            "occ": 1
        }
        
    dec_count = np.unique(np.array(decs), return_counts=True)
    cur_data = pd.DataFrame.from_dict(seen_comments, orient="index")
    
    #count occurences of each removal_count/mod decision pairing (these end up being the bar heights)
    cur_data = cur_data.groupby(['Removal Status', "# Users Supporting Removal"]).agg(frequency=("occ", 'sum'))
    cur_data = cur_data.reset_index()
    #if we don't want a plot to be generated
    if dataset_only:
        return cur_data, dec_count
    #generate plot
    if ax != None:
        ax.grid()
        chart = sns.barplot(ax=ax, x ="Removal Status", y="frequency", hue="# Users Supporting Removal", data=cur_data)
    else:
        chart = sns.barplot(x ="Removal Status", y="frequency", hue="# Users Supporting Removal", data=cur_data)
    chart.set(title=rule_map[rule])

    majorLocator = MultipleLocator(25)
    minorLocator = MultipleLocator(5)        
    chart.yaxis.set_major_locator(majorLocator)
    chart.yaxis.set_minor_locator(minorLocator)
    chart.minorticks_on()
    chart.tick_params(direction="out", length=4, width=2, colors="k", which="major", left=True, bottom=False)
    chart.tick_params(direction="out", length=2, width=1, colors="k", which="minor", left=True, bottom=False)
    
#Generates the part of the figures in section A4 of the main paper
#decision type should be one of user_opinion, user_perception, user_application
def gen_user_mod_bars(dataset, decision_type, loc):
    sns.set_palette(["#ff4a7d", "#4ac6ff", "#fde541"])
    sns.set_style("white", {'axes.grid': False})
    #Each subplot will correspond to a particular rule
    fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)
    #convert decision type to two capitalized words	
    capped = decision_type.split("_")
    title_decision_type = "{}{} {}{}".format(capped[0][0].upper(), capped[0][1:], capped[1][0].upper(), capped[1][1:]) 
    fig.suptitle('Co-occurences of {} Ratings with IRL Moderator Decisions'.format(title_decision_type), y=1.05)
    for i in range(0,6):        
        row = i // 3
        column = i % 3
        gen_bars(dataset, i, decision_type, axes[row][column])
        if i != 0:
            legend = axes[row][column].get_legend()
            if legend != None:
                axes[row][column].get_legend().remove()
        if not (column == 0 or row == 0):
            axes[row][column].set_ylabel("")
            axes[row][column].set_xlabel("")
        axes[row][column].grid(False)
               
    fig.tight_layout()
    sns.despine(left=True, bottom=False)
    sns.set_palette(["#ff4a7d", "#4ac6ff", "#fde541"])
    sns.set_style("white", {'axes.grid': False})
    plt.savefig("{}/{}-mod-bars.pdf".format(loc, long2short[decision_type]))


