import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

rule_map = {
    0: "Mod Survey Comments",
    1: "Rule 1 (Must Challenge OP)",
    2: "Rule 2 (Rudeness/Hostility)",
    3: "Rule 3 (Bad Faith Accusation)",
    4: "Rule 4 (Delta Abuse)",
    5: "Rule 5 (Off-Topic)"
}

#use mod determines whether to generate a plot for the mod-mod disagreements
def gen_heatmap(dataset, rule, user_column = "user_opinion", ax=None, cbar=True):
    sns.set(font_scale=1.3)
    seen_comments = {}
    subset = dataset[(dataset["user"] == True) & (dataset["rule"] == rule)]
    
    #map comments to list of user-supplied ratings
    for index, row in subset.iterrows():
        idx = row["comment_index"]
        if idx not in seen_comments:
            seen_comments[idx] = []
        seen_comments[idx].append(row[user_column])
        
    #fill lower triangle of heatmap
    heatmap_base = np.array([[0,0,0],[0,0,0],[0,0,0]])
    rating_type = user_column 
    for idx in seen_comments.keys():
        user_ratings = seen_comments[idx]
        if len(user_ratings) > 1: #lower triangle
            smaller = int(min(user_ratings[0] , user_ratings[1])) 
            bigger = int(max(user_ratings[0] , user_ratings[1]))
            heatmap_base[bigger][smaller] += 1  
    #plot heatmap using pre-specified axis if possible
    if ax != None:
        chart = sns.heatmap(ax=ax, vmin=0, vmax=180, data=heatmap_base, cmap = sns.light_palette("#ff4a7d", as_cmap=True), annot=True, fmt="g", 
                           xticklabels = ["Anti-removal", "Unsure", "Pro-removal"], yticklabels = ["Anti-removal", "Unsure","Pro-removal"], cbar=cbar)
    else:
        chart = sns.heatmap(data=heatmap_base, vmin=0, vmax=180, cmap = sns.light_palette("#ff4a7d", as_cmap=True), annot=True, fmt="g",
                                xticklabels = ["0 apply", ">=1 apply"], yticklabels = ["0 apply", ">=1 apply"], cbar=cbar)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 45)
    chart.set(title=rule_map[rule])

#heatmap type must be either: "user_opinion" "user_perception" or "user_application"
#generates the Figure 8 in the main paper plus figures 2 and 3 in the supplement
def gen_all_heatmaps(dataset, heat_map_type, loc):
    pair_count = 0
    uneq_count = 0
    pair_oc = {}
    fig = plt.figure(figsize=(15,10))
    axes = []
    gs = gridspec.GridSpec(4, 6)

    m = 0
    for i in range(0, 4, 2):
        for j in range(0, 6, 2):
            if m < 3:
                ax = plt.subplot(gs[i:i+2, j:j+2])
                axes.append(ax)
                m+=1
            elif m == 3:
                ax = plt.subplot(gs[i:i+2, 1:3])
                axes.append(ax)
                m+= 1
            else:
                ax = plt.subplot(gs[i:i+2, 3:5])
                axes.append(ax)
                break
    for i in range(1,6):
        gen_heatmap(dataset, i, heat_map_type, axes[i-1], cbar=i==1)
        if i != 1:
            legend = ax.get_legend()
            if legend != None:
                ax.get_legend().remove()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    long2short = {
        "user_opinion": "op",
        "user_perception": "perc",
        "user_application": "app"
    }
    plt.savefig("{}/user-{}-heatmap.pdf".format(loc, long2short[heat_map_type]))

