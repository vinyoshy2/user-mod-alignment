import jax.numpy as jnp
from jax import random, vmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
import xarray 
import matplotlib.gridspec as gridspec
from jax.scipy.special import expit
import seaborn as sns
import arviz as az
import numpyro
numpyro.set_platform("cpu")
from jax.config import config
config.update("jax_enable_x64", True)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from user_mod_bars import gen_bars

rule_map = {
    0: "User Survey Comments",
    1: "Rule 1 (Must Challenge OP)",
    2: "Rule 2 (Rudeness/Hostility)",
    3: "Rule 3 (Bad Faith Accusation)",
    4: "Rule 4 (Delta Abuse)",
    5: "Rule 5 (Off-Topic)"
}



#palettes for clear bars (ground truth) and colored bars (modeled outcomes)
palette_clear=["#ffffff", "#ffffff", "#ffffff"]
palette_colored = ["#ff4a7d", "#4ac6ff", "#fee143"]

#generates the posterior predictive plots for a single rule and decision type
def posterior_predictive_plots(dataset, inf_data, ax, rule, type1):
    #map inputted decision type to index in array of posterior samples
    type2ind = {
        "user_perception": 0,
        "user_opinion": 1,
        "user_application": 2,
        "irl": 3,
        "m_survey": 4
    }
    
    type1_ind = type2ind[type1]    
    rng_key = random.PRNGKey(1)
    rng_key, rng_key_ = random.split(rng_key)

    #draw from posterior
    posterior_samples = inf_data["posterior"]

    num_comments = max(dataset[(dataset["user"] == True) & (dataset["rule"] == rule)].comment_index.values)
    
    #extract relevant parameters from posterior -- special handling for mod survey comments
    if rule == 0:
        covs = posterior_samples["cov_0"][0,:,:4,:4]
    else:
        covs = posterior_samples["cov"][0,:,rule-1,:,:]
    mu_rs = posterior_samples["mu_r"][0,:,rule,:]
    user_sigmas = posterior_samples["user_sigma"][0,:,:]

    co_occurences = {}   
    
    #generate ground truth bars for rule/decision type using code from gen_mod_bars.py 
    sample_data, dec_count = gen_bars(dataset, rule, type1, dataset_only=True)

    two_decs = dec_count[1][1]
    
    #for each draw from the posterior, simulate re-running our experiment
    for i in range(0, mu_rs.shape[0]):
        cur_co_occurences = {}
        mu_r = mu_rs[i,:].to_numpy()
        chol_cov = covs[i,:,:].to_numpy()
        cov = np.einsum("ij, kj->ik", chol_cov, chol_cov)
  
        #simulate user biases
        user_slopes1 = np.random.normal(0, user_sigmas[i], size = (3, num_comments))
        user_slopes2 = np.random.normal(0, user_sigmas[i], size = (3, num_comments))
        user_slopes1 = np.concatenate([user_slopes1, np.zeros((1, num_comments))])
        user_slopes2 = np.concatenate([user_slopes2, np.zeros((1, num_comments))])
        
        #generate latent continuous values twice -- once for each decision per comment
        tendency = random.multivariate_normal(rng_key_, np.expand_dims(mu_r, axis=0), cov, shape=(num_comments,))
        tendency1 =  tendency + user_slopes1.T
        tendency2 = tendency + user_slopes2.T
        tend1 = tendency1[:,type1_ind]
        tend2 = tendency2[:,type1_ind]
        
        #map latent values to probabilities
        p_remove1 = expit(tend1)
        p_remove2 = expit(tend2)
        dec1 = np.random.binomial(n=1, p=p_remove1)
        dec2 = np.random.binomial(n=1, p=p_remove2)
        
        # get user-suggested removal count per comment
        dec_agg = dec1 + dec2
        dec_final = dec_agg[:two_decs]

        #simulate IRL decisions for each comment
        irl = tendency[:,type2ind["irl"]]
        pirl_remove = expit(irl)
        mod_dec = np.random.binomial(n=1, p=pirl_remove)
        
        #count how often different user-suggested removal counts co-occur with IRL mod removals/approvals
        for i in range(0, dec_final.shape[0]):
            pair = (dec_final[i], mod_dec[i])
            if pair not in cur_co_occurences:
                cur_co_occurences[pair] = 0
            cur_co_occurences[pair] += 1
        for key in cur_co_occurences.keys():
            if key not in co_occurences:
                co_occurences[key] = []
            co_occurences[key].append(cur_co_occurences[key])
    
    #reformat data for seaborn
    x = [[key[1] for elem in co_occurences[key]] for key in co_occurences.keys()]
    y = [co_occurences[key] for key in co_occurences.keys()]
    hue = [[key[0] for elem in co_occurences[key]] for key in co_occurences.keys()]
    x = [item for sublist in x for item in sublist]
    y = [item for sublist in y for item in sublist]
    hue = [item for sublist in hue for item in sublist]

    #plot simulated results     
    chart = sns.barplot(ax=ax, x=x, y=y, hue=hue, errorbar=("pi", 95), palette=palette_colored)
    #plot ground truth data
    g = sns.barplot(ax=ax, x ="Removal Status", y="frequency", hue="# Users Supporting Removal", data=sample_data,
                    palette=palette_clear, edgecolor="black", alpha=.3, linewidth=2)
    ax.set_title("Co-occurences of " + type1 + " with moderator decisions for Rule " + str(i))
    chart.set(title=rule_map[rule])

#Generates the posterior predictive plots in section A4 of the main paper
def plot_all(dataset, inf_data, dec_type, loc):
    #map decision types to proper names
    type2name = {
        "user_perception": "Predictions",
        "user_opinion": "Opinions",
        "user_application": "Rule Applications"
    }
    type2short = {
        "user_perception": "perc",
        "user_opinion": "op",
        "user_application": "app"
    }
    proper_name = type2name[dec_type]
    #create full figure 
    fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Posterior Predictive Simulations of User {} against Moderator Decisions'.format(proper_name),
                 y=1.05)
    sns.set(style="white")
    #for each rule, generates predictive plot
    for i in range(0,6):
        row = i // 3
        column = i % 3
        posterior_predictive_plots(dataset, inf_data, axes[row][column], rule=i, type1=dec_type)
    plt.savefig("{}/posterior-user-mod-bars-{}.pdf".format(loc, type2short[dec_type]))
