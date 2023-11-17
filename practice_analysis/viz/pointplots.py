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

#generates figure 9b in the main paper -- expects posterior draws of the correlations between the latent 
#probability of removal and the actual moderation decision made for each rule/decision-making context    
def plot_corrs(empirical_corrs, loc):

    #convert into xarrays
    msurvey_MIRL = [[empirical_corrs["msurvey_MIRL"][k][0] for k in range(3000)]]
    msurvey_MIRL = xarray.DataArray(msurvey_MIRL, name="", coords={"chain": range(1), "draw": range(3000)})
    msurvey_MIRL = msurvey_MIRL.dropna("draw")

    perc_MIRL = [[empirical_corrs["perc_MIRL"][k] for k in range(3000)]]
    perc_MIRL = xarray.DataArray(perc_MIRL, name="", coords={"chain": range(1), "draw": range(3000),
     "rule": range(0,6)})
    perc_MIRL = perc_MIRL.dropna("draw")

    app_MIRL = [[empirical_corrs["app_MIRL"][k] for k in range(3000)]]
    app_MIRL = xarray.DataArray(app_MIRL, name="", coords={"chain": range(1), "draw": range(3000),
     "rule": range(0,6)})
    app_MIRL = app_MIRL.dropna("draw")

    op_MIRL = [[empirical_corrs["op_MIRL"][k] for k in range(3000)]]
    op_MIRL = xarray.DataArray(op_MIRL, name="", coords={"chain": range(1), "draw": range(3000),
     "rule": range(0,6)})
    op_MIRL = op_MIRL.dropna("draw")
    
    #set up fig
    sns.set(style="white", font_scale=1.5)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6,15))
    palette = ["#ff4a7d", "#4ac6ff", "#fee143", "#d3d3d3"]
    
    #plot alignments for each rule
    for i in range(0, 6):
        ax = axes[i]
        ax.set_xlim([0,1.01])
        cur_app_MIRL = app_MIRL.sel(chain=0, rule=i)
        cur_op_MIRL = op_MIRL.sel(chain=0, rule=i)
        cur_perc_MIRL = perc_MIRL.sel(chain=0, rule=i)
                
        if i == 0:
            sns.set_palette(palette)
            ax.set_title("Mod Survey Comments", fontsize=16)
            
            cur_msurvey_MIRL = msurvey_MIRL.sel(chain=0) # user survey comments have an extra measure
            df = pd.DataFrame.from_dict({
                "type": ["User Opinion"]*3000  + ["User Prediction"]*3000 + ["User Application"]*3000 + \
                        ["Moderators(Survey)"]*3000 ,
                "correlation": np.concatenate([cur_op_MIRL, cur_perc_MIRL, cur_app_MIRL, cur_msurvey_MIRL],
                                                axis=0)},
                orient="columns")
        else:
            sns.set_palette(palette[:-1]) #don't need full palette for rules with 3 measures to plot
            ax.set_title("Rule " + str(i), fontsize=16)
            df = pd.DataFrame.from_dict({
                "type": ["User Opinion"]*3000  + ["User Prediction"]*3000 + ["User Application"]*3000 ,
                "correlation": np.concatenate([cur_op_MIRL, cur_perc_MIRL, cur_app_MIRL], axis=0)},
                orient="columns")

        g = sns.pointplot(ax=ax, y="type", hue="type", x="correlation", data = df, linestyles="None", errorbar=("pi", 95), capsize=.25)
        ax.get_yaxis().set_ticks([])
        h,l = ax.get_legend_handles_labels()
        # keep same handles, edit labels with names of choice
        ax.legend(handles=h, labels=['User Opinion', 'User Prediction', "User Application", "Survey Mod Opinion"])
        ax.legend()
        
        # only last subsection should contain legend
        if i != 0:
            legend = ax.get_legend()
            if legend != None:
                ax.get_legend().remove()
        else:
            ax.legend(prop={'size': 10})
        ax.set_ylabel("")
        ax.set_xlabel("")
        #only last subsection should contain x-axis label
        if i == 5:
            ax.set_xlabel("Correlation", fontsize=16)
        #modify ticks
        majorLocator = MultipleLocator(.2)
        minorLocator = MultipleLocator(.05)
        g.xaxis.set_major_locator(majorLocator)
        g.xaxis.set_minor_locator(minorLocator)
        g.minorticks_on()
        g.tick_params(direction="out", length=4, width=2, colors="k", which="major", left=False, bottom=True)
        g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", left=False, bottom=True)
    #final fig edits
    sns.despine(offset=5, left=True, bottom=False)
    fig.tight_layout()
    #save to file
    plt.savefig(loc)

#generates figure 9a in the main paper -- expects posterior draws of the mean latent affinity
# for removal for each rule/decision-making context    
def plot_means(empirical_means, loc):
    #convert to xarrays
    survey = [empirical_means["m_survey"]]
    survey = xarray.DataArray(survey, name="", coords={"chain": range(1), "draw": range(3000)})

    perc = [[empirical_means["perc"][k] for k in range(3000)]]

    perc = xarray.DataArray(perc, name="", coords={"chain": range(1), "draw": range(3000), "rule": range(0,6)})
    
    op = [[empirical_means["op"][k] for k in range(3000)]]
    op = xarray.DataArray(op, name="", coords={"chain": range(1), "draw": range(3000), "rule": range(0,6)})

    app = [[empirical_means["app"][k] for k in range(3000)]]
    app = xarray.DataArray(app, name="", coords={"chain": range(1), "draw": range(3000), "rule": range(0,6)})

    IRL = [[empirical_means["IRL"][k] for k in range(3000)]]
    IRL = xarray.DataArray(IRL, name="", coords={"chain": range(1), "draw": range(3000), "rule": range(0,6)})

    #set palette
    palette = ["#ff4a7d", "#4ac6ff", "#fee143", "#d3d3d3", "#808080"]
    sns.set(style="white", font_scale=1.5)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6,15))

    #plot each rule separately
    for i in range(0, 6):
        ax = axes[i]
        cur_app = app.sel(chain=0, rule=i)
        cur_op = op.sel(chain=0, rule=i)
        cur_perc = perc.sel(chain=0, rule=i)
        cur_IRL = IRL.sel(chain=0, rule=i)
        ax.set_xlim([-0.01,1.01])

        #special handling for mod survey comments
        if i == 0:
            sns.set_palette(palette)
            ax.set_title("Mod Survey Comments", fontsize=16)
            cur_msurvey = survey.sel(chain=0)

            df = pd.DataFrame.from_dict({"type": ["User Opinion"]*3000  + ["User Prediction"]*3000 + \
                                                ["User Application"]*3000 + ["Moderators (Survey)"]*3000 + \
                                                ["Moderators (IRL)"]*3000 ,
                                         "correlation": np.concatenate([cur_op, cur_perc, cur_app,
                                                                         cur_msurvey, cur_IRL], axis=0)},
                                           orient="columns")
        else:
            sns.set_palette(palette[:-2] + [palette[-1]]) #only use part of palette for most cases
            ax.set_title("Rule " + str(i), fontsize=16)
            df = pd.DataFrame.from_dict({"type": ["User Opinion"]*3000  + ["User Prediction"]*3000  + \
                                                 ["User Application"]*3000 + ["Moderators (IRL)"]*3000,
                                         "correlation": np.concatenate([cur_op, cur_perc, cur_app, cur_IRL],
                                                                          axis=0)},
                                        orient="columns")

        
        g = sns.pointplot(ax=ax, y="type", hue="type", x="correlation", data = df,
                          linestyles="None", errorbar=("pi", 95), capsize=.25)
        ax.get_yaxis().set_ticks([])
        h,l = ax.get_legend_handles_labels()

        # keep same handles, edit labels with names of choice
        ax.legend(handles=h, labels=['User Opinion', 'User Prediction', "User Application",
                                     "Mods (Survey)", "Mods (IRL)"])
        ax.legend()
        #only insert legend on final subplot
        if i != 0:
            legend = ax.get_legend()
            if legend != None:
                ax.get_legend().remove()
        else:
            ax.legend(prop={'size': 10})

        #only label x-axis on final subplot
        ax.set_ylabel("")
        ax.set_xlabel("")
        if i == 5:
            ax.set_xlabel("Probability", fontsize=16)
        
        #modify ticks
        majorLocator = MultipleLocator(.2)
        minorLocator = MultipleLocator(.05)        
        g.xaxis.set_major_locator(majorLocator)
        g.xaxis.set_minor_locator(minorLocator)
        g.minorticks_on()
        g.tick_params(direction="out", length=4, width=2, colors="k", which="major", left=False, bottom=True)
        g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", left=False, bottom=True)
    #final style edits
    sns.despine(offset=5, left=True, bottom=False)
    fig.tight_layout()
    #save and display
    plt.savefig(loc)
    plt.show()    

