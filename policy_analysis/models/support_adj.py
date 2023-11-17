import jax.numpy as jnp
from jax import random, vmap
import pandas as pd
import numpyro
import numpy as np
import xarray 
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro import deterministic
from jax.scipy.special import expit
import arviz as az
numpyro.set_platform("cpu")
from jax.config import config
config.update("jax_enable_x64", True)
pd.options.mode.chained_assignment = None

#Model described in section 4.3.2 and A2 of the main paper.
def mrp_agreement_scraped_only(scraped_dataset, self_report_dataset, predictor_bin_info, predictors, outcomes_scraped, outcomes_self_report):

    # use dirichlet-based prior for cutpoints from:
    # https://betanalpha.github.io/assets/case_studies/ordinal_regression.html#3_Cut_to_the_Chase
    with numpyro.handlers.reparam(config={"cutpoints": numpyro.infer.reparam.TransformReparam()}):
        cutpoints = numpyro.sample(
            "cutpoints",
            dist.TransformedDistribution(
                dist.Dirichlet(np.array([1,1,1,1,1])),
                dist.transforms.SimplexToOrderedTransform(0),
            ),
        )
        
    # draw slopes for each of the adjustment variables (one per rule)
    slope_map = {}
    base_bar = numpyro.sample("base_bar", dist.Normal(0, 1))
    base_std = numpyro.sample("base_std", dist.HalfNormal(1))
    base_z = numpyro.sample("base_z", dist.Normal(0, 1).expand([5]))
    base_slope = numpyro.deterministic("base_slope", base_bar + base_std*base_z)

    com_gen_bar = numpyro.sample("com_gen_bar", dist.Normal(0, 1))
    com_gen_z = numpyro.sample("com_gen_z", dist.Normal(0, 1).expand([5]))
    com_gen_std = numpyro.sample("com_gen_std", dist.HalfNormal(1))
    com_gen_slope = numpyro.deterministic("com_gen_slope", com_gen_bar + com_gen_std*com_gen_z)
    slope_map["com_gen"] = com_gen_slope #store for use in self report data

    rem_gen_bar = numpyro.sample("rem_gen_bar", dist.Normal(0, 1))
    rem_gen_z = numpyro.sample("rem_gen_z", dist.Normal(0, 1).expand([5]))
    rem_gen_std = numpyro.sample("rem_gen_std", dist.HalfNormal(1))
    rem_gen_slope = numpyro.deterministic("rem_gen_slope", rem_gen_bar + rem_gen_std*rem_gen_z)
    slope_map["rem_gen"] = rem_gen_slope

    rem_sub_bar = numpyro.sample("rem_sub_bar", dist.Normal(0, 1))
    rem_sub_z = numpyro.sample("rem_sub_z", dist.Normal(0, 1).expand([5]))
    rem_sub_std = numpyro.sample("rem_sub_std", dist.HalfNormal(1))
    rem_sub_slope = numpyro.deterministic("rem_sub_slope", rem_sub_bar + rem_sub_std*rem_sub_z)
    slope_map["rem_sub"] = rem_sub_slope

    com_sub_bar = numpyro.sample("com_sub_bar", dist.Normal(0, 1))
    com_sub_z = numpyro.sample("com_sub_z", dist.Normal(0, 1).expand([5]))
    com_sub_std = numpyro.sample("com_sub_std", dist.HalfNormal(1))
    com_sub_slope = numpyro.deterministic("com_sub_slope", com_sub_bar + com_sub_std*com_sub_z)
    slope_map["com_sub"] = com_sub_slope

    age_bar = numpyro.sample("age_bar", dist.Normal(0, 1))
    age_z = numpyro.sample("age_z", dist.Normal(0, 1).expand([5]))
    age_std = numpyro.sample("age_std", dist.HalfNormal(1))
    age_slope = numpyro.deterministic("age_slope", age_bar + age_std*age_z)
    slope_map["age"] = age_slope

    mod_bar_0 = numpyro.sample("mod_bar_0", dist.Normal(0, 1))
    mod_z_0 = numpyro.sample("mod_z_0", dist.Normal(0, 1).expand([5, 1]))
    mod_slope_0 = numpyro.deterministic("mod_slope_0", mod_bar_0 + mod_z_0)

    mod_bar_1 = numpyro.sample("mod_bar_1", dist.Normal(0, 1))
    mod_z_1 = numpyro.sample("mod_z_1", dist.Normal(0, 1).expand([5, 1]))
    mod_slope_1 = numpyro.deterministic("mod_slope_1", mod_bar_1 + mod_z_1)
   
    mod_slope = jnp.stack([mod_slope_0, mod_slope_1], axis=1)[:,:,0]

    rule_indices = scraped_dataset.rule.values.astype(int)

    #outcomes for users who allowed us to scrape their profiles for participation data    
    predictions = numpyro.sample("scraped_predictions", dist.OrderedLogistic(predictor=base_slope[rule_indices] + com_gen_slope[rule_indices]*scraped_dataset.com_gen.values.astype(float) + \
                rem_gen_slope[rule_indices]*scraped_dataset.rem_gen.values.astype(float) + com_sub_slope[rule_indices]*scraped_dataset.com_sub.values.astype(float) + \
                rem_sub_slope[rule_indices]*scraped_dataset.rem_sub.values.astype(float) + age_slope[rule_indices]*scraped_dataset.age.values.astype(float) + \
                mod_slope[rule_indices, scraped_dataset.is_mod.values.astype(int)], cutpoints=cutpoints), obs=outcomes_scraped )

    
    if self_report_dataset is not None and len(self_report_dataset) > 0:

        self_report_rule_indices = self_report_dataset.rule.values.astype(int)
        predictor_sums = []
        #generate latent value arrays for each adjustment variable
        for predictor in predictors:
            counts = predictor_bin_info[predictor]["counts"]
            type2endpoints = predictor_bin_info[predictor]["type2endpoints"]
            num_bins = max(type2endpoints.keys()) + 1
            max_length = max(counts.values())
            arrays = []
            #iterate over the different bins associated with the adjustment variable
            for bin in range(num_bins):
                endpoints = type2endpoints[bin]
                cur_count = counts[bin]
                #cases where we deterministic values (e.g. 0 comment removals on r/ChangeMyView)
                if type(endpoints) is not tuple:
                    arrays.append(jnp.concatenate([jnp.full((1, cur_count), endpoints), jnp.zeros((1, max_length-cur_count))], axis=1))
                #cases where we have a lower bound (e.g. 5+ comments removed from r/ChangeMyView)
                elif len(endpoints) == 1:
                    latent_vals = numpyro.sample("{}_{}".format(predictor, endpoints), dist.Exponential(1).expand([1, cur_count]))
                    latent_vals = latent_vals + endpoints[0]
                    arrays.append(jnp.concatenate([latent_vals, jnp.zeros((1, max_length-cur_count))], axis=1))
                #all other cases where we have two defined endpoints (e.g. 2-4 comments removed from r/ChangeMyView)
                else:
                    latent_vals = numpyro.sample("{}_{}".format(predictor, endpoints), dist.Uniform(endpoints[0], endpoints[1]).expand([1, cur_count]))
                    arrays.append(jnp.concatenate([latent_vals, jnp.zeros((1, max_length-cur_count))], axis=1))                    

            arrays = jnp.concatenate(arrays, axis=0)            
            #multiply by slope for each relevant user
            predictor_sum =  slope_map[predictor][self_report_dataset.rule.values.astype(int)]*arrays[self_report_dataset["{}_type".format(predictor)].values.astype(int),
                                                          self_report_dataset["{}_offset".format(predictor)].values.astype(int)]
            predictor_sums.append(predictor_sum) 

        #outcomes for users who self-reported their participation data -- note that predictor_sums is summed across the different predictor variables for each user
        predictions = numpyro.sample("self_report_predictions", dist.OrderedLogistic(predictor=base_slope[self_report_rule_indices]  +  sum(predictor_sums) + \
                    mod_slope[self_report_rule_indices, self_report_dataset.is_mod.values.astype(int)], cutpoints=cutpoints), obs=outcomes_self_report )

#runs the MRP adjusted policy support model
#standardizes the predictors and does some extra handling for the self-reported values for the adjustment vairables
#(recall these values are interval censored). This handling is described in more detail in adj_model.py in the
#practice_analysis/models/
def run_model(user_dataset, decision_dataset, standardize=True):
    key=lambda x: x[0]
    new_user_dataset = user_dataset.copy()
    user_scraped = new_user_dataset[new_user_dataset["self_report"] == False]
    user_self_report = new_user_dataset[new_user_dataset["self_report"] == True]
    means = {}
    stds = {}
    #compute means/stds for standardization
    means["com_gen"], stds["com_gen"] = np.mean(user_scraped.com_gen.values), np.std(user_scraped.com_gen.values)
    means["rem_gen"], stds["rem_gen"]  = np.mean(user_scraped.rem_gen.values), np.std(user_scraped.rem_gen.values)
    means["com_sub"], stds["com_sub"]  = np.mean(user_scraped.com_sub.values), np.std(user_scraped.com_sub.values)
    means["rem_sub"], stds["rem_sub"]  = np.mean(user_scraped.rem_sub.values), np.std(user_scraped.rem_sub.values)
    means["age"], stds["age"]  = np.mean(user_scraped.age.values), np.std(user_scraped.age.values)
    predictors = ["com_gen", "rem_gen", "com_sub", "rem_sub", "age"]
    if standardize:
        predictor_bin_info = {} 
        for predictor in predictors:
            #standardize non-binned data
            user_scraped[predictor] = (user_scraped[predictor] - means[predictor] ) / stds[predictor]
            endpoint2type = {}
            bin_counts = {}
            type2endpoints = {}
            bin_type = 0
            bin_offsets = []
            bin_types = []
            for index, row in user_self_report.iterrows():
                #standardize endpoint values
                if len(row[predictor]) == 1:
                    user_self_report.at[index, predictor] = ((row[predictor][0] - means[predictor]) / stds[predictor],)
                elif row[predictor][0] == row[predictor][1]:
                    user_self_report.at[index, predictor] = (row[predictor][0] - means[predictor]) / stds[predictor]
                else:
                    new_user_dataset.at[index, predictor] = ((row[predictor][0] - means[predictor]) / stds[predictor],
                                      (row[predictor][1] - means[predictor]) / stds[predictor])
                
                # update info about possible bin types
                cur_endpoint = user_self_report.at[index, predictor]
                if cur_endpoint not in endpoint2type:
                    endpoint2type[cur_endpoint] = bin_type
                    type2endpoints[bin_type] = cur_endpoint
                    bin_counts[bin_type] = 0
                    bin_type += 1

                #map current endpoint to bin type and 
                cur_bin_type = endpoint2type[cur_endpoint]
                bin_types.append(cur_bin_type)
                bin_offsets.append(bin_counts[cur_bin_type])
                bin_counts[cur_bin_type] += 1
            user_self_report[predictor+"_type"] = bin_types
            user_self_report[predictor+"_offset"] = bin_offsets
            predictor_bin_info[predictor] = {"counts": bin_counts, "type2endpoints": type2endpoints}

    scraped_dataset = decision_dataset.join(user_scraped ,on="user", how="inner")
    self_report_dataset = decision_dataset.join(user_self_report ,on="user", how="inner")
    cor_model = MCMC(NUTS(mrp_agreement_scraped_only, target_accept_prob=.8), num_warmup=1000, num_samples=2000, num_chains=4)
    dat_list = dict(
        scraped_dataset = scraped_dataset,
        self_report_dataset = self_report_dataset,
        predictor_bin_info = predictor_bin_info,
        predictors = ["com_gen", "rem_gen", "com_sub", "rem_sub", "age"]
    )
    dat_list["outcomes_scraped"] = scraped_dataset.answer.values.astype(int) - 1
    dat_list["outcomes_self_report"] = self_report_dataset.answer.values.astype(int) - 1

    cor_model.run(random.PRNGKey(2), **dat_list)
    return cor_model, means, stds

#predicts response values for each user in the population set (containing both survey respondents and non-respondents)
def post_stratify(posterior_samples, target_pop):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    posterior_predictive = Predictive(mrp_agreement_scraped_only, posterior_samples)
    dat_list = dict(
        scraped_dataset = target_pop,
        self_report_dataset = None,
        predictor_bin_info = None,
        predictors = ["com_gen", "rem_gen", "com_sub", "rem_sub", "age"]
    )
    dat_list["outcomes_scraped"] = None
    dat_list["outcomes_self_report"] = None

    results = posterior_predictive(rng_key_, **dat_list)
    return results
    
#predicts response values for each user in the population set (containing both survey respondents and non-respondents)    
def predicted_ratings(cor_model, decision_dataset, standardized_pop_df):
    gt_proportions = []
    for i in range(5):
        cur_ratings = []
        answers = decision_dataset[decision_dataset["rule"] == i].answer.values.astype(int)
        unique, counts = np.unique(answers, return_counts=True)
        for j in range(0,5):
            cur_ratings.append(counts[j] / len(answers))
        gt_proportions.append(cur_ratings)
    predicted_proportions = []
    for i in range(5):
        predicted_proportions.append([])
        standardized_pop_df["rule"] = i
        answers = post_stratify(cor_model.get_samples(), standardized_pop_df)["scraped_predictions"]
        for j in range(0, 2000):
            cur_ratings = []
            unique, counts = np.unique(answers[j], return_counts=True)
            for k in range(0,5):
                cur_ratings.append(counts[k] / len(answers[k]))
            predicted_proportions[-1].append(cur_ratings)
    return gt_proportions, predicted_proportions    
    

