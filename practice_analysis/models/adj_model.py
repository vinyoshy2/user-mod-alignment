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
import json
numpyro.set_platform("cpu")
from jax.config import config
from no_adj_model import matmul_per_rule, outer_per_rule, pb_corr, mapped_corrcoef
config.update("jax_enable_x64", True)
pd.options.mode.chained_assignment = None

#model with repsonse bias adjustments -- described in 4.3.3 and A.3.2   
#predictor_bin_info maps the name of each adjustment variable to a dictionary containing two entries: "counts" and "type2endpoints"
#type2endpoints is a list containing the endpoints for the different intervals for a particular adjustment variable (log-scaled+standardized where appropriate)
#these endpoints are either a tuple of two values (for a proper bin), a tuple of a single value (for a lower bound, e.g. comments more than hourly) or 
#a single value outside of a tuple when the response denotes a fixed value (e.g. has never had a comment removed before)
#counts contains a list denoting how many user responses fall into each bin
def correlation_model_MU_adjusted(dataset, scraped_dataset, self_report_dataset, user_map, mod_survey_map, predictors, predictor_bin_info):

    num_users = len(user_map)
    num_mods_survey = len(mod_survey_map)
    num_comments = [max(dataset[dataset["rule"] == i].comment_index.values)+1 for i in range(0,6)]
    max_length = max(num_comments)

    slope_map = {} #slope names to vectors
    #slopes for adjustment predictor variables
    
    #effects for # general comments
    com_gen_bar = numpyro.sample("com_gen_bar", dist.Normal(0, 1))
    com_gen_z = numpyro.sample("com_gen_z", dist.Normal(0, 1).expand([6, 3]))
    com_gen_slope = numpyro.deterministic("com_gen_slope", com_gen_bar + com_gen_z)
    slope_map["com_gen"] = com_gen_slope

    #effects for # removed general comments
    rem_gen_bar = numpyro.sample("rem_gen_bar", dist.Normal(0, 1))
    rem_gen_z = numpyro.sample("rem_gen_z", dist.Normal(0, 1).expand([6, 3]))
    rem_gen_slope = numpyro.deterministic("rem_gen_slope", rem_gen_bar + rem_gen_z) 
    slope_map["rem_gen"] = rem_gen_slope

    #effects for # removed CMV comments
    rem_sub_bar = numpyro.sample("rem_sub_bar", dist.Normal(0, 1))
    rem_sub_z = numpyro.sample("rem_sub_z", dist.Normal(0, 1).expand([6, 3]))
    rem_sub_slope = numpyro.deterministic("rem_sub_slope", rem_sub_bar + rem_sub_z)
    slope_map["rem_sub"] = rem_sub_slope

    #effects for # CMV comments
    com_sub_bar = numpyro.sample("com_sub_bar", dist.Normal(0, 1))
    com_sub_z = numpyro.sample("com_sub_z", dist.Normal(0, 1).expand([6, 3]))
    com_sub_slope = numpyro.deterministic("com_sub_slope", com_sub_bar + com_sub_z)
    slope_map["com_sub"] = com_sub_slope

    #effects for account age
    age_bar = numpyro.sample("age_bar", dist.Normal(0, 1))
    age_z = numpyro.sample("age_z", dist.Normal(0, 1).expand([6, 3]))
    age_slope = numpyro.deterministic("age_slope", age_bar + age_z)
    slope_map["age"] = age_slope

    #random effects for each user (deviation of each user from global mean) -- same as in no_adj.py
    user_sigma = numpyro.sample("user_sigma", dist.HalfNormal(1).expand([3,1]))
    user_z = numpyro.sample("user_z", dist.Normal(0,1).expand([3, num_users]))
    user_slopes = numpyro.deterministic("user_slopes", user_z*user_sigma)

    mod_bar = numpyro.sample("mod_bar", dist.Normal(0, 1))
    mod_z = numpyro.sample("mod_z", dist.Normal(0, 1).expand([6, 3, 2]))
    is_mod_slope = numpyro.deterministic("mod_slope", mod_bar + mod_z)

    #random effects for each moderator in survey (deviation of each moderator from global mean) -- same as in no_adj.py
    mod_survey_sigma = numpyro.sample("mod_survey_sigma", dist.HalfNormal(.5))
    mod_survey_z = numpyro.sample("mod_survey_z", dist.Normal(0,1).expand([num_mods_survey]))
    mod_survey_slopes = numpyro.deterministic("mod_survey_slopes", mod_survey_z*mod_survey_sigma)

    #rule specific mean for mod_decisions -- same as in no_adj.py (not special handling for mod survey comments) 
    mu_r_bar = numpyro.sample("mu_r_bar", dist.Normal(0,1).expand([6,1]))
    mu_r_sigma = numpyro.sample("mu_r_sigma", dist.HalfNormal(1).expand([6,1]))
    
    mu_r_zs = numpyro.sample("mu_r_zs", dist.Normal(0,1).expand([6, 4]))
    mu_r_z_survey = numpyro.sample("mu_r_z_survey", dist.Normal(0,1))
    mu_r = numpyro.deterministic("mu_r", mu_r_bar + mu_r_sigma * mu_r_zs)
    mu_r_0_survey= numpyro.deterministic("mu_r_0_survey", mu_r_bar[0] + mu_r_sigma[0] * mu_r_z_survey)


    #std for user and mod decisisions -- same as in no_adj.py
    sigma_user = numpyro.sample("sigma_user", dist.HalfNormal(1))
    sigma_users = numpyro.sample("sigma_users", dist.HalfNormal(sigma_user).expand([18]))

    sigma_mod = numpyro.sample("sigma_mod", dist.HalfNormal(1))
    sigma_mods = numpyro.sample("sigma_mods", dist.HalfNormal(sigma_mod).expand([7]))
    sigma = numpyro.deterministic("sigma", jnp.concatenate([jnp.reshape(sigma_users[:15], (5,3)), jnp.reshape(sigma_mods[:5], (5,1))], axis=1))
    sigma_0 = numpyro.deterministic("sigma_0", jnp.concatenate([jnp.reshape(sigma_users[15:], (1,3)), jnp.reshape(sigma_mods[5:], (1,2))], axis=1)[0])

    #covariance matrices 
    chol_0 = numpyro.sample("chol_0", dist.LKJCholesky(5, 1))
    chol_cov_0 = numpyro.deterministic("cov_0", jnp.dot(jnp.diag(sigma_0), chol_0))
    chol_cov_0 = jnp.expand_dims(chol_cov_0, axis=0)
    
    chol = numpyro.sample("chol", dist.LKJCholesky(4, jnp.ones(5)))
    chol_cov = numpyro.deterministic("cov", matmul_per_rule(vmap(jnp.diag, in_axes=0)(sigma), chol))
    chol_cov = jnp.pad(chol_cov, ((0,0), (0, 1), (0, 1)), 'constant', constant_values=0)
    chol_cov = jnp.concatenate([chol_cov_0, chol_cov], axis=0)
    
    # zero pad the z scores for each of the comment slopes so that we can do a single matrix multiplication to compute the actual slope values (for non-centered parameterization)
    tendency_z_0 = numpyro.sample("tendency_z_0", dist.Normal(0,1).expand([5, num_comments[0]]))    
    tendency_z_0 = jnp.pad(tendency_z_0, ((0,0), (0, max_length - num_comments[0])), 'constant', constant_values=0)
    
    tendency_z_1 = numpyro.sample("tendency_z_1", dist.Normal(0,1).expand([4, num_comments[1]]))
    tendency_z_1 = jnp.pad(tendency_z_1, ((0,1), (0, max_length - num_comments[1])), 'constant', constant_values=0)
    
    tendency_z_2 = numpyro.sample("tendency_z_2", dist.Normal(0,1).expand([4, num_comments[2]]))    
    tendency_z_2 = jnp.pad(tendency_z_2, ((0,1), (0, max_length - num_comments[2])), 'constant', constant_values=0)

    tendency_z_3 = numpyro.sample("tendency_z_3", dist.Normal(0,1).expand([4, num_comments[3]]))    
    tendency_z_3 = jnp.pad(tendency_z_3, ((0,1), (0, max_length - num_comments[3])), 'constant', constant_values=0)

    tendency_z_4 = numpyro.sample("tendency_z_4", dist.Normal(0,1).expand([4, num_comments[4]]))    
    tendency_z_4 = jnp.pad(tendency_z_4, ((0,1), (0, max_length - num_comments[4])), 'constant', constant_values=0)

    tendency_z_5 = numpyro.sample("tendency_z_5", dist.Normal(0,1).expand([4, num_comments[5]]))    
    tendency_z_5 = jnp.pad(tendency_z_5, ((0,1), (0, max_length - num_comments[5])), 'constant', constant_values=0)
    
    tendency_z = jnp.stack([tendency_z_0, tendency_z_1, tendency_z_2, tendency_z_3, tendency_z_4, tendency_z_5], axis=0)
    comment_slopes = matmul_per_rule(chol_cov, tendency_z)
    
    # get values of predictor variables
    user_rules = scraped_dataset.rule.values.astype(int)
    user_comment_indices = scraped_dataset.comment_index.values.astype(int)
    user_indices = scraped_dataset.user_index.values.astype(int)
    user_com_gen = scraped_dataset.com_gen.values.astype(float)
    user_com_sub = scraped_dataset.com_sub.values.astype(float)
    user_rem_gen = scraped_dataset.rem_gen.values.astype(float)
    user_rem_sub = scraped_dataset.rem_sub.values.astype(float)
    user_age = scraped_dataset.age.values.astype(float)
    user_is_mod = scraped_dataset.is_mod.values.astype(int)
    mod_irl = dataset[(dataset.user != True) & (dataset.survey  != True)]
    mod_irl_rules = mod_irl.rule.values.astype(int)
    mod_irl_comment_indices = mod_irl.comment_index.values.astype(int)
    mod_survey = dataset[(dataset.user != True) & (dataset.survey ==True)]
    mod_survey_comment_indices = mod_survey.comment_index.values.astype(int)
    mod_survey_mod_indices = mod_survey.mod_index.values.astype(int)

    #set tendencies based on mu's, varying user andcomment effects, and user characteristics
    user_perc_tendency = (mu_r[user_rules, 0] + comment_slopes[user_rules, 0, user_comment_indices] + \
                          com_gen_slope[user_rules, 0] * user_com_gen + com_sub_slope[user_rules, 0] * user_com_sub + \
                          rem_sub_slope[user_rules, 0] * user_rem_sub + rem_gen_slope[user_rules, 0] * user_rem_gen + \
                          age_slope[user_rules, 0] * user_age + is_mod_slope[user_rules, 0, user_is_mod] + user_slopes[0, user_indices])
    user_op_tendency = (mu_r[user_rules, 1] + comment_slopes[user_rules, 1, user_comment_indices] + \
                        com_gen_slope[user_rules, 1] * user_com_gen + com_sub_slope[user_rules, 1] * user_com_sub + \
                        rem_sub_slope[user_rules, 1] * user_rem_sub + rem_gen_slope[user_rules, 1] * user_rem_gen + \
                        age_slope[user_rules, 1] * user_age + is_mod_slope[user_rules, 1, user_is_mod] + user_slopes[1, user_indices])
    user_app_tendency = (mu_r[user_rules, 2] + comment_slopes[user_rules, 2, user_comment_indices] + \
                         com_gen_slope[user_rules, 2] * user_com_gen + com_sub_slope[user_rules,2] * user_com_sub + \
                         rem_sub_slope[user_rules,2] * user_rem_sub + rem_gen_slope[user_rules,2] * user_rem_gen + \
                         age_slope[user_rules,2] * user_age + is_mod_slope[user_rules, 2, user_is_mod] + user_slopes[2, user_indices])    
    
    mod_irl_tendency =  mu_r[mod_irl_rules, 3] +  (comment_slopes[mod_irl_rules, 3, mod_irl_comment_indices])

    mod_survey_tendency = (mu_r_0_survey + comment_slopes[0, 4, mod_survey_comment_indices] + \
                            mod_survey_slopes[mod_survey_mod_indices])

    #outcomes for ratings supplied by users who gave us their username in the survey (exact values for adjustment variables)
    u_perc_responses = numpyro.sample("u_perc_responses", dist.Binomial(total_count = 1, logits= user_perc_tendency),
                                      obs=scraped_dataset[(scraped_dataset.user == True)].user_perception.values.astype(int) // 2)
    u_op_responses = numpyro.sample("u_op_responses", dist.Binomial(total_count = 1, logits = user_op_tendency),
                                    obs=scraped_dataset[(scraped_dataset.user == True)].user_opinion.values.astype(int) // 2)
    u_app_responses = numpyro.sample("u_app_responses", dist.Binomial(total_count=1, logits = user_app_tendency),
                                    obs=scraped_dataset[(scraped_dataset.user == True)].user_application.values.astype(int) // 2)

    #mod generated outcomes
    m_irl_responses = numpyro.sample("m_irl_responses", dist.Binomial(total_count=1, logits = mod_irl_tendency),
                                    obs=dataset[(dataset.user != True) & (dataset.survey !=True)].mod_decision.values.astype(int))
    m_survey_responses =  numpyro.sample("m_survey_responses", dist.Binomial(total_count=1, logits=mod_survey_tendency),
                                        obs=dataset[(dataset.user != True) & (dataset.survey ==True)].mod_decision.values.astype(int))

    #handling for ratings supplied by users who self-reported adjustment variable values (binned)

    if self_report_dataset is not None:
        self_report_rule_indices = self_report_dataset.rule.values.astype(int)
        #generate latent value arrays
        predictor_sums = []
        #iterate over predictors and 
        for predictor in predictors:
            counts = predictor_bin_info[predictor]["counts"] #identify how many users fall into each bin (so that we know how many latent values to sample)
            type2endpoints = predictor_bin_info[predictor]["type2endpoints"] #identify the endpoints of each bin
            num_bins = max(type2endpoints.keys()) + 1
            max_length = max(counts.values())

            arrays = []
            for bin in range(num_bins):
                endpoints = type2endpoints[bin]
                cur_count = counts[bin]
                #cases where there are no endpoints and we have a single fixed value (e.g. 0 comment removals on r/ChangeMyView)
                if type(endpoints) is not tuple:
                    arrays.append(jnp.concatenate([jnp.full((1, cur_count), endpoints), jnp.zeros((1, max_length-cur_count))], axis=1))
                #if the bin has one endpoint, sample a value drawn from Exp(1) and add it to the top end of this endpoint
                elif len(endpoints) == 1:
                    latent_vals = numpyro.sample("{}_{}".format(predictor, endpoints), dist.Exponential(1).expand([1, cur_count]))
                    latent_vals = latent_vals + endpoints[0]
                    arrays.append(jnp.concatenate([latent_vals, jnp.zeros((1, max_length-cur_count))], axis=1))
                #if the bin has two endpoints, sample uniformly at random between these two endpoints
                else:
                    latent_vals = numpyro.sample("{}_{}".format(predictor, endpoints), dist.Uniform(endpoints[0], endpoints[1]).expand([1, cur_count]))
                    arrays.append(jnp.concatenate([latent_vals, jnp.zeros((1, max_length-cur_count))], axis=1))                    

            #pre-compute the contribution of this predictor variable to each latent continuous tendency 
            arrays = jnp.stack(arrays)
            slopes = slope_map[predictor][self_report_dataset.rule.values.astype(int)]
            predictor_per_point = arrays[self_report_dataset["{}_type".format(predictor)].values.astype(int),
                                                          self_report_dataset["{}_offset".format(predictor)].values.astype(int)]
            predictor_per_point = jnp.expand_dims(self_report_dataset["{}_offset".format(predictor)].values.astype(int), axis=1)
            predictor_sum = slopes *  predictor_per_point
            predictor_sums.append(predictor_sum)
        #add the contributions of all predictor variables together -- should have a single value for each rating supplied by a user who
        #self reported adjustment data
        predictor_sums = sum(predictor_sums)
        self_report_user_ids = self_report_dataset.self_report_id.values.astype(int)
 
        # set predictor variables
        user_ratings_dataset = self_report_dataset[(self_report_dataset.user == True)]
        user_rules = user_ratings_dataset.rule.values.astype(int)
        user_is_mod = user_ratings_dataset.is_mod.values.astype(int)
        user_comment_indices = user_ratings_dataset.comment_index.values.astype(int)
        user_indices = user_ratings_dataset.user_index.values.astype(int)
        #set tendencies
        user_perc_tendency = (mu_r[user_rules, 0] + comment_slopes[user_rules, 0, user_comment_indices] + 
                              predictor_sums[:,0] + is_mod_slope[user_rules, 0, user_is_mod] + user_slopes[0, user_indices])

        user_op_tendency = (mu_r[user_rules, 1] + comment_slopes[user_rules, 1, user_comment_indices] + \
                            predictor_sums[:, 1] + is_mod_slope[user_rules, 1, user_is_mod] + user_slopes[1, user_indices])


        user_app_tendency = (mu_r[user_rules, 2] + comment_slopes[user_rules, 2, user_comment_indices] + \
                            predictor_sums[:, 2] + is_mod_slope[user_rules, 2, user_is_mod] + user_slopes[2, user_indices])    

        #outcomes
        u_perc_responses = numpyro.sample("u_perc_responses_self_report", dist.Binomial(total_count=1, logits=user_perc_tendency),
                                          obs=self_report_dataset[(self_report_dataset.user == True)].user_perception.values.astype(int) //2)
        u_op_responses = numpyro.sample("u_op_responses_self_report", dist.Binomial(total_count=1, logits= user_op_tendency),
                                        obs=self_report_dataset[(self_report_dataset.user == True)].user_opinion.values.astype(int) // 2)
        u_app_responses = numpyro.sample("u_app_responses_self_report", dist.Binomial(total_count=1, logits = user_app_tendency),
                                        obs=self_report_dataset[(self_report_dataset.user == True)].user_application.values.astype(int) // 2)
            
def posterior_analysis_adjusted(cor_model, dataset, user_map, mod_survey_map, pop_df, pb=True):
    #sample values for model parameters from the posterior
    posterior = cor_model.get_samples()
    chol_covs_0 = posterior['cov_0']
    chol_covs = posterior['cov']

    mu_rs = posterior["mu_r"]
    mu_r_0_surveys = posterior["mu_r_0_survey"]
    age_slopes = posterior["age_slope"]
    is_mod_slopes = posterior["mod_slope"]
    com_sub_slopes = posterior["com_sub_slope"]
    com_gen_slopes = posterior["com_gen_slope"]
    rem_sub_slopes = posterior["rem_sub_slope"]
    rem_gen_slopes = posterior["rem_gen_slope"]
   
    #get values of adjustment variables -- note that we exlude binned responses here
    #underlying assumption is that self-reported and scraped data do not differ significantly
    all_ages = pop_df.age.values.astype(float)
    all_rem_gens = pop_df.rem_gen.values.astype(float)
    all_com_subs = pop_df.com_sub.values.astype(float)
    all_com_gens = pop_df.com_gen.values.astype(float)
    all_rem_subs = pop_df.rem_sub.values.astype(float)
    all_is_mods = pop_df.is_mod.values.astype(int)
    pop_size = len(all_ages)

    corrs = {"op_MIRL": [], "perc_MIRL": [], "msurvey_MIRL": [], "app_MIRL": []}
    means = {"IRL": [], "perc": [], "op": [], "m_survey": [], "app": []}
    
    #simulate conducting survey w/ participants who allowed us to scraped data 1000 times for each draw from the posterior 
    for i in range(0, 3000):
        if i % 500 == 0:
            print(i)
        mu_r, mu_r_0, chol_cov_0, chol_cov = mu_rs[i], mu_r_0_surveys[i], chol_covs_0[i], chol_covs[i]
        age_slope, is_mod_slope, com_sub_slope, com_gen_slope, rem_sub_slope, rem_gen_slope = age_slopes[i], is_mod_slopes[i], com_sub_slopes[i], com_gen_slopes[i], rem_sub_slopes[i], rem_gen_slopes[i]
        
        #generate part of latent tendencies that come from adjustment varaibles
        tendency_deviations_perc = np.outer(age_slope[:,0], all_ages) + np.outer(com_gen_slope[:,0], all_com_gens) + \
                                   np.outer(com_sub_slope[:,0], all_com_subs) + np.outer(rem_gen_slope[:,0], all_rem_gens) + \
                                   np.outer(rem_sub_slope[:,0], all_rem_subs) + is_mod_slope[:,0, all_is_mods]
        tendency_deviations_perc = np.transpose(np.stack([tendency_deviations_perc for i in range(1000)]), (1,0,2))       


        tendency_deviations_op = np.outer(age_slope[:,1], all_ages) + np.outer(com_gen_slope[:,1], all_com_gens) + \
                                   np.outer(com_sub_slope[:,1], all_com_subs) + np.outer(rem_gen_slope[:,1], all_rem_gens) + \
                                   np.outer(rem_sub_slope[:,1], all_rem_subs) + is_mod_slope[:, 1, all_is_mods]
        tendency_deviations_op= np.transpose(np.stack([tendency_deviations_op for i in range(1000)]), (1,0,2))


        tendency_deviations_app = np.outer(age_slope[:,2], all_ages) + np.outer(com_gen_slope[:,2], all_com_gens) + \
                                   np.outer(com_sub_slope[:,2], all_com_subs) + np.outer(rem_gen_slope[:,2], all_rem_gens) + \
                                   np.outer(rem_sub_slope[:,2], all_rem_subs) + is_mod_slope[:,2, all_is_mods]
        tendency_deviations_app= np.transpose(np.stack([tendency_deviations_app for i in range(1000)]), (1,0,2))       
        


        key = random.PRNGKey(0)
        cov_0 = jnp.einsum("jk,lk->jl", chol_cov_0, chol_cov_0)
        cov = jnp.einsum("ijk,ilk->ijl", chol_cov, chol_cov)

        #generate part of latent tendicies that are rule-specific, and combine with part that comes from adjustment variables
        tendency_0 = random.multivariate_normal(key, jnp.append(mu_r[0, :], mu_r_0), cov_0, shape=(1000,)).transpose()
        tendency = random.multivariate_normal(key, mu_r[1:, :], cov, shape=(1000,5)).transpose()
      
        user_perc_tendencies_0 = tendency_0[0,:]
        user_perc_tendencies_0 = np.expand_dims(user_perc_tendencies_0, axis=-1)
        user_perc_tendencies_0 = user_perc_tendencies_0 + tendency_deviations_perc[0,:]

        user_op_tendencies_0 = tendency_0[1,:]
        user_op_tendencies_0 = np.expand_dims(user_op_tendencies_0, axis=-1)
        user_op_tendencies_0 = user_op_tendencies_0 + tendency_deviations_op[0,:]

        user_app_tendencies_0 = tendency_0[2,:]
        user_app_tendencies_0 = np.expand_dims(user_app_tendencies_0, axis=-1)
        user_app_tendencies_0 = user_app_tendencies_0 + tendency_deviations_app[0,:]

        mod_irl_tendencies_0 = tendency_0[3,:]
        mod_survey_tendencies_0 = tendency_0[4,:]

        user_perc_tendencies = tendency[0,:]
        user_perc_tendencies = np.expand_dims(user_perc_tendencies, axis=-1)
        user_perc_tendencies = user_perc_tendencies + tendency_deviations_perc[1:,:]
        
        user_op_tendencies = tendency[1,:]
        user_op_tendencies = np.expand_dims(user_op_tendencies, axis=-1)
        user_op_tendencies = user_op_tendencies + tendency_deviations_op[1:,:]

        user_app_tendencies = tendency[2,:]
        user_app_tendencies = np.expand_dims(user_app_tendencies, axis=-1)
        user_app_tendencies = user_app_tendencies + tendency_deviations_app[1:,:]

        mod_irl_tendencies = tendency[3,:]

        #find avg probability of positive label per user across 1000 simulations of survey
        removal_proportion_perc =  np.mean(expit(user_perc_tendencies), axis=-1)
        removal_proportion_perc_0 =  np.mean(expit(user_perc_tendencies_0), axis=-1)

        #find avg probability of positive label across all 1000 simulations and all users
        avg_perc_probs_0 = np.mean(removal_proportion_perc_0, axis=-1)
        avg_perc_probs = np.mean(removal_proportion_perc, axis=1)
        avg_perc_probs = np.concatenate([np.expand_dims(avg_perc_probs_0, axis=0), avg_perc_probs])
        means["perc"].append(avg_perc_probs)

        #do the same for application and opinion data
        removal_proportion_app =  np.mean(expit(user_app_tendencies), axis=-1)
        removal_proportion_app_0 =  np.mean(expit(user_app_tendencies_0), axis=-1)

        avg_app_probs_0 = np.mean(removal_proportion_app_0, axis=-1)
        avg_app_probs = np.mean(removal_proportion_app, axis=1)
        avg_app_probs = np.concatenate([np.expand_dims(avg_app_probs_0, axis=0), avg_app_probs])
        means["app"].append(avg_app_probs)

        removal_proportion_op =  np.mean(expit(user_op_tendencies), axis=-1)
        removal_proportion_op_0 =  np.mean(expit(user_op_tendencies_0), axis=-1)
                
        avg_op_probs_0 = np.mean(removal_proportion_op_0, axis=-1)
        avg_op_probs = np.mean(removal_proportion_op, axis=1)
        avg_op_probs = np.concatenate([np.expand_dims(avg_op_probs_0, axis=0), avg_op_probs])
        means["op"].append(avg_op_probs)

        #Simulate IRL mod decisions
        mod_irl_probs_0 = expit(mod_irl_tendencies_0)
        mod_irl_probs = expit(mod_irl_tendencies)
        mod_irl_decisions_0 = np.random.binomial(n=1, p=mod_irl_probs_0)
        mod_irl_decisions = np.random.binomial(n=1, p=mod_irl_probs)

        avg_irl_probs_0 = np.mean(mod_irl_probs_0)
        avg_irl_probs = np.mean(mod_irl_probs, axis=1)
        avg_irl_probs = np.concatenate([jnp.expand_dims(avg_irl_probs_0, 0), avg_irl_probs], axis=0)
        means["IRL"].append(avg_irl_probs)
        mod_survey_probs_0 = expit(mod_survey_tendencies_0)
        avg_survey_probs = np.mean(mod_survey_probs_0)
        mod_survey_decisions_0 = np.random.binomial(n=1, p=mod_survey_probs_0)
        means["m_survey"].append(avg_survey_probs)

        #compute correlations and organize into dictionary
        mod_survey_irl = pb_corr(jnp.expand_dims(mod_survey_probs_0, axis=-1), jnp.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)

        perc_corr_irl = pb_corr(removal_proportion_perc, mod_irl_decisions)
        op_corr_irl = pb_corr(removal_proportion_op, mod_irl_decisions)
        app_corr_irl = pb_corr(removal_proportion_app, mod_irl_decisions)
        perc_corr_irl_0 = pb_corr(np.expand_dims(removal_proportion_op_0, axis=-1), np.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)
        op_corr_irl_0 = pb_corr(np.expand_dims(removal_proportion_perc_0, axis=-1), np.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)
        app_corr_irl_0 = pb_corr(np.expand_dims(removal_proportion_app_0, axis=-1), np.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)


        perc_corr_irl = np.concatenate([perc_corr_irl_0, perc_corr_irl], axis=0)
        op_corr_irl = np.concatenate([op_corr_irl_0, op_corr_irl], axis=0)
        app_corr_irl = np.concatenate([app_corr_irl_0, app_corr_irl], axis=0)

      
        corrs["op_MIRL"].append(op_corr_irl)
        corrs["perc_MIRL"].append(perc_corr_irl)
        corrs["app_MIRL"].append(app_corr_irl)
        corrs["msurvey_MIRL"].append(mod_survey_irl)

    return corrs, means

#sets up predictor bins and run model
def run_model_MU_adjusted(dataset, user_map, mod_survey_map, adjustment = False):
   
    #split dataset into scraped and self-reported components
    new_user_dataset = dataset.copy()
    user_scraped = new_user_dataset[(new_user_dataset["self_report"] == False) & (new_user_dataset["user"] == True)]
    user_self_report = new_user_dataset[(new_user_dataset["self_report"] == True) & (new_user_dataset["user"] == True)]
    means = {}
    stds = {}

    #compute means/stds of adjustment vars for standardization
    means["com_gen"], stds["com_gen"] = np.mean(user_scraped.com_gen.values), np.std(user_scraped.com_gen.values)
    means["rem_gen"], stds["rem_gen"]  = np.mean(user_scraped.rem_gen.values), np.std(user_scraped.rem_gen.values)
    means["com_sub"], stds["com_sub"]  = np.mean(user_scraped.com_sub.values), np.std(user_scraped.com_sub.values)
    means["rem_sub"], stds["rem_sub"]  = np.mean(user_scraped.rem_sub.values), np.std(user_scraped.rem_sub.values)
    means["age"], stds["age"]  = np.mean(user_scraped.age.values), np.std(user_scraped.age.values)
    predictors = ["com_gen", "rem_gen", "com_sub", "rem_sub", "age"]
    predictor_bin_info = {} 
    
    #organize predictors into bin_info format described above
    for predictor in predictors:
        #standardize non-binned data
        user_scraped[predictor] = (user_scraped[predictor] - means[predictor] ) / stds[predictor]
        endpoint2type = {}
        bin_counts = {}
        type2endpoints = {}
        bin_type = 0
        bin_offsets = []
        bin_types = []
        bin2offsets = {}
        for index, row in user_self_report.iterrows():
            #standardize endpoint values
            user = row["user_index"]
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
                bin2offsets[bin_type] = {}
                bin_type += 1
            #map current endpoint to bin type and 
            cur_bin_type = endpoint2type[cur_endpoint]

            if user not in bin2offsets[cur_bin_type]:
                bin2offsets[cur_bin_type][user] = bin_counts[cur_bin_type]
                bin_counts[cur_bin_type] += 1            
            bin_types.append(cur_bin_type)
            bin_offsets.append(bin2offsets[cur_bin_type][user])

        user_self_report[predictor+"_type"] = bin_types
        user_self_report[predictor+"_offset"] = bin_offsets
        predictor_bin_info[predictor] = {"counts": bin_counts, "type2endpoints": type2endpoints}


    #run_model
    cor_model = MCMC(NUTS(correlation_model_MU_adjusted, target_accept_prob = .9), num_warmup=1000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=True)
    dat_list = dict(
        dataset=dataset,
        scraped_dataset = user_scraped,
        self_report_dataset = user_self_report,
        user_map=user_map,
        mod_survey_map = mod_survey_map,
        predictors=predictors,
        predictor_bin_info = predictor_bin_info
    )
    cor_model.run(random.PRNGKey(1), **dat_list)
    return cor_model, means, stds




