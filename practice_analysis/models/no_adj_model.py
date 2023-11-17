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

def matmul_per_rule(A, B):
    return jnp.einsum("ijk,ikl->ijl", A, B)
def outer_per_rule(A, B):
    return jnp.einsum("ij,ik->ijk", A, B)

#computes the point biserial correlation (https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient)
#between a binary variable and a continuous variable. This should be exactly the same as taking the
#vanilla correlation between the two vectors so there probably isn't much of a point ot this function
def pb_corr(y_pred, y, population=False, batch_first = False):
    batch_ind = -1
    if batch_first:
      batch_ind = 0
    n_1 = jnp.sum(y, axis=batch_ind)
    n = y.shape[batch_ind]
    n_0 = n - n_1
    M_0 = jnp.sum((y_pred*((y == 0).astype(int))), axis=batch_ind) / n_0
    M_1 = jnp.sum((y_pred*((y == 1).astype(int))), axis=batch_ind) / n_1
    y_pred_bar = jnp.sum(y_pred, axis=batch_ind) / n
    if population:
        s_n = jnp.sqrt( jnp.sum(jnp.square(y_pred - jnp.expand_dims(y_pred_bar, batch_ind)), axis=batch_ind)/ (n) )
        return ((M_1 - M_0) / s_n)*jnp.sqrt((n_0*n_1) / (n**2))
    else:
        s_n = jnp.sqrt( jnp.sum(jnp.square(y_pred - jnp.expand_dims(y_pred_bar, batch_ind)), axis=batch_ind)/ (n-1) )
        return ((M_1 - M_0) / s_n)*jnp.sqrt((n_0*n_1) / (n*(n-1)))

#applies corrcoef across rules 
def mapped_corrcoef(x1, x2):
   corr = lambda x, y: jnp.corrcoef(x,y)[0,1]
   return vmap(corr, in_axes=(0,0))(x1, x2)

#Model described in equation 1 of section 4.2 of the paper
def correlation_model_MU(dataset, user_outcomes, user_map, mod_survey_map):

    num_users = len(user_map)
    num_mods_survey = len(mod_survey_map)
    num_comments = [max(dataset[dataset["rule"] == i].comment_index.values)+1 for i in range(0,6)]
    max_length = max(num_comments)

    #random effects for each moderator in survey (deviation of each moderator from global mean) -- Partial pooled
    mod_survey_sigma = numpyro.sample("mod_survey_sigma", dist.HalfNormal(1))
    mod_survey_z = numpyro.sample("mod_survey_z", dist.Normal(0,1).expand([num_mods_survey]))
    mod_survey_slopes = numpyro.deterministic("mod_survey_slopes", mod_survey_z*mod_survey_sigma)

    #random effects for each user (deviation of each user from global mean) -- Partial pooled
    user_sigma = numpyro.sample("user_sigma", dist.HalfNormal(1).expand([3,1]))
    user_z = numpyro.sample("user_z", dist.Normal(0,1).expand([3, num_users]))
    user_slopes = numpyro.deterministic("user_slopes", user_z*user_sigma)

    #rule specific mean for decisions -- Partially pooled across rules within each context
    mu_r_bar = numpyro.sample("mu_r_bar", dist.Normal(0,1).expand([1,5]))
    mu_r_sigma = numpyro.sample("mu_r_sigma", dist.HalfNormal(1).expand([1,5]))
    
    mu_r_zs = numpyro.sample("mu_r_zs", dist.Normal(0,1).expand([6, 4]))
    mu_r_z_survey = numpyro.sample("mu_r_z_survey", dist.Normal(0,1))
    mu_r = numpyro.deterministic("mu_r", mu_r_bar[0, :4] + mu_r_sigma[0, :4] * mu_r_zs)
    
    #comments included in the user survey have an extra context and therefore require additional handling
    mu_r_0_survey= numpyro.deterministic("mu_r_0_survey", mu_r_bar[0,4] + mu_r_sigma[0,4] * mu_r_z_survey)

    #std for user and mod decisisions -- Partially pooled across rules/contexts
    sigma_user = numpyro.sample("sigma_user", dist.HalfNormal(1))
    sigma_users = numpyro.sample("sigma_users", dist.HalfNormal(sigma_user).expand([18]))

    sigma_mod = numpyro.sample("sigma_mod", dist.HalfNormal(1))
    sigma_mods = numpyro.sample("sigma_mods", dist.HalfNormal(sigma_mod).expand([7]))
    sigma = numpyro.deterministic("sigma", jnp.concatenate([jnp.reshape(sigma_users[:15], (5,3)), jnp.reshape(sigma_mods[:5], (5,1))], axis=1))
    sigma_0 = numpyro.deterministic("sigma_0", jnp.concatenate([jnp.reshape(sigma_users[15:], (1,3)), jnp.reshape(sigma_mods[5:], (1,2))], axis=1)[0])

    #again, additionally handling for comments included in the user survey
    chol_0 = numpyro.sample("chol_0", dist.LKJCholesky(5, 1))
    chol_cov_0 = numpyro.deterministic("cov_0", jnp.dot(jnp.diag(sigma_0), chol_0))
    chol_cov_0 = jnp.expand_dims(chol_cov_0, axis=0)
    
    #covariances across contexts for each rule
    chol = numpyro.sample("chol", dist.LKJCholesky(4, jnp.ones(5)))

    chol_cov = numpyro.deterministic("cov", matmul_per_rule(vmap(jnp.diag, in_axes=0)(sigma), chol))
    chol_cov = jnp.pad(chol_cov, ((0,0), (0, 1), (0, 1)), 'constant', constant_values=0)
    chol_cov = jnp.concatenate([chol_cov_0, chol_cov], axis=0)
    
    # zero pad the z scores for each of the comment slopes so that we can do a single matrix multiplication to compute the actual slope values
    tendencies = []
    for i in range(0, 6):
        #mod survey comments have 5 outcome values rather than 4, leading to different padding when i=0
        #Using non-centered parameterization of MVNormal described here: 
        #https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html#51_Multivariate_Centered_and_Non-Centered_Parameterizations
        tendency_z = numpyro.sample("tendency_z_{}".format(i), dist.Normal(0,1).expand([4 + int(i == 0), num_comments[i]])) 
        tendency_z = jnp.pad(tendency_z, ((0,int(i != 0)), (0, max_length - num_comments[i])), 'constant', constant_values=0)
        tendencies.append(tendency_z)
        
    tendency_z = jnp.stack(tendencies, axis=0)
    comment_slopes = matmul_per_rule(chol_cov, tendency_z) 

    user_resps = dataset[(dataset.user == True)]
    mod_irl_resps = dataset[(dataset.user != True) & (dataset.survey != True)]
    mod_survey_resps = dataset[(dataset.user != True) & (dataset.survey != False)]

    #add in comment- and user- specific effects

    user_tendencies = (mu_r[user_resps.rule.values.astype(int), 0:3] +\
                         comment_slopes[user_resps.rule.values.astype(int), 0:3,
                          user_resps.comment_index.values.astype(int)] +\
                          user_slopes[0:3, user_resps.user_index.values.astype(int)].T)
    #add in comment- and mod- specific effects
    mod_irl_tendency =  mu_r[mod_irl_resps.rule.values.astype(int), 3] + \
                        (comment_slopes[mod_irl_resps.rule.values.astype(int), 3,
                                        mod_irl_resps.comment_index.values.astype(int)])
    mod_survey_tendency = (mu_r_0_survey + comment_slopes[0, 4, mod_survey_resps.comment_index.values.astype(int)] + \
                            mod_survey_slopes[mod_survey_resps.mod_index.values.astype(int)])


    #outcomes
    u_responses = numpyro.sample("u_responses", dist.Binomial(total_count = 1, logits= user_tendencies),
                                      obs=user_outcomes.T)

    m_irl_responses = numpyro.sample("m_irl_responses", dist.Binomial(total_count=1, logits = mod_irl_tendency),
                                    obs=mod_irl_resps.mod_decision.values.astype(int))
    m_survey_responses =  numpyro.sample("m_survey_responses", dist.Binomial(total_count=1, logits=mod_survey_tendency),
                                        obs=mod_survey_resps.mod_decision.values.astype(int))

#Runs the NUTS sampler for the unadjusted alignment model
def run_model_MU(dataset, user_map, mod_survey_map, adjustment = False):
    cor_model = MCMC(NUTS(correlation_model_MU, target_accept_prob = .9), num_warmup=1000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=True)
    #, max_tree_depth=13
    user_resps = dataset[(dataset.user == True)]
    dat_list = dict(
        dataset=dataset,
        user_outcomes = np.stack([user_resps.user_perception.values.astype(int)//2,
                                  user_resps.user_opinion.values.astype(int)//2, 
                                  user_resps.user_application.values.astype(int)//2], axis=0),
        user_map=user_map,
        mod_survey_map = mod_survey_map
    )
    cor_model.run(random.PRNGKey(2), **dat_list)
    return cor_model
#Generates the measures described in 5.3.1 and 5.3.2
def posterior_analysis(cor_model, dataset, user_map, mod_survey_map, pb=True):

    posterior = cor_model.get_samples()
    chol_covs_0 = posterior['cov_0']
    chol_covs = posterior['cov']

    mu_rs = posterior["mu_r"]
    mu_r_0_surveys = posterior["mu_r_0_survey"]

    corrs = {"op_MIRL": [], "perc_MIRL": [], "op_oracle": [], "op_perc": [], "msurvey_MIRL": [], "app_MIRL": []}
    means = {"IRL": [], "perc": [], "op": [], "m_survey": [], "app": []}
    
    op_percs = []
    # for each draw from the posterior
    for i in range(0, 3000):
    
        mu_r, mu_r_0, chol_cov_0, chol_cov = mu_rs[i], mu_r_0_surveys[i], chol_covs_0[i], chol_covs[i]

        key = random.PRNGKey(0)
        cov_0 = jnp.einsum("jk,lk->jl", chol_cov_0, chol_cov_0)
        cov = jnp.einsum("ijk,ilk->ijl", chol_cov, chol_cov)

        
        # Generate some data according to drawn model parameters
        tendency_0 = random.multivariate_normal(key, jnp.append(mu_r[0, :], mu_r_0), cov_0, shape=(3000,)).transpose()
        tendency = random.multivariate_normal(key, mu_r[1:, :], cov, shape=(3000,5)).transpose()
      
        #map to probabilities
        proportions_0 = expit(tendency_0)
        proportions = expit(tendency)

        #take avg prbability of removal in different contexts
        avg_0 = jnp.mean(proportions_0, axis=-1)
        avg = jnp.mean(proportions, axis=-1)
                
        avg_perc_probs = jnp.concatenate([jnp.expand_dims(avg_0[0], 0), avg[0]], axis=0)
        means["perc"].append(avg_perc_probs)

        avg_app_probs = jnp.concatenate([jnp.expand_dims(avg_0[2], 0), avg[2]], axis=0)
        means["app"].append(avg_app_probs)

        avg_op_probs = jnp.concatenate([jnp.expand_dims(avg_0[1], 0), avg[1]], axis=0)
        means["op"].append(avg_op_probs)

        #simulate moderator decisiosn
        mod_irl_decisions_0 = np.random.binomial(n=1, p=proportions_0[3])
        mod_irl_decisions = np.random.binomial(n=1, p=proportions[3])

        avg_irl_probs = jnp.concatenate([jnp.expand_dims(avg_0[3], 0), avg[3]], axis=0)
        means["IRL"].append(avg_irl_probs)
        
        #get correlation between survey mods and IRL mods
        mod_survey_irl = pb_corr(jnp.expand_dims(proportions_0[4], axis=-1),
            jnp.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)
        means["m_survey"].append(avg_0[5])

        #compute correlation between latent removal probability an actual decision made
        perc_corr_irl = pb_corr(proportions[0], mod_irl_decisions)
        op_corr_irl = pb_corr(proportions[1], mod_irl_decisions)
        app_corr_irl = pb_corr(proportions[2], mod_irl_decisions)
        
        perc_corr_irl_0 = pb_corr(jnp.expand_dims(proportions_0[0], axis=-1),
        jnp.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)
        
        op_corr_irl_0 = pb_corr(jnp.expand_dims(proportions_0[1], axis=-1),
        jnp.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)
        
        app_corr_irl_0 = pb_corr(jnp.expand_dims(proportions_0[2], axis=-1),
        jnp.expand_dims(mod_irl_decisions_0, axis=-1), batch_first=True)

        perc_corr_irl = jnp.concatenate([perc_corr_irl_0, perc_corr_irl], axis=0)
        op_corr_irl = jnp.concatenate([op_corr_irl_0, op_corr_irl], axis=0)
        app_corr_irl = jnp.concatenate([app_corr_irl_0, app_corr_irl], axis=0)
      
        corrs["op_MIRL"].append(op_corr_irl)
        corrs["perc_MIRL"].append(perc_corr_irl)
        corrs["app_MIRL"].append(app_corr_irl)
        corrs["msurvey_MIRL"].append(mod_survey_irl)

        #compute correlations between users' practice-awareness and practice-support answers
        cur_op_perc = np.zeros(6)
        for j in range(6):
            if j == 0:
                tmp_corr = jnp.corrcoef(proportions_0[0], proportions_0[1]) 
            else:
                tmp_corr = jnp.corrcoef(proportions[0, :, j-1], proportions[1, :, j-1])
            cur_op_perc[j] = tmp_corr[0,1]
        corrs["op_perc"].append(cur_op_perc)
        
    return corrs, means

