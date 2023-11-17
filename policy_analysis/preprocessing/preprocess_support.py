import json
import pandas as pd
import numpy as np

#standardize population values based on mean/stds used to fit model
# (this comes from scraped data of survey respondents, and not self-report data)
def standardize_pop_df(pop_df, means, stds):
    pop_df_copy = pop_df.copy()

    pop_df_copy["self_report"] = False
    pop_df_copy["com_gen"] = (pop_df_copy["com_gen"] - means["com_gen"]) / stds["com_gen"]
    pop_df_copy["rem_gen"] = (pop_df_copy["rem_gen"] - means["rem_gen"]) / stds["rem_gen"]
    pop_df_copy["com_sub"] = (pop_df_copy["com_sub"] - means["com_sub"]) / stds["com_sub"]
    pop_df_copy["rem_sub"] = (pop_df_copy["rem_sub"] - means["rem_sub"]) / stds["rem_sub"]
    pop_df_copy["age"] = (pop_df_copy["age"] - means["age"]) / stds["age"]
    return pop_df_copy

#load in the values for the adjustment variables scraped from both respondents+ nonrespondents
def load_population_participation(loc, log_scale =["age", "com_sub", "com_gen", "rem_sub", "rem_gen"]):
    with open(loc) as f:
	    pop_json = json.load(f)
	    for key in pop_json.keys():
	        for key2 in pop_json[key].keys():
	            if key2 in log_scale:
	                pop_json[key][key2] = np.log(1 + pop_json[key][key2])
    pop_df = pd.DataFrame.from_dict(pop_json, orient="index")
    rule_df = pd.DataFrame({"rule": list(range(5))})
    pop_df = pop_df.join(rule_df, how="cross")
    pop_df["answer"] = None
    return pop_df


#load in the json file containing the support responses
def load_support(loc):
    with open(loc) as f:
        support_json = json.load(f)
    cleaned_response_json = {}
    cleaned_participation_json = {}
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


    user2id = {}
    id2user = {}
    user_id = 0 
    self_report_user2id = {}
    self_report_id = 0
    
    #iterate over raw data and extract relevant values
    for key, value in support_json.items():
        clean_entry = {}
        clean_entry["answer"] = int(value["answer"])  # extract raw likert rating  
        if value["participation"] != None:
            clean_entry["rule"] = value["rule"] -1 #convert rule to index
            if value["user"] not in user2id: #if encountering new user, create new user_id and add to maps
                user2id[value["user"]] = user_id
                id2user[user_id] = value["user"]
                user_id += 1
            clean_entry["user"] = user2id[value["user"]]
            clean_participation = {}
            #load in participation data and log scale where necessary
            if value["participation"]["self_report"]:
                for key2 in value["participation"].keys():
                    if key2 == "is_mod" or key2 == "self_report":
                        clean_participation[key2] = value["participation"][key2]                    
                    else:
                        clean_participation[key2] = tuple([np.log(1+ val) for val in value["participation"][key2]])
                if value["user"] not in self_report_user2id:
                    self_report_user2id[value["user"]] = self_report_id
                    self_report_id += 1
                clean_participation["self_report_id"] = self_report_user2id[value["user"]]

            else:
                for key2 in value["participation"].keys():
                    if key2 == "is_mod":                    
                        clean_participation[key2] = value["participation"][key2]
                    else:
                        clean_participation[key2] = np.log(1+ value["participation"][key2])
            #add cleaned response and participation data to dataset
            cleaned_response_json[key] = clean_entry
            if user2id[value["user"]] not in cleaned_participation_json:
                cleaned_participation_json[user2id[value["user"]]] = clean_participation
    response_df = pd.DataFrame.from_dict(cleaned_response_json, orient="index")
    participation_df = pd.DataFrame.from_dict(cleaned_participation_json, orient="index")
    return response_df, participation_df

