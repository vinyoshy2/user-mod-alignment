import pandas as pd
import numpy as np
import json

adjustment_vars = ["age", "com_sub", "com_gen", "rem_sub", "rem_gen"]

#maps an _id to a index for ease of use in the model
def map_id(_id, map, count):
    if _id not in map:
        map[_id] = count
        count += 1
    return map[_id], count

#extracts information from a comment entry to create an entry for the IRL decision made for a comment
def gen_irl_entry(entry, key):
    return {
        "_id": key,
        "mod_decision": entry["orig_decision"],
        "comment_index": entry["comment_index"],
        "rule": entry["rule"],
        "user": False,
        "survey": False
    }
#extracts information from a comment entry to create an entry for survey decision made by a mod for a comment
def gen_mod_entry(entry, index, key):
    return {
        "_id": key,
        "mod_decision": entry["mod_decision" + str(index)],
        "comment_index": entry["comment_index"],
        "mod_index": entry["mod_index" + str(index)],
        "rule": entry["rule"],
        "user": False,
        "survey": True
    }
#extracts information from a comment entry to create an entry for survey decisions made by a user for a comment    
def gen_user_entry(entry, index, key):
    return {
        "_id": key,
        "user_opinion": entry["user_opinion" + str(index)],
        "user_perception": entry["user_perception" + str(index)],
        "user_application": entry["user_application" + str(index)],
        "comment_index": entry["comment_index"],
        "user_index": entry["user_index" + str(index)],
        "rule": entry["rule"],
        "user": True,
        "is_mod": entry["is_mod"+ str(index)],
        "self_report": entry["self_report"+ str(index)],
        "self_report_id": entry["self_report_id"+ str(index)]
    }
    
#pulls out the users associated with a particular comment and adds their participation information into the entry
def extract_user(entry, user_map, user_count, self_report_user2id, self_report_id):
    count = 1
    user_deciders = list(entry["user_decisions"].keys())
    for user in user_deciders:
        if len(entry["user_decisions"][user]) >= 3 and all([entry["user_decisions"][user][i] != None for i in range(3)]):
            entry["user_index" + str(count)], user_count = map_id(user, user_map, user_count)
            entry["user_opinion" + str(count)] = entry["user_decisions"][user][0]
            entry["user_perception" + str(count)] = entry["user_decisions"][user][1]
            entry["user_application" + str(count)] = 2*((type(entry["user_decisions"][user][2]) is list) and len(entry["user_decisions"][user][2]) >= 1)
            participation = entry["user_decisions"][user][3]
            if participation != None:
                if "com_sub" in participation and participation["com_sub"] != None:
                    for key1 in participation.keys():
                        entry[key1 + str(count)] = participation[key1]
                    if participation["self_report"]:
                        if user not in self_report_user2id:
                            self_report_user2id[user] = self_report_id
                            self_report_id += 1
                        entry["self_report_id" + str(count)] = self_report_user2id[user]
                    else: 
                        entry["self_report_id" + str(count)] = -1
                    count+= 1
    return entry, count, user_map, user_count, self_report_user2id, self_report_id


#Dataset located at FILENAME contains an entry for each studied comment
#gen_dataset loads this dataset and converts it into one containing an entry for decider,comment pair
#e.g. a comment in the original dataset will have an IRL decision, 2 survey moderator decisions, and
# a user opinion, prediction, and rule application decision for each user assigned to the comment in the user survey.
#In the resulting dataset we will have one entry for the IRL decision, one entry for each survey moderator decision,
#and one entry for each user assigned to the comment (containing the opinion, prediction, and rule application decisions of that user)
def gen_dataset(filename):

    with open(filename) as f:
        user_mod_data = json.load(f)
        
    user_map = {}
    mod_survey_map = {}
    mod_survey_count = 0
    user_count = 0
    other_action_op = 0
    other_action_perc = 0
    op_total = 0
    perc_total = 0
    comment_index = [0,0,0,0,0,0]
    self_report_user2id = {}
    self_report_id = 0

    user_mod_data_filtered = {}
    for key, entry in user_mod_data.items():
        entry["rule"] += 1
        if "orig_decision" not in entry:
            continue
        #extract mod decider indices
        if "new_decider" in  entry:
            entry["mod_index1"], mod_survey_count = map_id(entry["new_decider"][0], mod_survey_map, mod_survey_count)
            entry["mod_index2"], mod_survey_count = map_id(entry["new_decider"][1], mod_survey_map, mod_survey_count)
        
        #extract orig decider ids
        entry["orig_decision"] = int(entry["orig_decision"] == "removed")

        #extract user decisions
        count = 0
        if "user_decisions" in entry:
            entry, count, user_map, user_count, self_report_user2id, self_report_id = extract_user(entry, user_map, user_count, self_report_user2id, self_report_id)

        if count == 0: 
            continue
        #extract survey mod decisions
        if "new_decisions" in entry:
            for i in range(1,3):
                if "Unsure" in entry["new_decisions"][i-1]:
                    entry["mod_decision"+str(i)] = 0
                elif len(entry["new_decisions"][i-1]) > 0:
                    entry["mod_decision"+str(i)] = 1
                else:
                    entry["mod_decision"+str(i)] = 0        
        #remap other action responses where appropriate
        for i in range(1,min(count,3)):
            #input files should already be manually remapped so "other action" is set to 1 
            # if it indicates a preference for removal (described in Data Analysis section of paper)
            if entry["user_opinion" + str(i)] == 3:
                other_action_op += 1
                entry["user_opinion" + str(i)] = 0
            op_total +=1
            if entry["user_perception" + str(i)] == 3:
                other_action_perc += 1
                entry["user_perception" + str(i)] = 0
            perc_total += 1
        #assign rule-specific index for this comment
        entry["comment_index"] = comment_index[entry["rule"]]
        comment_index[entry["rule"]] += 1
        
        #split out entry across raters, maintain only relevant columns
        for i in range(1,count):
            #split out user-supplied survey ratings
            user_entry = gen_user_entry(entry, i, key)
            for adj in adjustment_vars:
                if user_entry["self_report"]:
                    if len(entry[adj+ str(i)]) == 2:
                        user_entry[adj] = (np.log(1 + entry[adj+ str(i)][0]), np.log(1 + entry[adj+ str(i)][1]))
                    else:
                        user_entry[adj] = (np.log(1 + entry[adj+ str(i)][0]), )
                else:
                    user_entry[adj] = np.log(1 + entry[adj+ str(i)])
            user_mod_data_filtered[key +"user"+ str(i)] = user_entry
        if "mod_decision1" in entry and "mod_decision2" in entry:
            #split out mod-supplied survey ratings
            for i in range(1,3):
                mod_entry = gen_mod_entry(entry, i, key)
                user_mod_data_filtered[key +"mod_survey"+ str(i)] = mod_entry
        if entry["orig_decision"] != None:
            #split out real life mod-supplied decisiosn
            irl_entry = gen_irl_entry(entry, key)
            user_mod_data_filtered[key +"mod_irl"] = irl_entry
    user_mod_df = pd.DataFrame.from_dict(user_mod_data_filtered, orient="index")
    return user_mod_df, user_map, mod_survey_map
    
#loads in the adjustment variable values for the sample of people who received a survey link (containing both respondents and non-respondents)
#standardizes the data based on means + stds of these variables in the population of users who volunteered their accounts for scraping (a majoritarian
# subset of survey respondents)
def load_pop_df(pop_json_loc, means, stds):
    adjustment_vars = ["age", "com_sub", "com_gen", "rem_sub", "rem_gen"]
    with open(pop_json_loc) as f:
        pop_json = json.load(f)
        for key in pop_json.keys():
            for key2 in pop_json[key].keys():
                if key2 in adjustment_vars:
                    pop_json[key][key2] = np.log(1 + pop_json[key][key2])
    pop_df = pd.DataFrame.from_dict(pop_json, orient="index")
    pop_df["com_sub"] = (pop_df["com_sub"] - means["com_sub"]) / stds["com_sub"]
    pop_df["com_gen"] = (pop_df["com_gen"] - means["com_gen"]) / stds["com_gen"]
    pop_df["rem_sub"] = (pop_df["rem_sub"] - means["rem_sub"]) / stds["rem_sub"]
    pop_df["rem_gen"] = (pop_df["rem_gen"] - means["rem_gen"]) / stds["rem_gen"]
    pop_df["age"] = (pop_df["age"] - means["age"]) / stds["age"]
    return pop_df
