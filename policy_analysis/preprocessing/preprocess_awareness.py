import pandas as pd
import json
import numpy as np

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
#runs the MRP adjusted policy support model
#standardizes the predictors and does some extra handling for the self-reported values for the adjustment vairables
#(recall these values are interval censored). This handling is described in more detail in adj_model.py in the
#practice_analysis/models/
def load_awareness(loc, log_scale = ["age", "com_gen", "com_sub", "rem_gen", "rem_sub"]):
    with open(loc) as f:
        awareness_json = json.load(f)
    cleaned_response_json = {}
    cleaned_participation_json = {}

    master_list = {}

    user2id = {}
    id2user = {}

    self_report_user2id = {}
    self_report_id = 0

    rule_id = 5
    user_id = 0 
    gen_id = 0
    missing_seen = set()
    for key, value in awareness_json.items():
        clean_entry = {}
        clean_entry["answer"] = int(value["answer"])
        #skip entries where we don't have participation data
        if value["participation"] != None:

            if value["rule"] not in rule2id: #handles the decoy rules
                print(value["rule"])
                rule2id[value["rule"]] = rule_id
                id2rule[rule_id] = value["rule"]
                rule_id += 1
            clean_entry["rule"] = rule2id[value["rule"]]

            if value["user"] not in user2id: #assigns integer indices to users
                user2id[value["user"]] = user_id
                id2user[user_id] = value["user"]
                user_id += 1
            clean_entry["user"] = user2id[value["user"]]
            clean_participation = {}
            #load in interval-censored participation data if its self-reported
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
            #load in scraped participation data
            else:
                for key2 in value["participation"].keys():

                    if key2 in log_scale:
                        clean_participation[key2] = np.log(1+ value["participation"][key2])
                    else:                    
                        clean_participation[key2] = value["participation"][key2]
            cleaned_response_json[key] = clean_entry
            if user2id[value["user"]] not in cleaned_participation_json:
                cleaned_participation_json[user2id[value["user"]]] = clean_participation

    response_df = pd.DataFrame.from_dict(cleaned_response_json, orient="index")
    participation_df = pd.DataFrame.from_dict(cleaned_participation_json, orient="index")
    return response_df, participation_df, id2rule
