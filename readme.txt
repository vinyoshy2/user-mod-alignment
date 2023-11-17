
Overview
Code and data associated with the CSCW paper "Measuring User-Moderator Alignment on r/ChangeMyView"

Organization
The code is organized into three notebooks: 

1. policy_awareness.ipynb contains the the logistic regression model used to conduct MRP
for the rule recognition task (survey data in awareness.json, data on adjustment variables
 out-of-sample in population_participation.json). It also contains visualizations of
the distributions of the adjustment variables in and out of survey sample. 

2. policy_support.ipynb contains the ordinal regression model used to conduct MRP for the
Likert-rating data on rule support (data in abstract.json). 

3. practice_analysis contains two versions of model used to generate our practice-support and
 practice-awareness measures -- one without MRP and one with MRP (data in practice.json).
 
The policy_analysis and practice_analysis folder contains most of the code behind each of these notebooks
(organized into "model", "viz", and "preprocessing" subfolders).

Feel free to direct any questions to vkoshy2@illinois.edu
 

