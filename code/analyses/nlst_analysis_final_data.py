# calculate metrics to evaluate Sybil's performance on NLST data 
# use the updated data from /data/lung/nlst/NLST_CT_raw 

# +
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils import resample

import sys
import os
import torch 
from argparse import Namespace
from tqdm import tqdm
import pickle 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# -



# Add the path to the directory containing the sybil module
sys.path.append('/workspace/home/tengyuezhang/sybil_cect/code/Sybil/')
from sybil.utils.metrics import concordance_index, get_survival_metrics
from sybil import Sybil, Serie

# =========== set the following parameters ===========
# file containing the risk scores of the cases and outcomes 
RISK_SCORE_FILE = '/workspace/home/tengyuezhang/sybil_cect/results/nlst_baseline/nlst_raw_features.csv'
# file containing bootstrap data (pickle)
BOOTSTRAP_DATA_FILE = '/workspace/home/tengyuezhang/sybil_cect/results/nlst_baseline/bootstrap_results_final_data.pkl'
# device to run the model on 
DEVICE = '0'
# number of bootstrap iterations to calculate metrics 
N_BOOTSTRAP = 1000
# maximum number of years of follow-up 
MAX_FOLLOWUP = 6
# =========== end of parameters ===========




# =========== helper functions ===========
def calculate_y_seq(event, time_to_event, max_followup=6):
    """
    Calculate the binary outcome sequence based on the event and time_to_event.
    Returns 
        y_seq: a numpy array of length max_followup, where y_seq[i] is 1 if the case was censored at year i, and 0 otherwise.
    """
    y_seq = np.zeros(max_followup)

    if event == 1:
        # Convert time_to_event to integer index (year)
        event_year = int(time_to_event)
        
        # Ensure the event year does not exceed the follow-up period
        event_year = min(event_year, max_followup)
        
        # Set all y_i from the event year onward to 1
        y_seq[event_year:] = 1
    
    return y_seq
# =========== end of helper functions =========== 


# +
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

# load the Sybil model 
model = Sybil("sybil_ensemble")

# load the risk scores  
df = pd.read_csv(RISK_SCORE_FILE)
# -

# generate per-year labels 
for i in range(MAX_FOLLOWUP): # create columns 
    df[f'y_{i}'] = np.nan 
for index, row in df.iterrows():
    event = row['event']
    time_to_event = row['years_to_event']
    y_seq = calculate_y_seq(event, time_to_event, max_followup=MAX_FOLLOWUP)
    for i in range(MAX_FOLLOWUP):
        df.at[index, f'y_{i}'] = y_seq[i]
# print statistics 
assert np.count_nonzero((df['years_to_event'] == 0) & (df['event'] == 1)) == np.count_nonzero((df['y_0']==1))
year0_diagnosis = np.count_nonzero((df['y_0']==1))
total = len(df)
year0_diagnosis_percent = year0_diagnosis/total 
print(f'Total number of patients = {total}')
print(f'Lung cancer diagnosis at baseline timepoint = {year0_diagnosis} ({100*year0_diagnosis/total:.2f}%)')

# +
# calculate metrics on the entire dataset (w/o bootstrapping)

# get scores and labels 
selected_columns = ['pred_risk_year_0', 'pred_risk_year_1', 'pred_risk_year_2', 
                    'pred_risk_year_3', 'pred_risk_year_4', 'pred_risk_year_5']
pred_risk_scores = df[selected_columns].values.tolist()

event_times = df['years_to_event'].tolist()
event_observed = df['event'].tolist()


# +
input_dict = {
    "probs": torch.tensor(pred_risk_scores), 
    "censors": torch.tensor(event_times), 
    "golds": torch.tensor(event_observed)
}

args = Namespace(
    max_followup=MAX_FOLLOWUP, censoring_distribution=model._censoring_dist
)

out = get_survival_metrics(input_dict, args)
print(out) ## NOTE: in the eval output, the index for year number starts at 1 instead of 0 

# +
# Plotting
plt.figure(figsize=(8, 6))
colors = ['#045275', '#089099', '#7CCBA2', '#FABF7B', '#E05C5C', '#AB1866']

for year in range(1, 2):
    # Extract FPR, TPR, and AUROC for the current year
    roc_curve_data = out[f'{year}_year_roc_curve']
    fpr = roc_curve_data['fpr']
    tpr = roc_curve_data['tpr']
    auroc = out[f'{year}_year_auc']

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'Year {year} ROC-AUC = {auroc:.4f}', color=colors[year-1])

# Plotting random chance
plt.plot([0, 1], [0, 1], 'k--', label='Random chance')

# Set plot labels and title
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('ROC Curve - NLST', fontsize=16)
plt.legend(loc='lower right', fontsize=12)

# Show plot
plt.show()

# +
# Plotting
plt.figure(figsize=(8, 6))

for year in range(1, 2):
    pr_curve_data = out[f'{year}_year_pr_curve']
    precision = pr_curve_data['precision'] 
    recall = pr_curve_data['recall']
    pr_auc = out[f'{year}_year_prauc']
    # plot pr curve 
    plt.plot(recall, precision, label=f'Year {year} AUPRC = {pr_auc:.4f}', color=colors[year-1])
# plot a horizontal line at the height year0_diagnosis_percent
if 'year0_diagnosis_percent' in locals() or 'year0_diagnosis_percent' in globals():
    plt.axhline(y=year0_diagnosis_percent, color='gray', linestyle='--', label=f'Year 0 Diagnosis Rate ({100*year0_diagnosis/total:.2f}%)')

# set plot labels and title
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-recall Curve - NLST', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.show()
# +
# now perform bootstrapping 

# ===== set bootstrap params =====
n_bootstraps = 5000 
random_seed = 42 
np.random.seed(random_seed)
# ==== end of bootstrap params =====
# -


if not os.path.exists(BOOTSTRAP_DATA_FILE): 
    # perform bootstrap 
    auroc_scores_all_years = {f'year_{i}': [] for i in range(6)} 
    auprc_scores_all_years = {f'year_{i}': [] for i in range(6)} 

    # define args
    args = Namespace(
        max_followup=6, censoring_distribution=model._censoring_dist
    )

    # to store bootstrapped metrics 
    bootstrapped_auroc = {f'year_{i+1}': [] for i in range(6)}
    bootstrapped_auprc = {f'year_{i+1}': [] for i in range(6)}

    for i in tqdm(range(n_bootstraps), desc="bootstrapping progress", ncols=100):
        # resample data with replacement 
        bootstrap_indices = resample(range(len(df)), replace=True, n_samples=len(df))
        df_bootstrap = df.iloc[bootstrap_indices]
        pred_risk_scores_bootstrap = df_bootstrap[selected_columns].values.tolist()
        event_times_bootstrap = df_bootstrap['years_to_event'].tolist() 
        event_observed_bootstrap = df_bootstrap['event'].tolist() 

        input_dict_bootstrap = {
            "probs": torch.tensor(pred_risk_scores_bootstrap), 
            "censors": torch.tensor(event_times_bootstrap), 
            "golds": torch.tensor(event_observed_bootstrap)
        }

        bootstrap_metrics = get_survival_metrics(input_dict_bootstrap, args)

        for year in range(1, 7): 
            bootstrapped_auroc[f'year_{year}'].append(bootstrap_metrics[f'{year}_year_auc'])
            bootstrapped_auprc[f'year_{year}'].append(bootstrap_metrics[f'{year}_year_prauc'])

    # save to BOOTSTRAP_DATA_FILE 
    bootstrap_results = {
        'bootstrapped_auroc': bootstrapped_auroc,  # AUROC
        'bootstrapped_auprc': bootstrapped_auprc, 
    }
    with open(BOOTSTRAP_DATA_FILE, 'wb') as f:
        pickle.dump(bootstrap_results, f)



if os.path.exists(BOOTSTRAP_DATA_FILE): 
    # read cached bootstrap data 
    with open(BOOTSTRAP_DATA_FILE, 'rb') as f:
        loaded_results = pickle.load(f)

    bootstrapped_auroc = loaded_results['bootstrapped_auroc']
    bootstrapped_auprc = loaded_results['bootstrapped_auprc']

# +
year=1
# 1) calculate mean AUROC and CIs for each year
auroc_results = {}
auroc_scores = bootstrapped_auroc[f'year_{year}']
mean_auroc = np.mean(auroc_scores)
lower_bound = np.percentile(auroc_scores, 2.5)
upper_bound = np.percentile(auroc_scores, 97.5)
auroc_results[f'year_{year}'] = {
    'mean_auroc': mean_auroc,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound
}

print('======== AUROC ========')
year_auroc = auroc_results[f'year_{year}']['mean_auroc']
year_lower = auroc_results[f'year_{year}']['lower_bound']
year_upper = auroc_results[f'year_{year}']['upper_bound']
print(f'Year {year} mean AUROC={year_auroc:.4f} (CI: [{year_lower:.2f}-{year_upper:.2f}])')
    
# 2) calculate mean AUPRC and CIs for each year 
auprc_results = {}
auprc_scores = bootstrapped_auprc[f'year_{year}']
mean_auprc = np.mean(auprc_scores)
lower_bound = np.percentile(auprc_scores, 2.5)
upper_bound = np.percentile(auprc_scores, 97.5)
auprc_results[f'year_{year}'] = {
    'mean_auprc': mean_auprc,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound
}

print('======== AUPRC ========')
year_auprc = auprc_results[f'year_{year}']['mean_auprc']
year_lower = auprc_results[f'year_{year}']['lower_bound']
year_upper = auprc_results[f'year_{year}']['upper_bound']
print(f'Year {year} mean AUROC={year_auprc:.4f} (CI: [{year_lower:.2f}-{year_upper:.2f}])')

# +
plt.figure(figsize=(8, 6))

year=1

# Extract FPR, TPR, and AUROC for the current year
roc_curve_data = out[f'{year}_year_roc_curve']
fpr = roc_curve_data['fpr']
tpr = roc_curve_data['tpr']
year_auroc = out[f'{year}_year_auc']
year_lower = auroc_results[f'year_{year}']['lower_bound']
year_upper = auroc_results[f'year_{year}']['upper_bound']

# Plot ROC curve
plt.plot(fpr, tpr, 
        label=f'AUROC={year_auroc:.4f},\nCI: [{year_lower:.2f}, {year_upper:.2f}]', 
         color=colors[year-1]
)

# Plotting random chance
plt.plot([0, 1], [0, 1], 'k--', label='Random chance')

# Set plot labels and title
plt.xlabel('False Positive Rate (FPR)', fontsize=20)
plt.ylabel('True Positive Rate (TPR)', fontsize=20)
plt.title(f'ROC Curve - NLST', fontsize=24)
plt.legend(loc='lower right', fontsize=20)
plt.show()

# +
plt.figure(figsize=(8, 6))

# Extract FPR, TPR, and AUROC for the current year
pr_curve_data = out[f'{year}_year_pr_curve']
recall = pr_curve_data['recall']
precision = pr_curve_data['precision']
year_auprc = out[f'{year}_year_prauc']
year_lower = auprc_results[f'year_{year}']['lower_bound']
year_upper = auprc_results[f'year_{year}']['upper_bound']

# Plot ROC curve
plt.plot(recall, precision, 
        label=f'AUPRC={year_auprc:.4f},\nCI: [{year_lower:.2f}, {year_upper:.2f}]', 
         color=colors[year-1]
)

# Plot horizontal line at 0.1515 to indicate the proportion of positive samples
plt.axhline(y=year0_diagnosis_percent, color='red', linestyle='--', 
            label=f'Pos. Rate = {year0_diagnosis_percent:.2f}')

# Set plot labels and title
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.title(f'PR Curve - NLST', fontsize=24)
plt.legend(loc='upper right', fontsize=20)
# Show plot
plt.show()
# -


