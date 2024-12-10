import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import torch 
from argparse import Namespace
from tqdm import tqdm
import pickle 
import glob 
import ast

# Add the path to the directory containing the sybil module
sys.path.append('/workspace/home/tengyuezhang/sybil_cect/code/Sybil/')
from sybil.utils.metrics import concordance_index, get_survival_metrics
from sybil import Sybil, Serie
from sybil import visualize_attentions_v2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
num_threads = os.cpu_count() // 5

IN_CASES_PATH = '/workspace/home/tengyuezhang/sybil_cect/data/ucla_cect/ucla_cect_98_baseline_timepoint.csv'
OUT_RISK_PATH = '/workspace/home/tengyuezhang/sybil_cect/results/ucla_cect/ucla_cect_98_risk_scores.csv'
# attention maps 
OUT_VIS_DIR_PATH = '/workspace/home/tengyuezhang/sybil_cect/visualizations/ucla_cect_98_attention_maps'
SAVE_ATTN_MAPS = True 
if SAVE_ATTN_MAPS and not os.path.exists(OUT_VIS_DIR_PATH):
    os.makedirs(OUT_VIS_DIR_PATH)

# Initialize the Sybil model
model = Sybil("sybil_ensemble")
num_years = 6

# Load the CSV file
all_cases = pd.read_csv(IN_CASES_PATH)
df = all_cases

for i in range(num_years):
    df[f'pred_risk_year_{i}'] = np.nan

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
   
    dicom_dir = row['Directory']
    # event = row['LungCancer']
    event = 0
    years_to_event = 1
    pid = row['pid']
    dicom_list = glob.glob(dicom_dir + '/*')
    serie = Serie(dicom_list, label=event, censor_time=years_to_event)
    
    results = model.predict([serie], return_attentions=True, threads=num_threads)
        
    # Update the risk scores columns for the current row
    for i in range(num_years):
        df.at[index, f'pred_risk_year_{i}'] = results.scores[0][i]
        
    # Save the updated DataFrame to the output CSV file at each iteration
    df.to_csv(OUT_RISK_PATH, index=False)
    
    # Save attention maps 
    if SAVE_ATTN_MAPS: 
        attentions = results.attentions

        series_with_attention = visualize_attentions_v2(
            serie,
            attentions = attentions,
            pid = pid, 
            save_directory = os.path.join(OUT_VIS_DIR_PATH, str(pid)),
            gain = 1, 
            save_pngs = True, 
            save_rep_slice = True,
        )

