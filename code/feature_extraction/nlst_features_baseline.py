# +
import numpy as np
import pandas as pd
import sys
import os
import torch 
from tqdm import tqdm
import glob 

# warnings 
import warnings 
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Sybil 
# Add the path to the directory containing the sybil module
sys.path.append('/workspace/home/tengyuezhang/sybil_cect/code/Sybil/')
from sybil.serie import Serie
from sybil import Sybil, Serie
from sybil import visualize_attentions_v2
# -

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# print(os.environ["CUDA_VISIBLE_DEVICES"])
num_threads = os.cpu_count() // 3

device_index = 2
print(torch.cuda.device_count())
if torch.cuda.is_available() and device_index < torch.cuda.device_count():
    device = f'cuda:{device_index}'
else:
    device = 'cpu'
    print(f'Device: {device}')

num_years = 6

# attention maps: 
OUT_VIS_DIR_PATH = "/workspace/home/tengyuezhang/sybil_cect/visualizations/nlst_attention_maps_baseline"
# selected series: 
IN_CASES_PATH = '/workspace/home/tengyuezhang/sybil_cect/data/nlst_baseline/final_nlst_baseline_cases.csv' 
SAVE_ATTN_MAPS = True 
if SAVE_ATTN_MAPS and not os.path.exists(OUT_VIS_DIR_PATH):
    os.makedirs(OUT_VIS_DIR_PATH)
# both features and risk scores: 
OUT_FEATURE_PATH = '/workspace/home/tengyuezhang/sybil_cect/results/nlst_baseline/nlst_raw_features.csv'

# Initialize the Sybil model
model = Sybil("sybil_ensemble", device=device)
num_features = 512

# =======================================
# Load the CSV file
# all_cases = pd.read_csv(IN_CASES_PATH)
# df = all_cases

# for i in range(num_years):
#     df[f'pred_risk_year_{i}'] = np.nan
# for i in range(num_features):
#     df[f'feature_{i}'] = np.nan 
# =======================================

df = pd.read_csv(OUT_FEATURE_PATH)

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):    
    if not np.isnan(row['pred_risk_year_0']): 
        continue 
    dicom_dir = row['Directory']
    event = 0
    years_to_event = 1
    pid = row['pid']
    dicom_list = glob.glob(dicom_dir + '/*')
    serie = Serie(dicom_list, label=event, censor_time=years_to_event)
    
    # get predicted risk scores and features from the last hidden layer (returned along with the attentions)
    results = model.predict([serie], return_attentions=True, threads=num_threads)
        
    # append risk scores 
    for i in range(num_years):
        df.at[index, f'pred_risk_year_{i}'] = results.scores[0][i]
    
    # append features 
    for i in range(num_features):
        # 'feature' for before relu 
        # 'hidden' for after relu
        df.at[index, f'feature_{i}'] = results.attentions[0]['features'][0, 0, i] 
        
    # save updated df 
    df.to_csv(OUT_FEATURE_PATH, index=False)
    
    if SAVE_ATTN_MAPS: 
        attentions = results.attentions
    series_with_attention = visualize_attentions_v2(
        serie,
        attentions = attentions,
        pid = pid, 
        save_directory = os.path.join(OUT_VIS_DIR_PATH, str(pid)),
        gain = 1, 
    )

