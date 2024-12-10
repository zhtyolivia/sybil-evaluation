# code modified from: 
# https://github.com/leonardgarcia90/Sybil-Project/blob/main/sybil_training.ipynb
import pydicom 
import os 
import pandas as pd 
import glob 
import pickle
import re
from datetime import datetime
import argparse 

# ============= Helper functions ============== # 

def get_first_file_in_every_folder(root_directory):
    """
    Traverse the root directory to find the first file in each subfolder.

    Parameters:
    root_directory (str): The path to the root directory containing the subfolders.

    Returns:
    List[str]: A list of paths to the first file found in each subfolder.
    """

    first_files = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if filenames:
            filenames.sort()
            first_file = os.path.join(dirpath, filenames[0])
            first_files.append(first_file)
    return first_files

def make_dicom_metadata_df(dicom_list, directory_list):
    """
    Create a DataFrame containing selected metadata from a list of DICOM files.

    Parameters:
    dicom_list (List[pydicom.dataset.FileDataset]): A list of DICOM file objects.
    directory_list (List[str]): A list of directories corresponding to the DICOM files.

    Returns:
    pd.DataFrame: A DataFrame with columns for AccessionNumber, SliceThickness, 
                  ConvolutionKernel, ImageOrientationPatient, Patient_ID, index, 
                  and the file directory.
    """

    dicom_metadata_list = []
    index = 0

    for dicom in dicom_list:
        
        if hasattr(dicom, 'ConvolutionKernel') and dicom.ConvolutionKernel is not None:
            dicom_metadata_list.append([dicom.AccessionNumber, dicom.SliceThickness, dicom.ConvolutionKernel,
                                        dicom.ImageOrientationPatient, dicom.PatientID, index, directory_list[index]])
 
            index += 1
    
        elif hasattr(dicom, 'SliceThickness') and dicom.SliceThickness is not None:
            
            dicom_metadata_list.append([dicom.AccessionNumber, dicom.SliceThickness, "missing",
                                        dicom.ImageOrientationPatient, dicom.PatientID, index, directory_list[index]])
            index +=1
            
        elif hasattr(dicom, 'ImageOrientationPatient') and dicom.SliceThickness is not None:
            
            dicom_metadata_list.append([dicom.AccessionNumber, 'missing', "missing",
                                        dicom.ImageOrientationPatient, dicom.PatientID, index, directory_list[index]])
            index +=1
              
        else:
            dicom_metadata_list.append([dicom.AccessionNumber, 'missing', "missing",
                                        'missing', dicom.PatientID, index, directory_list[index]])
            print(index)
            index +=1

        
    diacom_metadata_df = pd.DataFrame(dicom_metadata_list, columns=['AccessionNumber', 'SliceThickness', 'ConvolutionKernel'
                                                               ,'ImageOrientationPatient', 'Patient_ID',"index", "Directory"])
    return diacom_metadata_df

def get_ct_orientation(df, index):
    """
    Determine the orientation of a CT scan based on the 'ImageOrientationPatient' vector.

    Parameters:
    df (pd.DataFrame): The DataFrame containing DICOM metadata, including 'ImageOrientationPatient'.
    index (int): The index of the row in the DataFrame to process.

    Returns:
    str: A string indicating the CT orientation ('axial', 'sagittal', 'coronal', or 'unknown').
    """
    image_orientation_patient_string = str(df['ImageOrientationPatient'].iloc[index])
    image_vectors = image_orientation_patient_string.replace('[','').replace(']','').split(',')
    image_vectors = [round(float(x)) for x in image_vectors]

    #[1,0,0,0,1,0]
    if image_vectors[0] == 1 and image_vectors[4] == 1: 
        return 'axial'
    
    #[0,1,0,0,0,-1]
    elif image_vectors[1] == 1 and image_vectors[5] == -1: 
        return 'sagittal'
    
    #[1,0,0,0,0,-1]
    elif image_vectors[0] == 1 and image_vectors[5] == -1: 
        return 'coronal'
    
    else:
        return 'unknown'
    
def add_orientation_column(dicom_df):
    """
    Add a new column 'Orientation' to the DICOM DataFrame indicating the CT scan orientation.

    Parameters:
    dicom_df (pd.DataFrame): The DataFrame containing DICOM metadata.

    Modifies:
    dicom_df: Adds a new column 'Orientation' to the input DataFrame.
    """

    orientation = []
    
    for index in range(len(dicom_df)):
        if str(dicom_df['ImageOrientationPatient'].iloc[index]) == 'missing':
            orientation.append('missing')
        else:
            orientation.append(get_ct_orientation(dicom_df, index))

    dicom_df['Orientation'] = orientation

def update_ommitted_nlst_df(metadata_df, reduced_df):
    """
    Create a DataFrame of DICOM records that were not included in the reduced DataFrame.

    This function identifies and returns all DICOM records from the original metadata DataFrame 
    that are not present in the reduced DataFrame based on the 'Patient_ID'.

    Parameters:
    metadata_df (pd.DataFrame): The original DataFrame containing all DICOM metadata.
    reduced_df (pd.DataFrame): The DataFrame after filtering, containing a subset of the original records.

    Returns:
    pd.DataFrame: A DataFrame containing DICOM metadata records not present in the reduced_df.
    """
    
    # Identify rows in metadata_df where 'Patient_ID' is not in the reduced_df
    ommitted_nlst_df = metadata_df[~metadata_df['Patient_ID'].isin(reduced_df['Patient_ID'].tolist())]
    
    return ommitted_nlst_df

def add_timepoint(df):
    """
    Add a 'timepoint' column to the DataFrame, indicating how many years after the first scan each scan was taken.

    Parameters:
    df (pd.DataFrame): The DataFrame containing DICOM metadata including a 'Directory' column with scan dates.

    Returns:
    pd.DataFrame: The input DataFrame with an additional 'timepoint' column.
    """
    
    # Extract the year from the 'Directory' column using regex to identify the year component of the path
    df['Year'] = df['Directory'].apply(lambda x: int(re.search(r'\d{2}-\d{2}-(\d{4})', x).group(1)))

    # Calculate the difference in years for each patient's scan relative to their first scan
    # This calculates the difference between each year in the 'Year' column and the earliest year for each 'Patient_ID'
    df['Year_Difference'] = df.groupby('Patient_ID')['Year'].transform(lambda x: x - x.min())

    # Rename the 'Year_Difference' column to 'timepoint' to indicate the relative timing of each scan
    df_with_t = df.rename(columns={'Year_Difference': 'timepoint'})

    return df_with_t

def get_scr_days(row): 
    """
    Get the time in days from randomization to a particular scan by selecting 
    src_days0, src_days1, or src_days2 from clinical_df. 

    Returns: 
    int: The number of days from randomization to a particular scan. 
    """
    timepoint_col = f'scr_days{row["timepoint"]}'
    return row[timepoint_col]

def get_scr_days_approx(row): 
    """
    Get the time in days from randomization to a particular scan. 
    If this information is unavailable, use timepoint*365 to approximate. 
    
    Returns: 
    int: The number of days from randomization to a particular scan. 
    """
    timepoint_col = f'scr_days{row["timepoint"]}'
    if pd.notna(row[timepoint_col]):
        return row[timepoint_col]
    else:
        # Use timepoint * 365 as an approximation when the scr_days value is missing
        return row['timepoint'] * 36

# ========== End of helper functions ========== # 

def main():
    parser = argparse.ArgumentParser(description='Process NLST DICOM files and filter based on specific criteria.')
    
    parser.add_argument('--dicom-path', type=str, required=True, help='Path to the root directory containing NLST DICOM files.')
    parser.add_argument('--clinical-path', type=str, required=True, help='Path to the root directory containing NLST DICOM files.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the filtered CSV file.')
    parser.add_argument('--max-followup', type=int, default=1)
    parser.add_argument('--timepoint', 
                    type=str, 
                    required=True, 
                    choices=['baseline', 'latest'], 
                    help="Which timepoint to select for multiple timepoints of a patient")

    args = parser.parse_args()
    MAX_FOLLOWUP = args.max_followup
    # Get the first .dcm file for each scan 
    root_directory_nlst = args.dicom_path
    first_files_nlst = get_first_file_in_every_folder(root_directory_nlst)
    first_files_nlst.sort()

    # Check if the LICENSE file exists and remove it from the list if present
    license_file_path = os.path.join(root_directory_nlst, 'LICENSE')
    if license_file_path in first_files_nlst:
        first_files_nlst.remove(license_file_path)
    
    # Get the first dicom dataset for each of the first .dcm files 
    first_dicom_per_nlst_file = []
    for scan in first_files_nlst:
            first_dicom_per_nlst_file.append(pydicom.dcmread(scan,force=True))

    # Convert to metadata dataframe 
    nlst_dicom_metadata_df = make_dicom_metadata_df(first_dicom_per_nlst_file, first_files_nlst)

    # Add orientation to the df 
    add_orientation_column(nlst_dicom_metadata_df)

    # Add which timepoint it is to df (0, 1, or 2)
    nlst_dicom_metadata_df = add_timepoint(nlst_dicom_metadata_df)

    # Cast 'ConvolutionKernel' to string to deal with ['Br49d', '3'] value
    nlst_dicom_metadata_df['ConvolutionKernel'] = nlst_dicom_metadata_df['ConvolutionKernel'].astype(str)

    # Filter scans with the following properties: 
    # SliceThickness = 2.5, 2.0, ConvolutionKernel = STANDARD, B30f, Orientation = axial
    reduced_nlst_df1 = nlst_dicom_metadata_df[(nlst_dicom_metadata_df['SliceThickness'] == 2.5) & (nlst_dicom_metadata_df['Orientation'] == 'axial')
                                & (nlst_dicom_metadata_df['ConvolutionKernel'] == 'STANDARD')]

    reduced_nlst_df2 = nlst_dicom_metadata_df[(nlst_dicom_metadata_df['SliceThickness'] == 2.0) & (nlst_dicom_metadata_df['Orientation'] == 'axial')
                                & (nlst_dicom_metadata_df['ConvolutionKernel'] == 'B30f')]

    reduced_nlst_df = pd.concat([reduced_nlst_df1, reduced_nlst_df2], axis=0, ignore_index=True)

    # creates dataframe of patients not included in other dataset
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    # SliceThickness = 2.0, ConvolutionKernel = FC51, Orientation = axial
    reduced_nlst_df3 = ommitted_nlst_df[(ommitted_nlst_df['SliceThickness'] == 2.0) & (ommitted_nlst_df['Orientation'] == 'axial')
                                & (ommitted_nlst_df['ConvolutionKernel'] == 'FC51')]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df3], axis=0, ignore_index=True)
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    # SliceThickness = 1.25, ConvolutionKernel = STANDARD, Orientation = axial
    reduced_nlst_df4 = ommitted_nlst_df[(ommitted_nlst_df['SliceThickness'] == 1.25) & (ommitted_nlst_df['Orientation'] == 'axial')
                                & (ommitted_nlst_df['ConvolutionKernel'] == 'STANDARD')]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df4], axis=0, ignore_index=True)
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    # SliceThickness = 3.2, ConvolutionKernel = C, Orientation = axial
    reduced_nlst_df5 = ommitted_nlst_df[(ommitted_nlst_df['SliceThickness'] == 3.2) & (ommitted_nlst_df['Orientation'] == 'axial')
                                & (ommitted_nlst_df['ConvolutionKernel'] == 'C')]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df5], axis=0, ignore_index=True)
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    # SliceThickness = 2.0, ConvolutionKernel = C, Orientation = axial
    reduced_nlst_df6 = ommitted_nlst_df[(ommitted_nlst_df['SliceThickness'] == 2.0) & (ommitted_nlst_df['Orientation'] == 'axial')
                                & (ommitted_nlst_df['ConvolutionKernel'] == 'C')]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df6], axis=0, ignore_index=True)
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    # SliceThickness = 3.2, ConvolutionKernel = D, Orientation = axial
    reduced_nlst_df7 = ommitted_nlst_df[(ommitted_nlst_df['SliceThickness'] == 3.2) & (ommitted_nlst_df['Orientation'] == 'axial')
                                & (ommitted_nlst_df['ConvolutionKernel'] == 'D')]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df7], axis=0, ignore_index=True)
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    # SliceThickness = 1.0, ConvolutionKernel = B30f, Orientation = axial
    reduced_nlst_df8 = ommitted_nlst_df[(ommitted_nlst_df['SliceThickness'] == 1.0) & (ommitted_nlst_df['Orientation'] == 'axial')
                                & (ommitted_nlst_df['ConvolutionKernel'] == 'B30f')]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df8], axis=0, ignore_index=True)
    ommitted_nlst_df = nlst_dicom_metadata_df[~nlst_dicom_metadata_df['Patient_ID'].isin(reduced_nlst_df['Patient_ID'].tolist())]

    reduced_nlst_df9 = ommitted_nlst_df.loc[[491, 17001]]
    reduced_nlst_df = pd.concat([reduced_nlst_df, reduced_nlst_df9], axis=0, ignore_index=True)

    # Choose the last timepoint 
    if args.timepoint == 'baseline': 
        final_reduced_nlst_df = reduced_nlst_df.groupby('Patient_ID').head(1)
    if args.timepoint == 'latest':
        final_reduced_nlst_df = reduced_nlst_df.groupby('Patient_ID').tail(1)

    # remove suffix from dicom directories to get parent folder
    final_reduced_nlst_df['Directory'] = final_reduced_nlst_df['Directory'].apply(
        lambda directory: directory[:directory.rindex('/')])

    # rename patient id column to match other dataframe
    final_reduced_nlst_df = final_reduced_nlst_df.rename(columns={'Patient_ID': 'pid'})

    # cast patient id column to int
    final_reduced_nlst_df['pid'] = final_reduced_nlst_df['pid'].astype(int)

    # Now, add clinical variables 
    nlst_clinical_path = args.clinical_path 
    clinical_df = pd.read_csv(nlst_clinical_path)

    merged_df_approx = pd.merge(final_reduced_nlst_df, clinical_df, on='pid', how='left')
    merged_df_approx['days_from_randomization_to_scan'] = merged_df_approx.apply(get_scr_days_approx, axis=1)

    # Calculate days_from_scan_to_diagnosis
    merged_df_approx['days_from_scan_to_diagnosis'] = merged_df_approx.apply(
        lambda row: row['candx_days'] - row['days_from_randomization_to_scan'] if pd.notna(row['candx_days']) else pd.NA,
        axis=1
    )

    # Add a column 'years_from_scan_to_diagnosis' calculated by 'days_from_scan_to_diagnosis' divided by 365
    merged_df_approx['years_from_scan_to_diagnosis'] = merged_df_approx['days_from_scan_to_diagnosis'] // 365

    # Add a column 'event' that is 
    # 1 if years_from_scan_to_diagnosis is < MAX_FOLLOWUP 
    # 0 if years_from_scan_to_diagnosis is >= MAX_FOLLOWUP or is NaN 
    merged_df_approx['event'] = merged_df_approx['years_from_scan_to_diagnosis'].apply(
        lambda x: 1 if pd.notna(x) and x < MAX_FOLLOWUP else 0
    )

    # Calculate days_from_scan_to_last_followup
    merged_df_approx['days_from_scan_to_last_followup'] = merged_df_approx.apply(
        lambda row: row['fup_days'] - row['days_from_randomization_to_scan'] if pd.notna(row['fup_days']) else pd.NA,
        axis=1
    )

    # Add a column 'years_from_scan_to_last_followup' calculated by 'days_from_scan_to_last_followup' divided by 365 
    merged_df_approx['years_from_scan_to_last_followup'] = merged_df_approx['days_from_scan_to_last_followup'] // 365

    # Add a column called 'years_to_event' that is 
    # years_from_scan_to_last_followup if event is 0 
    # years_from_scan_to_diagnosis if event is 1 
    merged_df_approx['years_to_event'] = merged_df_approx.apply(
        lambda row: row['years_from_scan_to_diagnosis'] if row['event'] == 1 
                    else min(row['years_from_scan_to_last_followup'], MAX_FOLLOWUP-1),
        axis=1
    )

    # Assertions to ensure all 'event' and 'years_to_event' values are not NaN
    assert merged_df_approx['event'].notna().all(), "There are NaN values in the 'event' column."
    assert merged_df_approx['years_to_event'].notna().all(), "There are NaN values in the 'years_to_event' column."

    # Select the relevant columns
    cleaned_df = merged_df_approx[['pid', 'event', 'years_to_event', 'years_from_scan_to_diagnosis', 'years_from_scan_to_last_followup', 'Year', 'timepoint', 'AccessionNumber', 'Directory']]

    # Save the cleaned DataFrame to CSV
    cleaned_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()
