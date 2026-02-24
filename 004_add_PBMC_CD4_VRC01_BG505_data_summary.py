#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:11:00 2025

@author: wuw5
"""

import re
import sys
import os
import pandas as pd
from functools import reduce
from collections import defaultdict

from datetime import datetime
project_path = "/Volumes/wuw5$/notes/20250509_VRC01Character_plots/Data_process/20250507_C101High_related_rerun_VDJSHM"
filtered_cell_File = os.path.join(project_path, '003_modified_Subject_Sorted_9281Cells.xlsx')
filtered_cell_df = pd.read_excel(filtered_cell_File)
output_path = "/Volumes/wuw5$/notes/20250602_C101_add_PBMC_number/data_with_process_CellNumber/"

filtered_cell_df.shape
#Out[132]: (9281, 151)

PBMC_df_C107 = pd.read_excel('/Volumes/wuw5$/notes/20250602_C101_add_PBMC_number/C107_SampleLevelTracking_250304_MP.xlsx')
PBMC_df_C110 = pd.read_excel('/Volumes/wuw5$/notes/20250602_C101_add_PBMC_number/C110_SampleLevelTracking_240719_MP.xlsx')
PBMC_df_C101 = pd.read_excel('/Volumes/wuw5$/notes/20250602_C101_add_PBMC_number/C101_SampleLevelTracking_HD.xlsx')

filtered_cell_df['VisitID'] = filtered_cell_df['Timepoint_Sorted']
filtered_cell_df['DonorID'] = filtered_cell_df.apply(
    lambda row: f"{row['Trial']}{row['Subject_Sorted'][-5:]}"
    if row['Trial'] in ['C107', 'C110']
    else row['Subject_Sorted'],
    axis=1
)
C107_sort = PBMC_df_C107[['DonorID','VisitID','# of Cells Processed (mil)']]
C110_sort = PBMC_df_C110[['DonorID','VisitID','# of Cells Processed (mil)']]
C101_sort = PBMC_df_C101[['DonorID','VisitID','# of Cells Processed (mil)']]

##1. Combine the C107, C110, and C101 DataFrames
#First, concatenate these three into a single DataFrame:
all_sorts = pd.concat([C107_sort, C110_sort, C101_sort], ignore_index=True)


#2. Merge with filtered_cell_df
#Now, use pd.merge to bring the # of Cells Processed (mil) info into filtered_cell_df:

merged_df = pd.merge(
    filtered_cell_df,
    all_sorts,
    on=['DonorID', 'VisitID'],
    how='left'
)


merged_df['Flow_Index_BG505'] = merged_df['Flow_Index_BG505'].fillna('missing')

CD4_df = merged_df[
    (merged_df['Flow_Index_Main'].astype(str).str.contains("CD4", na=False))]
CD4_df.shape
#Out[3]: (7453, 154)


output_path = "/Volumes/wuw5$/notes/20250602_C101_add_PBMC_number/data_with_process_CellNumber/"
CD4_df.to_excel(os.path.join(output_path,'7453_CD4_data.xlsx'), index = False)


VRC01_df = merged_df[
    (merged_df['v_call_heavy'].astype(str).str.contains("IGHV1-2", na=False)) &  # Match IGHV1-2
    (merged_df['junction_aa_light'].astype(str).str.len() == 7)  # Length of junction_aa_light is 7
]
VRC01_df.shape
#Out[129]: (1791, 154)
VRC01_df.to_excel(os.path.join(output_path,'1791Cells_VRC01_data.xlsx'), index = False)


print(
    (
        (merged_df['Trial'] == "C101_Low") & 
        (merged_df['Flow_Index_BG505'] == "BG505+")
    ).any()
)
#False

########a new count based on only sample info#######

# Step 1: Make sure your DataFrames are as follows:
all_sample_counts = (
    filtered_cell_df
    .groupby(['Subject_Sorted', 'Timepoint_Sorted'])
    .size()
    .reset_index(name='All_Cell_Number')
)

cd4_sample_counts = (
    CD4_df
    .groupby(['Subject_Sorted', 'Timepoint_Sorted'])
    .size()
    .reset_index(name='CD4_Cell_Number')
)

vrc01_sample_counts = (
    VRC01_df
    .groupby(['Subject_Sorted', 'Timepoint_Sorted'])
    .size()
    .reset_index(name='VRC01_Cell_Number')
)    
    
######BG505 statist

BG505_df = filtered_cell_df[filtered_cell_df['Flow_Index_BG505'] == "BG505+"]
BG505_df.shape
#Out[30]: (760, 127)
BG505_posi_count = (
    BG505_df
    .groupby(['Subject_Sorted', 'Timepoint_Sorted'])
    .size()
    .reset_index(name='BG505+Cell_Number')
)
BG505_negative_df = filtered_cell_df[filtered_cell_df['Flow_Index_BG505'] == "BG505-"]
BG505_negative_df.shape
#Out[31]: (2051, 127)
BG505_nega_count =(
    BG505_negative_df
    .groupby(['Subject_Sorted', 'Timepoint_Sorted'])
    .size()
    .reset_index(name='BG505-Cell_Number')
)
BG505_df.to_excel(os.path.join(output_path,"760Cells_BG505_positive.xlsx"), index=False)
BG505_negative_df.to_excel(os.path.join(output_path,"2051Cells_BG505_negative.xlsx"), index=False)
# all_sample_counts, cd4_sample_counts, vrc01_sample_counts

# Step 2: Merge (outer join) on Subject_Sorted and Timepoint_Sorted
dfs = [all_sample_counts, cd4_sample_counts, vrc01_sample_counts]
merged_sample_counts = reduce(
    lambda left, right: pd.merge(left, right, on=['Subject_Sorted', 'Timepoint_Sorted'], how='outer'),
    dfs
).fillna(0)

# Step 3: Convert count columns to float for division
for col in ['All_Cell_Number', 'CD4_Cell_Number', 'VRC01_Cell_Number']:
    merged_sample_counts[col] = merged_sample_counts[col].astype(float)

# Step 4: Compute ratios (using pd.NA to avoid division by zero)
merged_sample_counts['cd4_ratio'] = merged_sample_counts['CD4_Cell_Number'] / merged_sample_counts['All_Cell_Number'].replace(0, pd.NA)
merged_sample_counts['vrc01_ratio_all'] = merged_sample_counts['VRC01_Cell_Number'] / merged_sample_counts['All_Cell_Number'].replace(0, pd.NA)
merged_sample_counts['vrc01_ratio_cd4'] = merged_sample_counts['VRC01_Cell_Number'] / merged_sample_counts['CD4_Cell_Number'].replace(0, pd.NA)


# List BG505 related data to the merged_sample_counts
dfs = [merged_sample_counts, BG505_nega_count, BG505_posi_count]

# Merge all together on the specified keys with outer join (to keep all rows)
merged_final = reduce(
    lambda left, right: pd.merge(left, right, on=['Subject_Sorted', 'Timepoint_Sorted'], how='outer'),
    dfs
)

#grep Trial info from each Subject_Sorted
#Step 1: Extract unique mapping of Subject_Sorted to Trial

# Get unique mapping of Subject_Sorted to Trial
subject_trial_map = filtered_cell_df[['Subject_Sorted', 'Trial','timepoint_week_GT1.1', 'Timepoint_Sorted']].drop_duplicates()
#Step 2: Merge Trial into merged_final based on Subject_Sorted
merged_final_1 = merged_final.merge(
    subject_trial_map,
    on=['Subject_Sorted','Timepoint_Sorted'],
    how='left'     # or 'inner' if you want only matching Subjects
)


merged_final_1.to_csv(os.path.join(output_path,"sample_CD4_VRC01_BG505_count.csv"),index=False)

summary = (
    merged_final_1
    .groupby('Trial')
    .agg(
        n_rows=('Trial', 'count'),
        unique_subjects=('Subject_Sorted', 'nunique'),
        unique_timepoints=('Timepoint_Sorted', 'nunique')
    )
    .reset_index()
)

print(summary)
summary.to_csv(os.path.join(output_path,"Trial_sample_count.csv"),index=False)


