import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.stats.multitest import multipletests

def find_most_recent_file(pattern):
    """
    Find the most recent file matching the given pattern.
    
    Parameters:
    -----------
    pattern : str
        File pattern to search for, including path
        
    Returns:
    --------
    str
        Path to the most recent file matching the pattern
    """
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    # Get the most recent file
    most_recent_file = max(matching_files, key=os.path.getctime)
    print(f"Most recent file for pattern '{pattern}': {most_recent_file}")
    return most_recent_file

def load_and_validate_va_performance(filepath):
    """
    Load and validate Visual Arrays performance metrics file.
    
    Parameters:
    -----------
    filepath : str
        Path to the VA performance metrics CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Validated VA performance data
    """
    print(f"\nLoading Visual Arrays performance metrics from: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data
    va_data = pd.read_csv(filepath)
    
    # Print columns and first three rows
    print("\nColumns in VA performance metrics:")
    print(va_data.columns.tolist())
    print("\nFirst three rows of VA performance metrics:")
    print(va_data.head(3))
    
    # Check required columns
    required_cols = ['PROLIFIC_PID', 'condition', 'set_size', 'delay', 'd_prime', 'mean_rt', 'n_valid_trials']
    missing_cols = [col for col in required_cols if col not in va_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in VA performance metrics: {missing_cols}")
    
    # Print unique values for key experimental design parameters
    print("\nUnique values in key columns:")
    print(f"condition: {va_data['condition'].unique().tolist()}")
    print(f"set_size: {va_data['set_size'].unique().tolist()}")
    print(f"delay: {va_data['delay'].unique().tolist()}")
    print(f"d_prime data type: {va_data['d_prime'].dtype}")
    print(f"mean_rt data type: {va_data['mean_rt'].dtype}")
    
    # Check for missing values in required columns
    missing_counts = va_data[required_cols].isnull().sum()
    print("\nMissing value counts in required columns:")
    print(missing_counts)
    
    # Filter out rows with missing values in key columns
    total_rows = len(va_data)
    va_data = va_data.dropna(subset=required_cols)
    filtered_rows = total_rows - len(va_data)
    print(f"\nFiltered out {filtered_rows} rows with missing values in required columns")
    
    return va_data

def load_and_validate_mrt_performance(filepath):
    """
    Load and validate Mental Rotation Task performance metrics file.
    
    Parameters:
    -----------
    filepath : str
        Path to the MRT performance metrics CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Validated MRT performance data
    """
    print(f"\nLoading Mental Rotation performance metrics from: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data
    mrt_data = pd.read_csv(filepath)
    
    # Print columns and first three rows
    print("\nColumns in MRT performance metrics:")
    print(mrt_data.columns.tolist())
    print("\nFirst three rows of MRT performance metrics:")
    print(mrt_data.head(3))
    
    # Check required columns
    required_cols = ['PROLIFIC_PID', 'angular_disparity', 'accuracy', 'mean_rt_correct', 'n_valid_trials']
    missing_cols = [col for col in required_cols if col not in mrt_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in MRT performance metrics: {missing_cols}")
    
    # Print unique values for key experimental design parameters
    print("\nUnique values in key columns:")
    print(f"angular_disparity: {mrt_data['angular_disparity'].unique().tolist()}")
    print(f"accuracy data type: {mrt_data['accuracy'].dtype}")
    print(f"mean_rt_correct data type: {mrt_data['mean_rt_correct'].dtype}")
    
    # Check for missing values in required columns
    missing_counts = mrt_data[required_cols].isnull().sum()
    print("\nMissing value counts in required columns:")
    print(missing_counts)
    
    # Filter out rows with missing values in key columns
    total_rows = len(mrt_data)
    mrt_data = mrt_data.dropna(subset=required_cols)
    filtered_rows = total_rows - len(mrt_data)
    print(f"\nFiltered out {filtered_rows} rows with missing values in required columns")
    
    return mrt_data

def load_and_validate_mrt_regression(filepath):
    """
    Load and validate Mental Rotation Task regression metrics file.
    
    Parameters:
    -----------
    filepath : str
        Path to the MRT regression metrics CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Validated MRT regression data
    """
    print(f"\nLoading Mental Rotation regression metrics from: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data
    mrt_reg_data = pd.read_csv(filepath)
    
    # Print columns and first three rows
    print("\nColumns in MRT regression metrics:")
    print(mrt_reg_data.columns.tolist())
    print("\nFirst three rows of MRT regression metrics:")
    print(mrt_reg_data.head(3))
    
    # Check required columns
    required_cols = ['PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 'excluded', 'exclusion_reason']
    missing_cols = [col for col in required_cols if col not in mrt_reg_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in MRT regression metrics: {missing_cols}")
    
    # Print unique values for excluded column
    print("\nUnique values in key columns:")
    print(f"excluded: {mrt_reg_data['excluded'].unique().tolist()}")
    print(f"rt_by_angle_slope data type: {mrt_reg_data['rt_by_angle_slope'].dtype}")
    print(f"rt_by_angle_intercept data type: {mrt_reg_data['rt_by_angle_intercept'].dtype}")
    
    # Check for missing values in required columns
    missing_counts = mrt_reg_data[required_cols].isnull().sum()
    print("\nMissing value counts in required columns:")
    print(missing_counts)
    
    # Filter out rows with missing values in key columns (except exclusion_reason which can be NA)
    total_rows = len(mrt_reg_data)
    mrt_reg_data = mrt_reg_data.dropna(subset=['PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 'excluded'])
    filtered_rows = total_rows - len(mrt_reg_data)
    print(f"\nFiltered out {filtered_rows} rows with missing values in required columns")
    
    return mrt_reg_data

def load_and_validate_va_condition_effects(filepath):
    """
    Load and validate Visual Arrays condition effects file.
    
    Parameters:
    -----------
    filepath : str
        Path to the VA condition effects CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Validated VA condition effects data
    """
    print(f"\nLoading Visual Arrays condition effects from: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data
    va_effects_data = pd.read_csv(filepath)
    
    # Print columns and first three rows
    print("\nColumns in VA condition effects:")
    print(va_effects_data.columns.tolist())
    print("\nFirst three rows of VA condition effects:")
    print(va_effects_data.head(3))
    
    # Check required columns
    required_cols = ['PROLIFIC_PID', 'set_size_effect_delay1', 'set_size_effect_delay3', 
                     'delay_effect_size3', 'delay_effect_size5', 'excluded', 'exclusion_reason']
    missing_cols = [col for col in required_cols if col not in va_effects_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in VA condition effects: {missing_cols}")
    
    # Print unique values for excluded column
    print("\nUnique values in key columns:")
    print(f"excluded: {va_effects_data['excluded'].unique().tolist()}")
    print(f"set_size_effect_delay1 data type: {va_effects_data['set_size_effect_delay1'].dtype}")
    print(f"set_size_effect_delay3 data type: {va_effects_data['set_size_effect_delay3'].dtype}")
    print(f"delay_effect_size3 data type: {va_effects_data['delay_effect_size3'].dtype}")
    print(f"delay_effect_size5 data type: {va_effects_data['delay_effect_size5'].dtype}")
    
    # Check for missing values in required columns
    missing_counts = va_effects_data[required_cols].isnull().sum()
    print("\nMissing value counts in required columns:")
    print(missing_counts)
    
    # Filter out rows with missing values in key columns (except exclusion_reason which can be NA)
    total_rows = len(va_effects_data)
    va_effects_data = va_effects_data.dropna(subset=['PROLIFIC_PID', 'set_size_effect_delay1', 
                                                    'set_size_effect_delay3', 'delay_effect_size3', 
                                                    'delay_effect_size5', 'excluded'])
    filtered_rows = total_rows - len(va_effects_data)
    print(f"\nFiltered out {filtered_rows} rows with missing values in required columns")
    
    return va_effects_data

def reshape_va_data(va_data):
    """
    Reshape Visual Arrays data to have one row per participant with columns for 
    each condition's d-prime.
    
    Parameters:
    -----------
    va_data : pandas.DataFrame
        Visual Arrays performance data
        
    Returns:
    --------
    pandas.DataFrame
        Reshaped Visual Arrays data
    """
    print("\nReshaping Visual Arrays data...")
    
    # Create a unique condition identifier
    va_data['condition_id'] = 'SS' + va_data['set_size'].astype(str) + '_D' + va_data['delay'].astype(str)
    
    # Pivot the data to have one row per participant
    va_wide = va_data.pivot(index='PROLIFIC_PID', 
                            columns='condition_id', 
                            values='d_prime')
    
    # Rename columns to be more descriptive
    va_wide.columns = ['d_prime_' + col for col in va_wide.columns]
    
    # Reset index to make PROLIFIC_PID a column
    va_wide = va_wide.reset_index()
    
    print("\nReshaped Visual Arrays data:")
    print(va_wide.columns.tolist())
    print(va_wide.head(2))
    
    return va_wide

def reshape_mrt_data(mrt_data):
    """
    Reshape Mental Rotation data to have one row per participant with columns for 
    accuracy and RT at each angular disparity.
    
    Parameters:
    -----------
    mrt_data : pandas.DataFrame
        Mental Rotation performance data
        
    Returns:
    --------
    pandas.DataFrame
        Reshaped Mental Rotation data
    """
    print("\nReshaping Mental Rotation data...")
    
    # Create a unique condition identifier for angular disparity
    mrt_data['angle_id'] = 'angle_' + mrt_data['angular_disparity'].astype(str)
    
    # Pivot the accuracy data
    mrt_acc_wide = mrt_data.pivot(index='PROLIFIC_PID', 
                                 columns='angle_id', 
                                 values='accuracy')
    
    # Rename columns to be more descriptive
    mrt_acc_wide.columns = ['accuracy_' + col for col in mrt_acc_wide.columns]
    
    # Pivot the RT data
    mrt_rt_wide = mrt_data.pivot(index='PROLIFIC_PID', 
                                columns='angle_id', 
                                values='mean_rt_correct')
    
    # Rename columns to be more descriptive
    mrt_rt_wide.columns = ['rt_' + col for col in mrt_rt_wide.columns]
    
    # Merge the accuracy and RT data
    mrt_wide = pd.merge(mrt_acc_wide, mrt_rt_wide, left_index=True, right_index=True)
    
    # Calculate overall accuracy and RT (average across all angles)
    mrt_data_grouped = mrt_data.groupby('PROLIFIC_PID').agg({
        'accuracy': 'mean',
        'mean_rt_correct': 'mean'
    }).rename(columns={
        'accuracy': 'accuracy_overall',
        'mean_rt_correct': 'rt_overall'
    })
    
    # Merge with the wide data
    mrt_wide = pd.merge(mrt_wide, mrt_data_grouped, left_index=True, right_index=True)
    
    # Reset index to make PROLIFIC_PID a column
    mrt_wide = mrt_wide.reset_index()
    
    print("\nReshaped Mental Rotation data:")
    print(mrt_wide.columns.tolist())
    print(mrt_wide.head(2))
    
    return mrt_wide

def merge_all_data(va_wide, mrt_wide, mrt_reg_data, va_effects_data):
    """
    Merge all datasets on PROLIFIC_PID and filter out excluded participants.
    
    Parameters:
    -----------
    va_wide : pandas.DataFrame
        Reshaped Visual Arrays data
    mrt_wide : pandas.DataFrame
        Reshaped Mental Rotation data
    mrt_reg_data : pandas.DataFrame
        Mental Rotation regression metrics
    va_effects_data : pandas.DataFrame
        Visual Arrays condition effects
        
    Returns:
    --------
    pandas.DataFrame
        Merged data with excluded participants filtered out
    """
    print("\nMerging all datasets...")
    
    # Merge all datasets
    merged_data = va_wide.merge(mrt_wide, on='PROLIFIC_PID', how='inner')
    merged_data = merged_data.merge(mrt_reg_data[['PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 'excluded']], 
                                   on='PROLIFIC_PID', how='inner', suffixes=('', '_mrt'))
    merged_data = merged_data.merge(va_effects_data[['PROLIFIC_PID', 'set_size_effect_delay1', 'set_size_effect_delay3', 
                                                   'delay_effect_size3', 'delay_effect_size5', 'excluded']], 
                                   on='PROLIFIC_PID', how='inner', suffixes=('', '_va'))
    
    print(f"\nMerged data shape before exclusions: {merged_data.shape}")
    print(f"Number of participants before exclusions: {merged_data['PROLIFIC_PID'].nunique()}")
    
    # Filter out excluded participants
    excluded_mrt = merged_data[merged_data['excluded'] == True]
    excluded_va = merged_data[merged_data['excluded_va'] == True]
    
    print(f"\nNumber of participants excluded from MRT: {len(excluded_mrt)}")
    print(f"Number of participants excluded from VA: {len(excluded_va)}")
    
    # Keep only non-excluded participants
    filtered_data = merged_data[(merged_data['excluded'] == False) & (merged_data['excluded_va'] == False)]
    
    print(f"\nMerged data shape after exclusions: {filtered_data.shape}")
    print(f"Number of participants after exclusions: {filtered_data['PROLIFIC_PID'].nunique()}")
    
    # Calculate average d-prime across all conditions for each participant
    d_prime_cols = [col for col in filtered_data.columns if col.startswith('d_prime_')]
    filtered_data['d_prime_average'] = filtered_data[d_prime_cols].mean(axis=1)
    
    # Calculate average accuracy and RT for small angles (0, 50) and large angles (100, 150)
    filtered_data['accuracy_small_angles'] = filtered_data[['accuracy_angle_0', 'accuracy_angle_50']].mean(axis=1)
    filtered_data['accuracy_large_angles'] = filtered_data[['accuracy_angle_100', 'accuracy_angle_150']].mean(axis=1)
    filtered_data['rt_small_angles'] = filtered_data[['rt_angle_0', 'rt_angle_50']].mean(axis=1)
    filtered_data['rt_large_angles'] = filtered_data[['rt_angle_100', 'rt_angle_150']].mean(axis=1)
    
    # Calculate average d-prime for different set sizes and delays
    filtered_data['d_prime_set_size_3'] = filtered_data[['d_prime_SS3_D1', 'd_prime_SS3_D3']].mean(axis=1)
    filtered_data['d_prime_set_size_5'] = filtered_data[['d_prime_SS5_D1', 'd_prime_SS5_D3']].mean(axis=1)
    filtered_data['d_prime_delay_1'] = filtered_data[['d_prime_SS3_D1', 'd_prime_SS5_D1']].mean(axis=1)
    filtered_data['d_prime_delay_3'] = filtered_data[['d_prime_SS3_D3', 'd_prime_SS5_D3']].mean(axis=1)
    
    print("\nMerged and processed data columns:")
    print(filtered_data.columns.tolist())
    print("\nFirst two rows of merged data:")
    print(filtered_data.head(2))
    
    return filtered_data

def calculate_descriptive_statistics(data):
    """
    Calculate descriptive statistics for all variables to be correlated.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged data
        
    Returns:
    --------
    pandas.DataFrame
        Descriptive statistics
    """
    print("\nCalculating descriptive statistics...")
    
    # List of measures to include in descriptives
    va_measures = [col for col in data.columns if col.startswith('d_prime_')]
    va_measures += ['set_size_effect_delay1', 'set_size_effect_delay3', 
                   'delay_effect_size3', 'delay_effect_size5']
    
    mrt_measures = [col for col in data.columns if col.startswith('accuracy_')]
    mrt_measures += [col for col in data.columns if col.startswith('rt_')]
    mrt_measures += ['rt_by_angle_slope', 'rt_by_angle_intercept']
    
    all_measures = va_measures + mrt_measures
    
    # Initialize lists to store results
    measure_list = []
    condition_list = []
    mean_list = []
    sd_list = []
    min_list = []
    max_list = []
    n_list = []
    
    # Calculate statistics for each measure
    for measure in all_measures:
        # Determine condition based on measure name
        if '_SS' in measure or 'set_size' in measure or 'delay_effect' in measure:
            condition = 'VA'
        elif 'angle' in measure or 'rt_by_angle' in measure:
            condition = 'MRT'
        else:
            condition = 'Overall'
        
        # Calculate statistics
        mean_val = data[measure].mean()
        sd_val = data[measure].std()
        min_val = data[measure].min()
        max_val = data[measure].max()
        n_val = data[measure].count()
        
        # Append to lists
        measure_list.append(measure)
        condition_list.append(condition)
        mean_list.append(mean_val)
        sd_list.append(sd_val)
        min_list.append(min_val)
        max_list.append(max_val)
        n_list.append(n_val)
    
    # Create dataframe
    descriptives = pd.DataFrame({
        'measure': measure_list,
        'condition': condition_list,
        'mean': mean_list,
        'sd': sd_list,
        'min': min_list,
        'max': max_list,
        'n': n_list
    })
    
    print("\nDescriptive statistics:")
    print(descriptives.head())
    
    return descriptives

def calculate_correlations(data):
    """
    Calculate Pearson correlations between Visual Arrays and Mental Rotation performance.
    
    This function computes correlations between VA d-prime measures and MRT accuracy/RT measures
    across all experimental conditions. It also calculates confidence intervals using
    Fisher's r-to-z transformation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged data containing all performance metrics from both tasks
        
    Returns:
    --------
    pandas.DataFrame
        Correlation results with columns for VA measure, VA condition, MRT measure,
        MRT condition, Pearson's r, p-value, confidence intervals, and sample size
    """
    print("\nCalculating correlations between VA and MRT measures...")
    
    # Define VA measures (d-prime for each condition)
    va_measures = {
        'd_prime_SS3_D1': 'SS3_D1',
        'd_prime_SS3_D3': 'SS3_D3',
        'd_prime_SS5_D1': 'SS5_D1',
        'd_prime_SS5_D3': 'SS5_D3'
    }
    
    # Define MRT accuracy measures
    mrt_acc_measures = {
        'accuracy_angle_0': 'angle_0',
        'accuracy_angle_50': 'angle_50',
        'accuracy_angle_100': 'angle_100',
        'accuracy_angle_150': 'angle_150'
    }
    
    # Define MRT RT measures
    mrt_rt_measures = {
        'rt_angle_0': 'angle_0',
        'rt_angle_50': 'angle_50',
        'rt_angle_100': 'angle_100',
        'rt_angle_150': 'angle_150'
    }
    
    # Initialize lists to store correlation results
    va_measure_list = []
    va_condition_list = []
    mrt_measure_list = []
    mrt_condition_list = []
    pearson_r_list = []
    p_value_list = []
    r_lower_ci_list = []
    r_upper_ci_list = []
    n_list = []
    
    # 1. Correlate d-prime with MRT accuracy at each angular disparity
    for va_col, va_cond in va_measures.items():
        for mrt_col, mrt_cond in mrt_acc_measures.items():
            # Calculate correlation
            r, p = stats.pearsonr(data[va_col], data[mrt_col])
            
            # Calculate confidence intervals using Fisher's z-transformation
            n = data[[va_col, mrt_col]].dropna().shape[0]
            # Handle edge cases where r is close to +/-1
            if abs(r) > 0.999:
                r = np.sign(r) * 0.999  # Cap at +/-0.999 to avoid infinity in arctanh
                print(f"Warning: Correlation coefficient capped at {r} to avoid numerical issues")
            
            z = np.arctanh(r)
            # Ensure n > 3 to avoid division by zero
            if n <= 3:
                print(f"Warning: Sample size {n} is too small for confidence intervals")
                se = float('nan')
            else:
                se = 1/np.sqrt(n-3)
            
            z_lower = z - 1.96*se
            z_upper = z + 1.96*se
            r_lower = np.tanh(z_lower)
            r_upper = np.tanh(z_upper)
            
            print(f"Correlation between {va_col} and {mrt_col}: r={r:.3f}, p={p:.3f}, n={n}")
            
            # Append to lists
            va_measure_list.append('d_prime')
            va_condition_list.append(va_cond)
            mrt_measure_list.append('accuracy')
            mrt_condition_list.append(mrt_cond)
            pearson_r_list.append(r)
            p_value_list.append(p)
            r_lower_ci_list.append(r_lower)
            r_upper_ci_list.append(r_upper)
            n_list.append(n)
    
    # 2. Correlate d-prime with MRT RT at each angular disparity
    for va_col, va_cond in va_measures.items():
        for mrt_col, mrt_cond in mrt_rt_measures.items():
            # Calculate correlation
            r, p = stats.pearsonr(data[va_col], data[mrt_col])
            
            # Calculate confidence intervals
            n = data[[va_col, mrt_col]].dropna().shape[0]
            # Handle edge cases where r is close to +/-1
            if abs(r) > 0.999:
                r = np.sign(r) * 0.999  # Cap at +/-0.999 to avoid infinity in arctanh
                print(f"Warning: Correlation coefficient capped at {r} to avoid numerical issues")
            
            z = np.arctanh(r)
            # Ensure n > 3 to avoid division by zero
            if n <= 3:
                print(f"Warning: Sample size {n} is too small for confidence intervals")
                se = float('nan')
            else:
                se = 1/np.sqrt(n-3)
            
            z_lower = z - 1.96*se
            z_upper = z + 1.96*se
            r_lower = np.tanh(z_lower)
            r_upper = np.tanh(z_upper)
            
            # Append to lists
            va_measure_list.append('d_prime')
            va_condition_list.append(va_cond)
            mrt_measure_list.append('rt')
            mrt_condition_list.append(mrt_cond)
            pearson_r_list.append(r)
            p_value_list.append(p)
            r_lower_ci_list.append(r_lower)
            r_upper_ci_list.append(r_upper)
            n_list.append(n)
    
    # 3. Correlate VA set size effects with MRT RT-by-angle slope
    va_effect_measures = {
        'set_size_effect_delay1': 'set_size_effect_delay1',
        'set_size_effect_delay3': 'set_size_effect_delay3',
        'delay_effect_size3': 'delay_effect_size3',
        'delay_effect_size5': 'delay_effect_size5'
    }
    
    for va_col, va_cond in va_effect_measures.items():
        # Calculate correlation with RT slope
        r, p = stats.pearsonr(data[va_col], data['rt_by_angle_slope'])
        
        # Calculate confidence intervals
        n = data[[va_col, 'rt_by_angle_slope']].dropna().shape[0]
        z = np.arctanh(r)
        se = 1/np.sqrt(n-3)
        z_lower = z - 1.96*se
        z_upper = z + 1.96*se
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        
        # Append to lists
        va_measure_list.append(va_col)
        va_condition_list.append('effect')
        mrt_measure_list.append('rt_by_angle_slope')
        mrt_condition_list.append('overall')
        pearson_r_list.append(r)
        p_value_list.append(p)
        r_lower_ci_list.append(r_lower)
        r_upper_ci_list.append(r_upper)
        n_list.append(n)
    
    # Create dataframe with correlation results
    corr_results = pd.DataFrame({
        'VA_measure': va_measure_list,
        'VA_condition': va_condition_list,
        'MRT_measure': mrt_measure_list,
        'MRT_condition': mrt_condition_list,
        'pearson_r': pearson_r_list,
        'p_value': p_value_list,
        'r_lower_ci': r_lower_ci_list,
        'r_upper_ci': r_upper_ci_list,
        'n': n_list
    })
    
    print("\nCorrelation results (first few rows):")
    print(corr_results.head())
    
    return corr_results

def apply_fdr_correction(corr_results):
    """
    Apply False Discovery Rate (FDR) correction to correlation p-values.
    
    Parameters:
    -----------
    corr_results : pandas.DataFrame
        Correlation results
        
    Returns:
    --------
    pandas.DataFrame
        Correlation results with FDR correction
    """
    print("\nApplying FDR correction...")
    
    # Extract p-values for d-prime correlations (first 32 correlations)
    d_prime_corrs = corr_results[corr_results['VA_measure'] == 'd_prime']
    
    # Apply FDR correction
    _, p_corrected, _, _ = multipletests(d_prime_corrs['p_value'], method='fdr_bh', alpha=0.05)
    
    # Add corrected p-values to results
    corr_results.loc[corr_results['VA_measure'] == 'd_prime', 'p_value_fdr_corrected'] = p_corrected
    
    # For other correlations, no FDR correction
    corr_results.loc[corr_results['VA_measure'] != 'd_prime', 'p_value_fdr_corrected'] = corr_results.loc[corr_results['VA_measure'] != 'd_prime', 'p_value']
    
    # Add significance flags
    corr_results['significant_uncorrected'] = corr_results['p_value'] < 0.05
    corr_results['significant_fdr_corrected'] = corr_results['p_value_fdr_corrected'] < 0.05
    
    # Count significant correlations
    sig_uncorrected = corr_results['significant_uncorrected'].sum()
    sig_corrected = corr_results['significant_fdr_corrected'].sum()
    
    print(f"\nNumber of significant correlations (uncorrected): {sig_uncorrected}")
    print(f"Number of significant correlations (FDR-corrected): {sig_corrected}")
    print(f"Expected by chance (at alpha = 0.05): {len(corr_results) * 0.05:.1f}")
    
    return corr_results

def test_hypothesis1(data):
    """
    Test Hypothesis 1: Correlation between average d-prime and MRT measures.
    
    This function tests the hypothesis that there is a significant positive correlation
    between Visual Arrays performance (d-prime) and Mental Rotation Task accuracy,
    and a significant negative correlation between d-prime and RT-by-angle slope.
    
    The hypothesis is considered supported if:
    1. The correlation between d-prime and accuracy is ≥ 0.30
    2. The correlation between d-prime and RT-by-angle slope is ≤ -0.30
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged data containing all performance metrics from both tasks
        
    Returns:
    --------
    pandas.DataFrame
        Hypothesis 1 test results with correlation pairs, Pearson's r, p-values,
        confidence intervals, and whether the hypothesis is supported
    """
    print("\nTesting Hypothesis 1...")
    
    # Initialize lists to store results
    correlation_pair_list = []
    pearson_r_list = []
    p_value_list = []
    r_lower_ci_list = []
    r_upper_ci_list = []
    hypothesis_supported_list = []
    n_list = []
    
    # 1. Correlate average d-prime with overall MRT accuracy
    r, p = stats.pearsonr(data['d_prime_average'], data['accuracy_overall'])
    
    # Calculate confidence intervals
    n = data[['d_prime_average', 'accuracy_overall']].dropna().shape[0]
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z_lower = z - 1.96*se
    z_upper = z + 1.96*se
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    
    # Check if hypothesis is supported (r ≥ 0.30)
    hypothesis_supported = r >= 0.30
    
    # Append to lists
    correlation_pair_list.append('d_prime_average vs. accuracy_overall')
    pearson_r_list.append(r)
    p_value_list.append(p)
    r_lower_ci_list.append(r_lower)
    r_upper_ci_list.append(r_upper)
    hypothesis_supported_list.append(hypothesis_supported)
    n_list.append(n)
    
    # 2. Correlate average d-prime with RT-by-angle slope
    r, p = stats.pearsonr(data['d_prime_average'], data['rt_by_angle_slope'])
    
    # Calculate confidence intervals
    n = data[['d_prime_average', 'rt_by_angle_slope']].dropna().shape[0]
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z_lower = z - 1.96*se
    z_upper = z + 1.96*se
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    
    # Check if hypothesis is supported (r ≤ -0.30)
    hypothesis_supported = r <= -0.30
    
    # Append to lists
    correlation_pair_list.append('d_prime_average vs. rt_by_angle_slope')
    pearson_r_list.append(r)
    p_value_list.append(p)
    r_lower_ci_list.append(r_lower)
    r_upper_ci_list.append(r_upper)
    hypothesis_supported_list.append(hypothesis_supported)
    n_list.append(n)
    
    # Create dataframe with results
    h1_results = pd.DataFrame({
        'correlation_pair': correlation_pair_list,
        'pearson_r': pearson_r_list,
        'p_value': p_value_list,
        'r_lower_ci': r_lower_ci_list,
        'r_upper_ci': r_upper_ci_list,
        'hypothesis_supported': hypothesis_supported_list,
        'n': n_list
    })
    
    print("\nHypothesis 1 results:")
    print(h1_results)
    
    return h1_results

def test_hypothesis2(data):
    """
    Test Hypothesis 2: Modulation by task complexity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged data
        
    Returns:
    --------
    pandas.DataFrame
        Hypothesis 2 test results
    """
    print("\nTesting Hypothesis 2...")
    
    # Initialize lists to store results
    comparison_list = []
    r1_list = []
    r2_list = []
    z_score_list = []
    p_value_list = []
    r_difference_list = []
    hypothesis_supported_list = []
    n_list = []
    
    # 1. Compare set size 3 vs. set size 5 correlations with MRT accuracy
    r1, _ = stats.pearsonr(data['d_prime_set_size_3'], data['accuracy_overall'])
    r2, _ = stats.pearsonr(data['d_prime_set_size_5'], data['accuracy_overall'])
    
    # Fisher's r-to-z transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    n = data[['d_prime_set_size_3', 'd_prime_set_size_5', 'accuracy_overall']].dropna().shape[0]
    
    # Calculate z-score and p-value
    se_diff = np.sqrt(1/(n-3) + 1/(n-3))
    z_score = (z1 - z2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if hypothesis is supported (difference in r ≥ 0.15)
    r_difference = abs(r1 - r2)
    hypothesis_supported = r_difference >= 0.15
    
    # Append to lists
    comparison_list.append('set_size_3 vs. set_size_5 (accuracy)')
    r1_list.append(r1)
    r2_list.append(r2)
    z_score_list.append(z_score)
    p_value_list.append(p_value)
    r_difference_list.append(r_difference)
    hypothesis_supported_list.append(hypothesis_supported)
    n_list.append(n)
    
    # 2. Compare 1s delay vs. 3s delay correlations with MRT accuracy
    r1, _ = stats.pearsonr(data['d_prime_delay_1'], data['accuracy_overall'])
    r2, _ = stats.pearsonr(data['d_prime_delay_3'], data['accuracy_overall'])
    
    # Fisher's r-to-z transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    n = data[['d_prime_delay_1', 'd_prime_delay_3', 'accuracy_overall']].dropna().shape[0]
    
    # Calculate z-score and p-value
    se_diff = np.sqrt(1/(n-3) + 1/(n-3))
    z_score = (z1 - z2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if hypothesis is supported
    r_difference = abs(r1 - r2)
    hypothesis_supported = r_difference >= 0.15
    
    # Append to lists
    comparison_list.append('delay_1 vs. delay_3 (accuracy)')
    r1_list.append(r1)
    r2_list.append(r2)
    z_score_list.append(z_score)
    p_value_list.append(p_value)
    r_difference_list.append(r_difference)
    hypothesis_supported_list.append(hypothesis_supported)
    n_list.append(n)
    
    # 3. Compare smaller angles vs. larger angles correlations with d-prime
    r1, _ = stats.pearsonr(data['d_prime_average'], data['accuracy_small_angles'])
    r2, _ = stats.pearsonr(data['d_prime_average'], data['accuracy_large_angles'])
    
    # Fisher's r-to-z transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    n = data[['d_prime_average', 'accuracy_small_angles', 'accuracy_large_angles']].dropna().shape[0]
    
    # Calculate z-score and p-value
    se_diff = np.sqrt(1/(n-3) + 1/(n-3))
    z_score = (z1 - z2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if hypothesis is supported
    r_difference = abs(r1 - r2)
    hypothesis_supported = r_difference >= 0.15
    
    # Append to lists
    comparison_list.append('small_angles vs. large_angles (accuracy)')
    r1_list.append(r1)
    r2_list.append(r2)
    z_score_list.append(z_score)
    p_value_list.append(p_value)
    r_difference_list.append(r_difference)
    hypothesis_supported_list.append(hypothesis_supported)
    n_list.append(n)
    
    # 4. Repeat comparisons for RT
    # Set size comparison with RT
    r1, _ = stats.pearsonr(data['d_prime_set_size_3'], data['rt_overall'])
    r2, _ = stats.pearsonr(data['d_prime_set_size_5'], data['rt_overall'])
    
    # Fisher's r-to-z transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    n = data[['d_prime_set_size_3', 'd_prime_set_size_5', 'rt_overall']].dropna().shape[0]
    
    # Calculate z-score and p-value
    se_diff = np.sqrt(1/(n-3) + 1/(n-3))
    z_score = (z1 - z2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if hypothesis is supported
    r_difference = abs(r1 - r2)
    hypothesis_supported = r_difference >= 0.15
    
    # Append to lists
    comparison_list.append('set_size_3 vs. set_size_5 (RT)')
    r1_list.append(r1)
    r2_list.append(r2)
    z_score_list.append(z_score)
    p_value_list.append(p_value)
    r_difference_list.append(r_difference)
    hypothesis_supported_list.append(hypothesis_supported)
    n_list.append(n)
    
    # Create dataframe with results
    h2_results = pd.DataFrame({
        'comparison': comparison_list,
        'r1': r1_list,
        'r2': r2_list,
        'z_score': z_score_list,
        'p_value': p_value_list,
        'r_difference': r_difference_list,
        'hypothesis_supported': hypothesis_supported_list,
        'n': n_list
    })
    
    print("\nHypothesis 2 results:")
    print(h2_results)
    
    return h2_results

def create_correlation_heatmap(data, corr_results):
    """
    Create visualization matrices of correlations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged data containing all performance metrics
    corr_results : pandas.DataFrame
        Correlation results with VA and MRT measures
        
    Returns:
    --------
    None
        Saves heatmap visualizations to output files
    """
    print("\nCreating correlation heatmaps...")
    
    try:
        # Extract d-prime correlations with accuracy
        acc_corrs = corr_results[(corr_results['VA_measure'] == 'd_prime') & 
                                (corr_results['MRT_measure'] == 'accuracy')]
        
        print(f"Creating accuracy correlation heatmap with {len(acc_corrs)} correlations")
        
        # Reshape to matrix form
        acc_matrix = acc_corrs.pivot(index='VA_condition', 
                                   columns='MRT_condition', 
                                   values='pearson_r')
        
        # Create heatmap for accuracy correlations
        plt.figure(figsize=(10, 8))
        sns.heatmap(acc_matrix, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5, 
                    center=0, fmt='.2f')
        plt.title('Correlations between VA d-prime and MRT Accuracy')
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save figure
        accuracy_heatmap_path = 'outputs/accuracy_correlation_heatmap.png'
        plt.savefig(accuracy_heatmap_path)
        print(f"Saved accuracy correlation heatmap to: {accuracy_heatmap_path}")
        
        # Extract d-prime correlations with RT
        rt_corrs = corr_results[(corr_results['VA_measure'] == 'd_prime') & 
                               (corr_results['MRT_measure'] == 'rt')]
        
        print(f"Creating RT correlation heatmap with {len(rt_corrs)} correlations")
        
        # Reshape to matrix form
        rt_matrix = rt_corrs.pivot(index='VA_condition', 
                                 columns='MRT_condition', 
                                 values='pearson_r')
        
        # Create heatmap for RT correlations
        plt.figure(figsize=(10, 8))
        sns.heatmap(rt_matrix, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5, 
                    center=0, fmt='.2f')
        plt.title('Correlations between VA d-prime and MRT Response Time')
        plt.tight_layout()
        
        # Save figure
        rt_heatmap_path = 'outputs/rt_correlation_heatmap.png'
        plt.savefig(rt_heatmap_path)
        print(f"Saved RT correlation heatmap to: {rt_heatmap_path}")
        
        print("\nHeatmaps saved to outputs/ directory")
    except Exception as e:
        print(f"Error creating correlation heatmaps: {str(e)}")
        print("Continuing with analysis despite visualization error")

def main():
    """
    Main function to execute the cross-task correlation analysis.
    """
    print("Starting cross-task correlation analysis...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find input files
        va_perf_file = find_most_recent_file("outputs/VA_performance_metrics_*.csv")
        mrt_perf_file = find_most_recent_file("outputs/MRT_performance_metrics_*.csv")
        mrt_reg_file = find_most_recent_file("outputs/MRT_regression_metrics_*.csv")
        va_effects_file = find_most_recent_file("outputs/VA_condition_effects_*.csv")
        
        # Load and validate data
        va_data = load_and_validate_va_performance(va_perf_file)
        mrt_data = load_and_validate_mrt_performance(mrt_perf_file)
        mrt_reg_data = load_and_validate_mrt_regression(mrt_reg_file)
        va_effects_data = load_and_validate_va_condition_effects(va_effects_file)
        
        # Reshape data
        va_wide = reshape_va_data(va_data)
        mrt_wide = reshape_mrt_data(mrt_data)
        
        # Merge all data and filter out excluded participants
        merged_data = merge_all_data(va_wide, mrt_wide, mrt_reg_data, va_effects_data)
        
        # Calculate descriptive statistics
        descriptives = calculate_descriptive_statistics(merged_data)
        
        # Calculate correlations
        corr_results = calculate_correlations(merged_data)
        
        # Apply FDR correction
        corr_results = apply_fdr_correction(corr_results)
        
        # Test hypotheses
        h1_results = test_hypothesis1(merged_data)
        h2_results = test_hypothesis2(merged_data)
        
        # Create visualization
        create_correlation_heatmap(merged_data, corr_results)
        
        # Save results to CSV files
        output_files = [
            f'outputs/cross_task_correlations_{timestamp}.csv',
            f'outputs/hypothesis1_results_{timestamp}.csv',
            f'outputs/hypothesis2_results_{timestamp}.csv',
            f'outputs/correlation_descriptives_{timestamp}.csv'
        ]
        
        corr_results.to_csv(output_files[0], index=False)
        h1_results.to_csv(output_files[1], index=False)
        h2_results.to_csv(output_files[2], index=False)
        descriptives.to_csv(output_files[3], index=False)
        
        print(f"\nResults saved to CSV files with timestamp {timestamp}")
        for file_path in output_files:
            print(f"Saved output file: {file_path}")
        print("Finished execution")
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
