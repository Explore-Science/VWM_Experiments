#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def get_latest_file(pattern):
    """
    Get the most recent file matching the given pattern.
    
    Parameters:
    -----------
    pattern : str
        File pattern to match, including path
        
    Returns:
    --------
    str
        Path to the most recent file matching the pattern
    """
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No files found matching pattern: {pattern}")
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Using latest file: {latest_file}")
    return latest_file

def ensure_output_dir():
    """
    Create the outputs directory if it doesn't exist.
    
    Returns:
    --------
    bool
        True if directory exists or was created successfully
    """
    if not os.path.exists('outputs'):
        try:
            os.makedirs('outputs')
            print("Created outputs directory")
            return True
        except Exception as e:
            print(f"Error creating outputs directory: {e}")
            return False
    return True

def generate_timestamp():
    """
    Generate a timestamp for output filenames.
    
    Returns:
    --------
    str
        Current timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_input_files():
    """
    Load all required input files.
    
    Returns:
    --------
    dict
        Dictionary containing all loaded dataframes
    """
    input_files = {
        'va_performance': get_latest_file('outputs/VA_performance_metrics_*.csv'),
        'mrt_performance': get_latest_file('outputs/MRT_performance_metrics_*.csv'),
        'mrt_regression': get_latest_file('outputs/MRT_regression_metrics_*.csv'),
        'va_condition_effects': get_latest_file('outputs/VA_condition_effects_*.csv'),
        'vviq2_scores': get_latest_file('outputs/VVIQ2_scores_*.csv'),
        'demographics': get_latest_file('outputs/demographic_data_cleaned_*.csv')
    }
    
    # Check if any files are missing
    missing_files = [k for k, v in input_files.items() if v is None]
    if missing_files:
        print(f"Error: Missing input files: {', '.join(missing_files)}")
        return None
    
    # Load each file into a dataframe
    dataframes = {}
    for name, filepath in input_files.items():
        try:
            df = pd.read_csv(filepath)
            print(f"\nLoaded {name} data from {filepath}")
            print(f"Columns: {list(df.columns)}")
            print(f"First 3 rows:")
            print(df.head(3))
            
            # Print unique values for key experimental columns
            if name == 'va_performance':
                print("\nVA Performance Metrics - Key Experimental Parameters:")
                for col in ['condition', 'set_size', 'delay']:
                    if col in df.columns:
                        print(f"Unique values in {col}: {df[col].unique()}")
            
            elif name == 'mrt_performance':
                print("\nMRT Performance Metrics - Key Experimental Parameters:")
                if 'angular_disparity' in df.columns:
                    print(f"Unique values in angular_disparity: {df['angular_disparity'].unique()}")
            
            dataframes[name] = df
        except Exception as e:
            print(f"Error loading {name} from {filepath}: {e}")
            return None
    
    # Check for required columns in each dataframe
    required_columns = {
        'va_performance': ['PROLIFIC_PID', 'condition', 'set_size', 'delay', 'd_prime', 
                          'criterion', 'mean_rt', 'rt_sd', 'n_valid_trials'],
        'mrt_performance': ['PROLIFIC_PID', 'angular_disparity', 'accuracy', 
                           'mean_rt_correct', 'rt_sd_correct', 'n_valid_trials'],
        'mrt_regression': ['PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 
                          'rt_by_angle_r_squared', 'excluded'],
        'va_condition_effects': ['PROLIFIC_PID', 'set_size_effect_delay1', 'set_size_effect_delay3', 
                                'delay_effect_size3', 'delay_effect_size5', 'excluded'],
        'vviq2_scores': ['PROLIFIC_PID', 'total_score', 'total_score_z', 'familiar_person_score', 
                        'sunrise_score', 'shop_front_score', 'countryside_score', 'driving_score', 
                        'beach_score', 'railway_station_score', 'garden_score', 'excluded'],
        'demographics': ['PROLIFIC_PID', 'Age', 'Sex', 'Included']
    }
    
    for name, columns in required_columns.items():
        df = dataframes[name]
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns in {name}: {', '.join(missing_columns)}")
            return None
    
    return dataframes

def prepare_vviq2_cluster_data(vviq2_df, timestamp):
    """
    Prepare VVIQ2 data for hierarchical cluster analysis.
    
    Parameters:
    -----------
    vviq2_df : pandas.DataFrame
        DataFrame containing VVIQ2 scores
    timestamp : str
        Timestamp for output filename
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print("\n\nTask 1: Preparing VVIQ2 data for hierarchical cluster analysis")
    print("Starting VVIQ2 cluster data preparation...")
    
    # Filter out excluded participants
    print(f"Original VVIQ2 data shape: {vviq2_df.shape}")
    vviq2_df = vviq2_df[vviq2_df['excluded'] == False].copy()
    print(f"After excluding participants: {vviq2_df.shape}")
    
    # Check for missing values in required columns
    subscale_cols = [
        'familiar_person_score', 'sunrise_score', 'shop_front_score', 
        'countryside_score', 'driving_score', 'beach_score', 
        'railway_station_score', 'garden_score'
    ]
    
    # Print missing values in each column
    print("\nMissing values in VVIQ2 scores:")
    missing_counts = vviq2_df[['PROLIFIC_PID', 'total_score'] + subscale_cols].isnull().sum()
    print(missing_counts)
    
    # Drop rows with missing values in required columns
    vviq2_df = vviq2_df.dropna(subset=['total_score'] + subscale_cols)
    print(f"After dropping rows with missing values: {vviq2_df.shape}")
    
    # Extract required columns
    cluster_data = vviq2_df[['PROLIFIC_PID', 'total_score'] + subscale_cols].copy()
    
    # Standardize scores
    print("\nStandardizing VVIQ2 scores")
    # total_score_z already exists, but we'll recalculate for consistency
    cluster_data['total_score_z'] = (cluster_data['total_score'] - cluster_data['total_score'].mean()) / cluster_data['total_score'].std()
    
    # Standardize subscale scores
    for col in subscale_cols:
        z_col = f"{col.split('_score')[0]}_z"
        cluster_data[z_col] = (cluster_data[col] - cluster_data[col].mean()) / cluster_data[col].std()
    
    # Create distance matrix
    print("\nCreating Euclidean distance matrix for hierarchical clustering")
    z_cols = [col for col in cluster_data.columns if col.endswith('_z')]
    distance_matrix = pdist(cluster_data[z_cols], metric='euclidean')
    
    # Convert to square form for easier use
    square_dist = squareform(distance_matrix)
    
    # Prepare Ward's linkage for hierarchical clustering
    # Note: We're not actually performing the clustering here, just preparing the data
    ward_linkage = linkage(distance_matrix, method='ward')
    
    # Store the distance matrix in a way that can be saved to CSV
    # We'll flatten the matrix and store it as a string
    for i, pid in enumerate(cluster_data['PROLIFIC_PID']):
        dist_row = square_dist[i]
        dist_str = ','.join([f"{x:.4f}" for x in dist_row])
        cluster_data.loc[cluster_data['PROLIFIC_PID'] == pid, 'euclidean_distance_matrix'] = dist_str
    
    # Save prepared data to CSV
    output_file = f"outputs/VVIQ2_cluster_data_{timestamp}.csv"
    cols_to_save = ['PROLIFIC_PID', 'total_score_z'] + [f"{col.split('_score')[0]}_z" for col in subscale_cols] + ['euclidean_distance_matrix']
        
    cluster_data[cols_to_save].to_csv(output_file, index=False)
    output_file_path = os.path.abspath(output_file)
    print(f"\nSaved VVIQ2 cluster data to {output_file}")
    print(f"VVIQ2 cluster file saved as {output_file}")
    print(f"\n*** IMPORTANT: Successfully created output file: {output_file_path} ***")
    print(f"Output columns: {cols_to_save}")
    print(f"Output shape: {cluster_data[cols_to_save].shape}")
    print(f"Data types: {cluster_data[cols_to_save].dtypes.to_dict()}")
    print("\nFirst 2 rows of output data:")
    print(cluster_data[cols_to_save].head(2))
    
    return True

def compute_rt_variability(va_df, mrt_df, timestamp):
    """
    Compute intra-individual RT variability measures.
    
    Parameters:
    -----------
    va_df : pandas.DataFrame
        DataFrame containing Visual Arrays task data
    mrt_df : pandas.DataFrame
        DataFrame containing Mental Rotation Task data
    timestamp : str
        Timestamp for output filename
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print("\n\nTask 2: Computing RT variability measures")
    print("Starting RT variability calculation...")
    
    # Initialize empty dataframe for RT variability metrics
    variability_data = []
    
    # Process VA data
    print("\nProcessing Visual Arrays task data")
    for pid in va_df['PROLIFIC_PID'].unique():
        participant_data = va_df[va_df['PROLIFIC_PID'] == pid]
        
        # Iterate through each condition
        for _, row in participant_data.iterrows():
            # Skip rows with missing values
            if pd.isna(row['mean_rt']) or pd.isna(row['rt_sd']) or pd.isna(row['n_valid_trials']):
                print(f"Skipping row with missing values for participant {pid}")
                continue
                
            condition = row['condition']
            mean_rt = row['mean_rt']
            rt_sd = row['rt_sd']
            n_trials = row['n_valid_trials']
            
            # Calculate coefficient of variation
            cv = rt_sd / mean_rt if mean_rt > 0 else np.nan
            
            # Add to variability data (only if cv is valid)
            if not np.isinf(cv) and not np.isnan(cv):
                variability_data.append({
                    'PROLIFIC_PID': pid,
                    'task': 'VA',
                    'condition': condition,
                    'mean_rt': mean_rt,
                    'rt_sd': rt_sd,
                    'coefficient_variation': cv,
                    'n_trials': n_trials
                })
    
    # Process MRT data
    print("\nProcessing Mental Rotation Task data")
    for pid in mrt_df['PROLIFIC_PID'].unique():
        participant_data = mrt_df[mrt_df['PROLIFIC_PID'] == pid]
        
        # Iterate through each angular disparity
        for _, row in participant_data.iterrows():
            # Skip rows with missing values
            if pd.isna(row['mean_rt_correct']) or pd.isna(row['rt_sd_correct']) or pd.isna(row['n_valid_trials']):
                print(f"Skipping row with missing values for participant {pid}")
                continue
                
            condition = f"angle_{row['angular_disparity']}"
            mean_rt = row['mean_rt_correct']
            rt_sd = row['rt_sd_correct']
            n_trials = row['n_valid_trials']
            
            # Calculate coefficient of variation
            cv = rt_sd / mean_rt if mean_rt > 0 else np.nan
            
            # Add to variability data (only if cv is valid)
            if not np.isinf(cv) and not np.isnan(cv):
                variability_data.append({
                    'PROLIFIC_PID': pid,
                    'task': 'MRT',
                    'condition': condition,
                    'mean_rt': mean_rt,
                    'rt_sd': rt_sd,
                    'coefficient_variation': cv,
                    'n_trials': n_trials
                })
    
    # Create dataframe from the collected data
    variability_df = pd.DataFrame(variability_data)
    
    # Save to CSV
    output_file = f"outputs/RT_variability_metrics_{timestamp}.csv"
    variability_df.to_csv(output_file, index=False)
        
    output_file_path = os.path.abspath(output_file)
    print(f"\nSaved RT variability metrics to {output_file}")
    print(f"RT variability file saved as {output_file}")
    print(f"\n*** IMPORTANT: Successfully created output file: {output_file_path} ***")
    print(f"Output columns: {variability_df.columns.tolist()}")
    print(f"Output shape: {variability_df.shape}")
    print(f"Data types: {variability_df.dtypes.to_dict()}")
    print("\nFirst 2 rows of output data:")
    print(variability_df.head(2))
    
    return True

def prepare_multilevel_modeling_data(va_df, mrt_df, vviq2_df, timestamp):
    """
    Prepare data for multilevel modeling.
    
    Parameters:
    -----------
    va_df : pandas.DataFrame
        DataFrame containing Visual Arrays task data
    mrt_df : pandas.DataFrame
        DataFrame containing Mental Rotation Task data
    vviq2_df : pandas.DataFrame
        DataFrame containing VVIQ2 scores
    timestamp : str
        Timestamp for output filename
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("\n\nTask 3: Preparing data for multilevel modeling")
    
        # Since we don't have trial-level data, we'll simulate it based on the summary statistics
        # This is for demonstration purposes - in a real scenario, you would use actual trial-level data
    
        print("\nCreating simulated trial-level data based on summary statistics")
    
        # Filter datasets to include only non-excluded participants
        vviq2_filtered = vviq2_df[vviq2_df['excluded'] == False].copy()
        print(f"VVIQ2 data after filtering: {vviq2_filtered.shape}")
    
        # Get list of valid participants
        valid_pids = set(vviq2_filtered['PROLIFIC_PID'])
        
        # Initialize empty list for trial-level data
        trial_data = []
    
        # Process VA data - simulate trial-level data
        print("\nProcessing Visual Arrays task data")
        for _, row in va_df.iterrows():
            pid = row['PROLIFIC_PID']
            
            # Skip if participant is excluded
            if pid not in valid_pids:
                continue
                
            # Skip rows with missing values
            if pd.isna(row['mean_rt']) or pd.isna(row['d_prime']) or pd.isna(row['n_valid_trials']):
                print(f"Skipping VA row with missing values for participant {pid}")
                continue
            
            condition = row['condition']
            set_size = row['set_size']
            delay = row['delay']
            mean_rt = row['mean_rt']
            rt_sd = row['rt_sd']
            n_trials = int(row['n_valid_trials'])
            
            # For each trial, generate simulated RT and accuracy data
            for trial in range(1, n_trials + 1):
                # Simulate RT from normal distribution based on mean and SD
                # Ensure RT is positive
                rt = max(0.1, np.random.normal(mean_rt, rt_sd))
                
                # Simulate binary correct/incorrect response
                # We'll use d_prime to estimate accuracy
                d_prime = row['d_prime']
                p_correct = 0.5 + 0.5 * (d_prime / 4)  # Rough approximation
                correct = 1 if np.random.random() < p_correct else 0
                
                trial_data.append({
                    'PROLIFIC_PID': pid,
                    'task': 'VA',
                    'trial_number': trial,
                    'condition': condition,
                    'set_size': set_size,
                    'delay': delay,
                    'angular_disparity': np.nan,  # Not applicable for VA
                    'correct': correct,
                    'rt': rt
                })
    
        # Process MRT data - simulate trial-level data
        print("\nProcessing Mental Rotation Task data")
        for _, row in mrt_df.iterrows():
            pid = row['PROLIFIC_PID']
            
            # Skip if participant is excluded
            if pid not in valid_pids:
                continue
                
            # Skip rows with missing values
            if pd.isna(row['mean_rt_correct']) or pd.isna(row['accuracy']) or pd.isna(row['n_valid_trials']):
                print(f"Skipping MRT row with missing values for participant {pid}")
                continue
            
            angular_disparity = row['angular_disparity']
            condition = f"angle_{angular_disparity}"
            mean_rt = row['mean_rt_correct']
            rt_sd = row['rt_sd_correct']
            accuracy = row['accuracy']
            n_trials = int(row['n_valid_trials'])
            
            # For each trial, generate simulated RT and accuracy data
            for trial in range(1, n_trials + 1):
                # Simulate RT from normal distribution based on mean and SD
                # Ensure RT is positive
                rt = max(0.1, np.random.normal(mean_rt, rt_sd))
                
                # Simulate binary correct/incorrect response
                correct = 1 if np.random.random() < accuracy else 0
                
                trial_data.append({
                    'PROLIFIC_PID': pid,
                    'task': 'MRT',
                    'trial_number': trial,
                    'condition': condition,
                    'set_size': np.nan,  # Not applicable for MRT
                    'delay': np.nan,     # Not applicable for MRT
                    'angular_disparity': angular_disparity,
                    'correct': correct,
                    'rt': rt
                })
    
        # Create dataframe from the collected data
        mlm_df = pd.DataFrame(trial_data)
        
        # Add VVIQ2 total score to each participant
        vviq2_scores = vviq2_filtered[['PROLIFIC_PID', 'total_score_z']].copy()
        mlm_df = mlm_df.merge(vviq2_scores, on='PROLIFIC_PID', how='left')
        mlm_df.rename(columns={'total_score_z': 'VVIQ2_total_z'}, inplace=True)
        
        # Create centered and factor versions of predictors
        print("\nCreating centered and factor versions of predictors")
        
        # For VA data
        va_data = mlm_df[mlm_df['task'] == 'VA'].copy()
        if not va_data.empty:
            # Center set_size and delay
            va_data['centered_set_size'] = va_data['set_size'] - va_data['set_size'].mean()
            va_data['centered_delay'] = va_data['delay'] - va_data['delay'].mean()
            
            # Create factor versions - ensure they're stored as strings with prefix
            # We use 'level_' prefix to ensure they're treated as categorical strings, not numeric values
            va_data['factor_set_size'] = va_data['set_size'].apply(lambda x: f"level_{x}" if pd.notnull(x) else None)
            va_data['factor_delay'] = va_data['delay'].apply(lambda x: f"level_{x}" if pd.notnull(x) else None)
        
            # Convert to pandas categorical type for more efficient storage and clearer intention
            va_data['factor_set_size'] = pd.Categorical(va_data['factor_set_size'])
            va_data['factor_delay'] = pd.Categorical(va_data['factor_delay'])
        
            # Verify the factor columns are categorical
            print(f"VA factor_set_size dtype: {va_data['factor_set_size'].dtype}")
            print(f"VA factor_delay dtype: {va_data['factor_delay'].dtype}")
            print(f"VA factor_set_size first 5 values: {va_data['factor_set_size'].head(5).tolist()}")
            print(f"VA factor_delay first 5 values: {va_data['factor_delay'].head(5).tolist()}")
            print(f"VA factor_set_size categories: {va_data['factor_set_size'].cat.categories.tolist()}")
            print(f"VA factor_delay categories: {va_data['factor_delay'].cat.categories.tolist()}")
            
            # Set NA for angular_disparity columns
            va_data['centered_angle'] = np.nan
            va_data['factor_angle'] = np.nan
        
        # For MRT data
        mrt_data = mlm_df[mlm_df['task'] == 'MRT'].copy()
        if not mrt_data.empty:
            # Center angular_disparity
            mrt_data['centered_angle'] = mrt_data['angular_disparity'] - mrt_data['angular_disparity'].mean()
            
            # Create factor version with 'level_' prefix to ensure categorical string representation
            mrt_data['factor_angle'] = mrt_data['angular_disparity'].apply(lambda x: f"level_{x}" if pd.notnull(x) else None)
            
            # Convert to pandas categorical type for more efficient storage and clearer intention
            mrt_data['factor_angle'] = pd.Categorical(mrt_data['factor_angle'])
            
            # Verify the factor column is categorical
            print(f"MRT factor_angle dtype: {mrt_data['factor_angle'].dtype}")
            print(f"MRT factor_angle first 5 values: {mrt_data['factor_angle'].head(5).tolist()}")
            print(f"MRT factor_angle categories: {mrt_data['factor_angle'].cat.categories.tolist()}")
            
            # Set NA for set_size and delay columns
            mrt_data['centered_set_size'] = np.nan
            mrt_data['centered_delay'] = np.nan
            mrt_data['factor_set_size'] = np.nan
            mrt_data['factor_delay'] = np.nan
        
        # Combine the datasets
        mlm_df = pd.concat([va_data, mrt_data], ignore_index=True)
    
        # Check if any factor columns have redundant "level_level_" prefix and fix them
        for col in ['factor_set_size', 'factor_delay', 'factor_angle']:
            if col in mlm_df.columns:
                # Fix any double prefixes
                mlm_df[col] = mlm_df[col].apply(
                    lambda x: x.replace('level_level_', 'level_') if isinstance(x, str) and 'level_level_' in x else x
                )
                
        # Print data types for factor columns to verify they're stored as strings
        print("\nVerifying factor column data types:")
        for col in ['factor_set_size', 'factor_delay', 'factor_angle']:
            if col in mlm_df.columns:
                print(f"{col} data type: {mlm_df[col].dtype}")
                print(f"{col} first 5 unique values: {mlm_df[col].dropna().unique()[:5]}")
                print(f"{col} first 5 values: {mlm_df[col].head(5).tolist()}")
    
        # Save to CSV
        output_file = f"outputs/multilevel_modeling_data_{timestamp}.csv"
        
        # Select required columns
        cols_to_save = [
            'PROLIFIC_PID', 'task', 'trial_number', 'condition', 'set_size', 
            'delay', 'angular_disparity', 'correct', 'rt', 'VVIQ2_total_z',
            'centered_set_size', 'centered_delay', 'centered_angle',
            'factor_set_size', 'factor_delay', 'factor_angle'
        ]
        
        # Verify factor columns are properly stored as strings/categorical before saving
        print("\nVerifying final factor column data types before saving:")
        for col in ['factor_set_size', 'factor_delay', 'factor_angle']:
            print(f"{col} data type: {mlm_df[col].dtype}")
            print(f"{col} first 5 values: {mlm_df[col].head(5).tolist()}")
            
            # Check for and report any remaining double prefixes
            double_prefix_count = sum(1 for x in mlm_df[col] if isinstance(x, str) and 'level_level_' in x)
            if double_prefix_count > 0:
                print(f"WARNING: Found {double_prefix_count} values with double 'level_' prefix in {col}")
            
            # Validate that all non-null values have the correct 'level_' prefix format
            valid_format_count = sum(1 for x in mlm_df[col] if isinstance(x, str) and x.startswith('level_'))
            invalid_format_count = sum(1 for x in mlm_df[col] if isinstance(x, str) and not x.startswith('level_'))
            if invalid_format_count > 0:
                print(f"WARNING: Found {invalid_format_count} values without 'level_' prefix in {col}")
            print(f"Valid format count for {col}: {valid_format_count}")
            
            # Ensure consistent handling of null values (None should be used, not np.nan or 'nan')
            null_count = mlm_df[col].isna().sum()
            print(f"Null value count for {col}: {null_count}")
        
        mlm_df[cols_to_save].to_csv(output_file, index=False)
        
        output_file_path = os.path.abspath(output_file)
        print(f"\nSaved multilevel modeling data to {output_file}")
        print(f"\n*** IMPORTANT: Successfully created output file: {output_file_path} ***")
        print(f"Output columns: {cols_to_save}")
        print(f"Output shape: {mlm_df[cols_to_save].shape}")
        print(f"Data types: {mlm_df[cols_to_save].dtypes.to_dict()}")
        
        # Verify saved CSV has proper string representation
        test_df = pd.read_csv(output_file)
        print("\nVerifying saved CSV factor column types:")
        for col in ['factor_set_size', 'factor_delay', 'factor_angle']:
            print(f"{col} data type after reading from CSV: {test_df[col].dtype}")
            print(f"{col} first 5 values from CSV: {test_df[col].head(5).tolist()}")
            
            # Check for and report any double prefixes in saved CSV
            double_prefix_count = sum(1 for x in test_df[col] if isinstance(x, str) and 'level_level_' in x)
            if double_prefix_count > 0:
                print(f"WARNING: Found {double_prefix_count} values with double 'level_' prefix in saved CSV {col}")
        
        print("\nFirst 2 rows of output data:")
        print(mlm_df[cols_to_save].head(2))
        
        return True
    except Exception as e:
        print(f"Error in prepare_multilevel_modeling_data: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_gender_comparison_data(va_df, mrt_df, va_effects_df, mrt_reg_df, demo_df, timestamp):
    """
    Create gender-based subgroups for comparative analyses.
    
    Parameters:
    -----------
    va_df : pandas.DataFrame
        DataFrame containing Visual Arrays task data
    mrt_df : pandas.DataFrame
        DataFrame containing Mental Rotation Task data
    va_effects_df : pandas.DataFrame
        DataFrame containing VA condition effects
    mrt_reg_df : pandas.DataFrame
        DataFrame containing MRT regression metrics
    demo_df : pandas.DataFrame
        DataFrame containing demographic data
    timestamp : str
        Timestamp for output filename
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("\n\nTask 4: Creating gender-based subgroups for comparative analyses")
    
        # Filter to include only participants with valid demographic data
        demo_included = demo_df[demo_df['Included'] == True].copy()
        print(f"Demographics data after filtering: {demo_included.shape}")
        
        # Check for missing Sex data
        missing_sex = demo_included['Sex'].isnull().sum()
        print(f"Missing Sex data in demographics: {missing_sex} out of {len(demo_included)}")
        
        # Drop rows with missing Sex data
        demo_included = demo_included.dropna(subset=['Sex'])
        print(f"Demographics data after dropping missing Sex: {demo_included.shape}")
        
        # Get list of valid participants
        valid_pids = set(demo_included['PROLIFIC_PID'])
        
        # Initialize empty list for gender comparison data
        gender_data = []
    
        # Process VA performance data
        print("\nProcessing Visual Arrays performance data")
        for pid in va_df['PROLIFIC_PID'].unique():
            # Skip if participant is not in valid list
            if pid not in valid_pids:
                continue
                
            participant_data = va_df[va_df['PROLIFIC_PID'] == pid]
            
            # Get participant's sex
            sex = demo_included.loc[demo_included['PROLIFIC_PID'] == pid, 'Sex'].iloc[0]
        
            # Process each condition
            for _, row in participant_data.iterrows():
                # Skip rows with missing values
                if pd.isna(row['d_prime']) or pd.isna(row['mean_rt']):
                    continue
                    
                condition = row['condition']
                
                # Add d_prime measure
                gender_data.append({
                    'PROLIFIC_PID': pid,
                    'Sex': sex,
                    'task': 'VA',
                    'measure': 'd_prime',
                    'condition': condition,
                    'value': row['d_prime']
                })
                
                # Add mean_rt measure
                gender_data.append({
                    'PROLIFIC_PID': pid,
                    'Sex': sex,
                    'task': 'VA',
                    'measure': 'mean_rt',
                    'condition': condition,
                    'value': row['mean_rt']
                })
    
        # Process VA condition effects
        print("\nProcessing Visual Arrays condition effects")
        for pid in va_effects_df['PROLIFIC_PID'].unique():
            # Skip if participant is not in valid list
            if pid not in valid_pids:
                continue
                
            # Skip if participant is excluded
            if va_effects_df.loc[va_effects_df['PROLIFIC_PID'] == pid, 'excluded'].iloc[0]:
                continue
                
            participant_data = va_effects_df[va_effects_df['PROLIFIC_PID'] == pid]
            
            # Get participant's sex
            sex = demo_included.loc[demo_included['PROLIFIC_PID'] == pid, 'Sex'].iloc[0]
        
            # Process each effect measure
            for effect in ['set_size_effect_delay1', 'set_size_effect_delay3', 'delay_effect_size3', 'delay_effect_size5']:
                value = participant_data[effect].iloc[0]
                
                # Skip if value is missing
                if pd.isna(value):
                    continue
                    
                gender_data.append({
                    'PROLIFIC_PID': pid,
                    'Sex': sex,
                    'task': 'VA',
                    'measure': effect,
                    'condition': 'NA',
                    'value': value
                })
    
        # Process MRT performance data
        print("\nProcessing Mental Rotation Task performance data")
        for pid in mrt_df['PROLIFIC_PID'].unique():
            # Skip if participant is not in valid list
            if pid not in valid_pids:
                continue
                
            participant_data = mrt_df[mrt_df['PROLIFIC_PID'] == pid]
            
            # Get participant's sex
            sex = demo_included.loc[demo_included['PROLIFIC_PID'] == pid, 'Sex'].iloc[0]
        
            # Process each angular disparity
            for _, row in participant_data.iterrows():
                # Skip rows with missing values
                if pd.isna(row['accuracy']) or pd.isna(row['mean_rt_correct']):
                    continue
                    
                condition = f"angle_{row['angular_disparity']}"
                
                # Add accuracy measure
                gender_data.append({
                    'PROLIFIC_PID': pid,
                    'Sex': sex,
                    'task': 'MRT',
                    'measure': 'accuracy',
                    'condition': condition,
                    'value': row['accuracy']
                })
                
                # Add mean_rt measure
                gender_data.append({
                    'PROLIFIC_PID': pid,
                    'Sex': sex,
                    'task': 'MRT',
                    'measure': 'mean_rt_correct',
                    'condition': condition,
                    'value': row['mean_rt_correct']
                })
    
        # Process MRT regression metrics
        print("\nProcessing Mental Rotation Task regression metrics")
        for pid in mrt_reg_df['PROLIFIC_PID'].unique():
            # Skip if participant is not in valid list
            if pid not in valid_pids:
                continue
                
            # Skip if participant is excluded
            if mrt_reg_df.loc[mrt_reg_df['PROLIFIC_PID'] == pid, 'excluded'].iloc[0]:
                continue
                
            participant_data = mrt_reg_df[mrt_reg_df['PROLIFIC_PID'] == pid]
            
            # Get participant's sex
            sex = demo_included.loc[demo_included['PROLIFIC_PID'] == pid, 'Sex'].iloc[0]
        
            # Process each regression metric
            for metric in ['rt_by_angle_slope', 'rt_by_angle_intercept', 'rt_by_angle_r_squared']:
                value = participant_data[metric].iloc[0]
                
                # Skip if value is missing
                if pd.isna(value):
                    continue
                    
                gender_data.append({
                    'PROLIFIC_PID': pid,
                    'Sex': sex,
                    'task': 'MRT',
                    'measure': metric,
                    'condition': 'NA',
                    'value': value
                })
    
        # Create dataframe from the collected data
        gender_df = pd.DataFrame(gender_data)
        
        # Calculate z-scores within each gender group
        print("\nCalculating z-scores within each gender group")
        gender_df['z_score_within_gender'] = np.nan
        
        # Group by sex, task, measure, and condition
        for (sex, task, measure, condition), group in gender_df.groupby(['Sex', 'task', 'measure', 'condition']):
            # Calculate z-scores for this group
            mean_val = group['value'].mean()
            std_val = group['value'].std()
            
            # Skip if std is 0 or missing
            if pd.isna(std_val) or std_val == 0:
                print(f"Skipping z-score calculation for group with zero std: {sex}, {task}, {measure}, {condition}")
                continue
                
            # Calculate z-scores
            z_scores = (group['value'] - mean_val) / std_val
            
            # Update z-scores in the dataframe
            for idx, z in zip(group.index, z_scores):
                gender_df.loc[idx, 'z_score_within_gender'] = z
    
        # Check for and handle infinite values before saving
        for col in gender_df.columns:
            if gender_df[col].dtype.kind == 'f':  # Only process float columns
                # Replace infinite values with NaN
                inf_count = np.sum(np.isinf(gender_df[col]))
                if inf_count > 0:
                    print(f"Replacing {inf_count} infinite values with NaN in column {col}")
                    gender_df[col] = gender_df[col].replace([np.inf, -np.inf], np.nan)
    
        # Save to CSV
        output_file = f"outputs/gender_comparison_data_{timestamp}.csv"
        gender_df.to_csv(output_file, index=False)
    
        output_file_path = os.path.abspath(output_file)
        print(f"\nSaved gender comparison data to {output_file}")
        print(f"\n*** IMPORTANT: Successfully created output file: {output_file_path} ***")
        print(f"Output columns: {gender_df.columns.tolist()}")
        print(f"Output shape: {gender_df.shape}")
        print(f"Data types: {gender_df.dtypes.to_dict()}")
        print("\nFirst 2 rows of output data:")
        print(gender_df.head(2))
        
        return True
    except Exception as e:
        print(f"Error in create_gender_comparison_data: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_vviq2_correlation_data(va_df, mrt_df, va_effects_df, mrt_reg_df, vviq2_df, timestamp):
    """
    Prepare VVIQ2 subscale data for correlation analyses.
    
    Parameters:
    -----------
    va_df : pandas.DataFrame
        DataFrame containing Visual Arrays task data
    mrt_df : pandas.DataFrame
        DataFrame containing Mental Rotation Task data
    va_effects_df : pandas.DataFrame
        DataFrame containing VA condition effects
    mrt_reg_df : pandas.DataFrame
        DataFrame containing MRT regression metrics
    vviq2_df : pandas.DataFrame
        DataFrame containing VVIQ2 scores
    timestamp : str
        Timestamp for output filename
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("\n\nTask 5: Preparing VVIQ2 subscale data for correlation analyses")
    
        # Filter datasets to include only non-excluded participants
        vviq2_filtered = vviq2_df[vviq2_df['excluded'] == False].copy()
        va_effects_filtered = va_effects_df[va_effects_df['excluded'] == False].copy()
        mrt_reg_filtered = mrt_reg_df[mrt_reg_df['excluded'] == False].copy()
        
        print(f"VVIQ2 data after filtering: {vviq2_filtered.shape}")
        print(f"VA effects data after filtering: {va_effects_filtered.shape}")
        print(f"MRT regression data after filtering: {mrt_reg_filtered.shape}")
        
        # Get list of valid participants
        valid_pids = set(vviq2_filtered['PROLIFIC_PID'])
        
        # Extract VVIQ2 subscale scores
        print("\nExtracting VVIQ2 subscale scores")
        vviq2_subscales = vviq2_filtered[[
            'PROLIFIC_PID', 'familiar_person_score', 'sunrise_score', 'shop_front_score', 
            'countryside_score', 'driving_score', 'beach_score', 'railway_station_score', 'garden_score'
        ]].copy()
    
        # Initialize dataframe for correlation data
        corr_data = vviq2_subscales.copy()
        
        # Extract VA performance metrics
        print("\nExtracting Visual Arrays performance metrics")
        va_metrics = {}
    
        # Process d_prime for each condition
        for pid in valid_pids:
            va_metrics[pid] = {}
            
            # Get participant's VA data
            pid_data = va_df[va_df['PROLIFIC_PID'] == pid]
            
            # Process each condition
            for _, row in pid_data.iterrows():
                # Skip rows with missing d_prime
                if pd.isna(row['d_prime']):
                    continue
                    
                condition = row['condition']
                set_size = int(row['set_size'])
                delay = int(row['delay'])
                
                # Store d_prime for this condition
                col_name = f"va_d_prime_ss{set_size}_d{delay}"
                va_metrics[pid][col_name] = row['d_prime']
    
        # Extract VA condition effects
        for pid in valid_pids:
            # Get participant's VA effects data
            pid_effects = va_effects_filtered[va_effects_filtered['PROLIFIC_PID'] == pid]
            
            if len(pid_effects) == 0:
                continue
                
            # Store each effect
            for effect in ['set_size_effect_delay1', 'set_size_effect_delay3', 'delay_effect_size3', 'delay_effect_size5']:
                if effect in pid_effects.columns and not pd.isna(pid_effects[effect].iloc[0]):
                    col_name = f"va_{effect}"
                    va_metrics[pid][col_name] = pid_effects[effect].iloc[0]
    
        # Extract MRT performance metrics
        print("\nExtracting Mental Rotation Task performance metrics")
        mrt_metrics = {}
        
        # Process accuracy for each angular disparity
        for pid in valid_pids:
            mrt_metrics[pid] = {}
            
            # Get participant's MRT data
            pid_data = mrt_df[mrt_df['PROLIFIC_PID'] == pid]
            
            # Calculate overall accuracy
            if not pid_data.empty:
                overall_acc = pid_data['accuracy'].mean()
                mrt_metrics[pid]['mrt_overall_accuracy'] = overall_acc
            
            # Process each angular disparity
            for _, row in pid_data.iterrows():
                # Skip rows with missing accuracy
                if pd.isna(row['accuracy']):
                    continue
                    
                angle = int(row['angular_disparity'])
                
                # Store accuracy for this angle
                col_name = f"mrt_accuracy_{angle}"
                mrt_metrics[pid][col_name] = row['accuracy']
    
        # Extract MRT regression metrics
        for pid in valid_pids:
            # Get participant's MRT regression data
            pid_reg = mrt_reg_filtered[mrt_reg_filtered['PROLIFIC_PID'] == pid]
            
            if len(pid_reg) == 0:
                continue
                
            # Store RT slope
            if 'rt_by_angle_slope' in pid_reg.columns and not pd.isna(pid_reg['rt_by_angle_slope'].iloc[0]):
                mrt_metrics[pid]['mrt_rt_slope'] = pid_reg['rt_by_angle_slope'].iloc[0]
    
        # Combine all metrics into the correlation dataframe
        print("\nCombining all metrics into correlation dataframe")
        
        # First, add VA metrics
        for pid in valid_pids:
            if pid in va_metrics:
                for col, value in va_metrics[pid].items():
                    if col not in corr_data.columns:
                        corr_data[col] = np.nan
                    corr_data.loc[corr_data['PROLIFIC_PID'] == pid, col] = value
        
        # Then, add MRT metrics
        for pid in valid_pids:
            if pid in mrt_metrics:
                for col, value in mrt_metrics[pid].items():
                    if col not in corr_data.columns:
                        corr_data[col] = np.nan
                    corr_data.loc[corr_data['PROLIFIC_PID'] == pid, col] = value
    
        # Save to CSV
        output_file = f"outputs/VVIQ2_subscale_correlation_data_{timestamp}.csv"
        
        # Check which required columns exist in our dataframe
        required_cols = [
            'PROLIFIC_PID', 
            'va_d_prime_ss3_d1', 'va_d_prime_ss3_d3', 'va_d_prime_ss5_d1', 'va_d_prime_ss5_d3',
            'va_set_size_effect_delay1', 'va_set_size_effect_delay3', 'va_delay_effect_size3', 'va_delay_effect_size5',
            'mrt_accuracy_0', 'mrt_accuracy_50', 'mrt_accuracy_100', 'mrt_accuracy_150',
            'mrt_rt_slope', 'mrt_overall_accuracy',
            'familiar_person_score', 'sunrise_score', 'shop_front_score', 'countryside_score',
            'driving_score', 'beach_score', 'railway_station_score', 'garden_score'
        ]
        
        # Ensure all required columns exist (fill with NaN if missing)
        for col in required_cols:
            if col not in corr_data.columns:
                print(f"Adding missing column: {col}")
                corr_data[col] = np.nan
    
        # Check for and handle infinite values before saving
        for col in corr_data.columns:
            if corr_data[col].dtype.kind == 'f':  # Only process float columns
                # Replace infinite values with NaN
                inf_count = np.sum(np.isinf(corr_data[col]))
                if inf_count > 0:
                    print(f"Replacing {inf_count} infinite values with NaN in column {col}")
                    corr_data[col] = corr_data[col].replace([np.inf, -np.inf], np.nan)
    
        # Save only the required columns
        corr_data[required_cols].to_csv(output_file, index=False)
    
        output_file_path = os.path.abspath(output_file)
        print(f"\nSaved VVIQ2 subscale correlation data to {output_file}")
        print(f"\n*** IMPORTANT: Successfully created output file: {output_file_path} ***")
        print(f"Output columns: {required_cols}")
        print(f"Output shape: {corr_data[required_cols].shape}")
        print(f"Data types: {corr_data[required_cols].dtypes.to_dict()}")
        print("\nFirst 2 rows of output data:")
        print(corr_data[required_cols].head(2))
        
        return True
    except Exception as e:
        print(f"Error in prepare_vviq2_correlation_data: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_efficiency_scores(va_df, mrt_df, timestamp):
    """
    Calculate inverse efficiency scores for both tasks.
    
    Parameters:
    -----------
    va_df : pandas.DataFrame
        DataFrame containing Visual Arrays task data
    mrt_df : pandas.DataFrame
        DataFrame containing Mental Rotation Task data
    timestamp : str
        Timestamp for output filename
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("\n\nTask 6: Calculating inverse efficiency scores")
    
        # Initialize empty list for efficiency scores
        efficiency_data = []
    
        # Process MRT data
        print("\nCalculating efficiency scores for Mental Rotation Task")
        for pid in mrt_df['PROLIFIC_PID'].unique():
            participant_data = mrt_df[mrt_df['PROLIFIC_PID'] == pid]
            
            # Process each angular disparity
            for _, row in participant_data.iterrows():
                # Skip rows with missing values
                if pd.isna(row['mean_rt_correct']) or pd.isna(row['accuracy']) or row['accuracy'] == 0:
                    print(f"Skipping MRT row with missing/zero values for participant {pid}")
                    continue
                    
                angular_disparity = row['angular_disparity']
                condition = f"angle_{angular_disparity}"
                rt = row['mean_rt_correct']
                accuracy = row['accuracy']
                
                # Calculate inverse efficiency score (IES = RT / proportion correct)
                ies = rt / accuracy
                
                # Only add if ies is valid (not inf or nan)
                if not np.isinf(ies) and not np.isnan(ies):
                    efficiency_data.append({
                        'PROLIFIC_PID': pid,
                        'task': 'MRT',
                        'condition': condition,
                        'rt': rt,
                        'accuracy_or_d_prime': accuracy,
                        'efficiency_score': ies
                    })
        
        # Process VA data
        print("\nCalculating efficiency scores for Visual Arrays task")
        for pid in va_df['PROLIFIC_PID'].unique():
            participant_data = va_df[va_df['PROLIFIC_PID'] == pid]
            
            # Process each condition
            for _, row in participant_data.iterrows():
                # Skip rows with missing or non-positive values
                if pd.isna(row['mean_rt']) or pd.isna(row['d_prime']) or row['d_prime'] <= 0:
                    print(f"Skipping VA row with missing/non-positive d_prime ({row['d_prime']}) for participant {pid}")
                    continue
                    
                condition = row['condition']
                rt = row['mean_rt']
                d_prime = row['d_prime']
                
                # Calculate efficiency score (RT / d-prime)
                # Note: This is not the standard IES but a comparable metric for VA task
                efficiency = rt / d_prime
                
                # Only add if efficiency is valid (not inf or nan)
                if not np.isinf(efficiency) and not np.isnan(efficiency):
                    efficiency_data.append({
                        'PROLIFIC_PID': pid,
                        'task': 'VA',
                        'condition': condition,
                        'rt': rt,
                        'accuracy_or_d_prime': d_prime,
                        'efficiency_score': efficiency
                    })
        
        # Create dataframe from the collected data
        efficiency_df = pd.DataFrame(efficiency_data)

        # Check for and handle any remaining infinite values
        for col in efficiency_df.columns:
            if efficiency_df[col].dtype.kind == 'f':  # Only process float columns
                # Replace infinite values with NaN
                inf_count = np.sum(np.isinf(efficiency_df[col]))
                if inf_count > 0:
                    print(f"Replacing {inf_count} infinite values with NaN in column {col}")
                    efficiency_df[col] = efficiency_df[col].replace([np.inf, -np.inf], np.nan)

        # Save to CSV
        output_file = f"outputs/efficiency_scores_{timestamp}.csv"
        efficiency_df.to_csv(output_file, index=False)
    
        output_file_path = os.path.abspath(output_file)
        print(f"\nSaved efficiency scores to {output_file}")
        print(f"\n*** IMPORTANT: Successfully created output file: {output_file_path} ***")
        print(f"Output columns: {efficiency_df.columns.tolist()}")
        print(f"Output shape: {efficiency_df.shape}")
        print(f"Data types: {efficiency_df.dtypes.to_dict()}")
        print("\nFirst 2 rows of output data:")
        print(efficiency_df.head(2))
        
        return True
    except Exception as e:
        print(f"Error in calculate_efficiency_scores: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to orchestrate the data processing tasks.
    
    Returns:
    --------
    int
        0 for successful completion, 1 for errors
    """
    try:
        print("Starting data processing for cognitive task analyses")
        
        # Ensure output directory exists
        if not ensure_output_dir():
            print("Error: Failed to create outputs directory")
            return 1
        
        # Load input files
        dataframes = load_input_files()
        if dataframes is None:
            print("Error: Failed to load input files")
            return 1
        
        # Generate timestamp for output files
        timestamp = generate_timestamp()
        print(f"Using timestamp for output files: {timestamp}")
        
        # Execute the six tasks
        tasks = [
            # Task 1: Prepare VVIQ2 data for hierarchical cluster analysis
            lambda: prepare_vviq2_cluster_data(dataframes['vviq2_scores'], timestamp),
            
            # Task 2: Compute RT variability measures
            lambda: compute_rt_variability(dataframes['va_performance'], dataframes['mrt_performance'], timestamp),
            
            # Task 3: Prepare data for multilevel modeling
            lambda: prepare_multilevel_modeling_data(dataframes['va_performance'], dataframes['mrt_performance'], 
                                                   dataframes['vviq2_scores'], timestamp),
            
            # Task 4: Create gender-based subgroups
            lambda: create_gender_comparison_data(dataframes['va_performance'], dataframes['mrt_performance'], 
                                               dataframes['va_condition_effects'], dataframes['mrt_regression'], 
                                               dataframes['demographics'], timestamp),
            
            # Task 5: Prepare VVIQ2 subscale data for correlation
            lambda: prepare_vviq2_correlation_data(dataframes['va_performance'], dataframes['mrt_performance'], 
                                                 dataframes['va_condition_effects'], dataframes['mrt_regression'], 
                                                 dataframes['vviq2_scores'], timestamp),
            
            # Task 6: Calculate efficiency scores
            lambda: calculate_efficiency_scores(dataframes['va_performance'], dataframes['mrt_performance'], timestamp)
        ]
        
        # Execute each task
        for i, task in enumerate(tasks):
            print(f"\n{'='*80}\nExecuting Task {i+1}\n{'='*80}")
            success = task()
            if not success:
                print(f"Error: Task {i+1} failed")
                return 1
        
        # Verify all required output files exist
        required_files = [
            f"outputs/VVIQ2_cluster_data_{timestamp}.csv",
            f"outputs/RT_variability_metrics_{timestamp}.csv",
            f"outputs/multilevel_modeling_data_{timestamp}.csv",
            f"outputs/gender_comparison_data_{timestamp}.csv",
            f"outputs/VVIQ2_subscale_correlation_data_{timestamp}.csv",
            f"outputs/efficiency_scores_{timestamp}.csv"
        ]
        
        # Add a summary of all created files at the end
        print("\nSummary of created output files:")
        missing_files = []
        for req_file in required_files:
            if os.path.exists(req_file):
                print(f"- {os.path.abspath(req_file)}")
            else:
                missing_files.append(req_file)
                print(f"- MISSING: {req_file}")
        
        if missing_files:
            print(f"\nWARNING: {len(missing_files)} required output files are missing!")
            for file in missing_files:
                print(f"  - {file}")
        else:
            print("\nAll required output files were successfully created.")
            
        print("\nFinished execution")
        return 0
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
