#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
from datetime import datetime
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import warnings
import traceback
from statsmodels.stats.multitest import fdrcorrection
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

def get_latest_file(pattern):
    """
    Get the most recent file matching the given pattern.
    
    Parameters:
    -----------
    pattern : str
        The file pattern to match, including path and wildcards.
        
    Returns:
    --------
    str
        The path to the most recent file matching the pattern.
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Latest file found for pattern '{pattern}': {latest_file}")
    return latest_file

def create_output_directory():
    """
    Create the outputs directory if it doesn't exist.
    """
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory")
    else:
        print("'outputs' directory already exists")

def load_va_performance_metrics():
    """
    Load the VA performance metrics data.
    
    Returns:
    --------
    pandas.DataFrame
        The VA performance metrics data with required columns.
    """
    file_path = get_latest_file('outputs/VA_performance_metrics_*.csv')
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print column names and first few rows
    print("\nVA Performance Metrics Columns:")
    print(df.columns.tolist())
    print("\nVA Performance Metrics First 3 Rows:")
    print(df.head(3))
    
    # Print unique values for key experimental design parameters
    print("\nUnique values for key experimental design parameters:")
    print(f"condition: {df['condition'].unique().tolist()}")
    print(f"set_size: {df['set_size'].unique().tolist()}")
    print(f"delay: {df['delay'].unique().tolist()}")
    print(f"d_prime data type: {df['d_prime'].dtype}")
    print(f"mean_rt data type: {df['mean_rt'].dtype}")
    
    # Check for required columns
    required_columns = ['PROLIFIC_PID', 'condition', 'set_size', 'delay', 'd_prime', 'mean_rt', 'n_valid_trials']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in VA performance metrics: {missing_columns}")
    
    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values in VA performance metrics required columns:")
    print(missing_values)
    
    # Select only required columns
    df = df[required_columns]
    
    return df

def load_mrt_performance_metrics():
    """
    Load the MRT performance metrics data.
    
    Returns:
    --------
    pandas.DataFrame
        The MRT performance metrics data with required columns.
    """
    file_path = get_latest_file('outputs/MRT_performance_metrics_*.csv')
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print column names and first few rows
    print("\nMRT Performance Metrics Columns:")
    print(df.columns.tolist())
    print("\nMRT Performance Metrics First 3 Rows:")
    print(df.head(3))
    
    # Print unique values for key experimental design parameters
    print("\nUnique values for key experimental design parameters:")
    print(f"angular_disparity: {df['angular_disparity'].unique().tolist()}")
    print(f"accuracy data type: {df['accuracy'].dtype}")
    print(f"mean_rt_correct data type: {df['mean_rt_correct'].dtype}")
    
    # Check for required columns
    required_columns = ['PROLIFIC_PID', 'angular_disparity', 'accuracy', 'mean_rt_correct', 'n_valid_trials']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in MRT performance metrics: {missing_columns}")
    
    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values in MRT performance metrics required columns:")
    print(missing_values)
    
    # Select only required columns
    df = df[required_columns]
    
    return df

def load_mrt_regression_metrics():
    """
    Load the MRT regression metrics data.
    
    Returns:
    --------
    pandas.DataFrame
        The MRT regression metrics data with required columns.
    """
    file_path = get_latest_file('outputs/MRT_regression_metrics_*.csv')
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print column names and first few rows
    print("\nMRT Regression Metrics Columns:")
    print(df.columns.tolist())
    print("\nMRT Regression Metrics First 3 Rows:")
    print(df.head(3))
    
    # Print data types for key metrics
    print("\nData types for key metrics:")
    print(f"rt_by_angle_slope data type: {df['rt_by_angle_slope'].dtype}")
    print(f"rt_by_angle_intercept data type: {df['rt_by_angle_intercept'].dtype}")
    print(f"rt_by_angle_r_squared data type: {df['rt_by_angle_r_squared'].dtype}")
    print(f"excluded data type: {df['excluded'].dtype}")
    
    # Check for required columns
    required_columns = ['PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 
                        'rt_by_angle_r_squared', 'excluded', 'exclusion_reason']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in MRT regression metrics: {missing_columns}")
    
    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values in MRT regression metrics required columns:")
    print(missing_values)
    
    # Select only required columns
    df = df[required_columns]
    
    return df

def load_va_condition_effects():
    """
    Load the VA condition effects data.
    
    Returns:
    --------
    pandas.DataFrame
        The VA condition effects data with required columns.
    """
    file_path = get_latest_file('outputs/VA_condition_effects_*.csv')
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print column names and first few rows
    print("\nVA Condition Effects Columns:")
    print(df.columns.tolist())
    print("\nVA Condition Effects First 3 Rows:")
    print(df.head(3))
    
    # Print data types for key metrics
    print("\nData types for key metrics:")
    print(f"set_size_effect_delay1 data type: {df['set_size_effect_delay1'].dtype}")
    print(f"set_size_effect_delay3 data type: {df['set_size_effect_delay3'].dtype}")
    print(f"delay_effect_size3 data type: {df['delay_effect_size3'].dtype}")
    print(f"delay_effect_size5 data type: {df['delay_effect_size5'].dtype}")
    print(f"excluded data type: {df['excluded'].dtype}")
    
    # Check for required columns
    required_columns = ['PROLIFIC_PID', 'set_size_effect_delay1', 'set_size_effect_delay3',
                        'delay_effect_size3', 'delay_effect_size5', 'excluded', 'exclusion_reason']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in VA condition effects: {missing_columns}")
    
    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values in VA condition effects required columns:")
    print(missing_values)
    
    # Select only required columns
    df = df[required_columns]
    
    return df

def load_vviq2_scores():
    """
    Load the VVIQ2 scores data.
    
    Returns:
    --------
    pandas.DataFrame
        The VVIQ2 scores data with required columns.
    """
    file_path = get_latest_file('outputs/VVIQ2_scores_*.csv')
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print column names and first few rows
    print("\nVVIQ2 Scores Columns:")
    print(df.columns.tolist())
    print("\nVVIQ2 Scores First 3 Rows:")
    print(df.head(3))
    
    # Print data types for key metrics
    print("\nData types for key metrics:")
    print(f"total_score data type: {df['total_score'].dtype}")
    print(f"total_score_z data type: {df['total_score_z'].dtype}")
    print(f"excluded data type: {df['excluded'].dtype}")
    
    # Check for required columns
    required_columns = ['PROLIFIC_PID', 'total_score', 'total_score_z', 'familiar_person_score',
                        'sunrise_score', 'shop_front_score', 'countryside_score', 'driving_score',
                        'beach_score', 'railway_station_score', 'garden_score', 'excluded', 'exclusion_reason']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in VVIQ2 scores: {missing_columns}")
    
    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values in VVIQ2 scores required columns:")
    print(missing_values)
    
    # Select only required columns
    df = df[required_columns]
    
    return df

def merge_and_prepare_data(va_perf, mrt_perf, mrt_reg, va_effects, vviq2):
    """
    Merge all datasets and prepare for analysis.
    
    Parameters:
    -----------
    va_perf : pandas.DataFrame
        VA performance metrics data.
    mrt_perf : pandas.DataFrame
        MRT performance metrics data.
    mrt_reg : pandas.DataFrame
        MRT regression metrics data.
    va_effects : pandas.DataFrame
        VA condition effects data.
    vviq2 : pandas.DataFrame
        VVIQ2 scores data.
        
    Returns:
    --------
    tuple
        (merged_data, va_wide, mrt_acc_wide, mrt_rt_wide)
        - merged_data: DataFrame with all participants and key measures
        - va_wide: VA data in wide format for ANCOVA
        - mrt_acc_wide: MRT accuracy data in wide format for ANCOVA
        - mrt_rt_wide: MRT RT data in wide format for ANCOVA
    """
    # Filter out excluded participants from each dataset
    print("\nFiltering out excluded participants...")
    
    # Convert 'excluded' to boolean if it's not already
    for df in [mrt_reg, va_effects, vviq2]:
        if df['excluded'].dtype != bool:
            # Handle various possible formats
            if df['excluded'].dtype == 'object':
                df['excluded'] = df['excluded'].map({'TRUE': True, 'True': True, 'true': True, 
                                                    'FALSE': False, 'False': False, 'false': False})
            else:
                df['excluded'] = df['excluded'].astype(bool)
    
    # Print exclusion counts before filtering
    print(f"MRT regression excluded count: {mrt_reg['excluded'].sum()} of {len(mrt_reg)}")
    print(f"VA effects excluded count: {va_effects['excluded'].sum()} of {len(va_effects)}")
    print(f"VVIQ2 excluded count: {vviq2['excluded'].sum()} of {len(vviq2)}")
    
    # Get list of excluded participants
    excluded_pids = set()
    excluded_pids.update(mrt_reg.loc[mrt_reg['excluded'], 'PROLIFIC_PID'].tolist())
    excluded_pids.update(va_effects.loc[va_effects['excluded'], 'PROLIFIC_PID'].tolist())
    excluded_pids.update(vviq2.loc[vviq2['excluded'], 'PROLIFIC_PID'].tolist())
    
    print(f"Total unique excluded participants: {len(excluded_pids)}")
    
    # Filter out excluded participants from all datasets
    va_perf = va_perf[~va_perf['PROLIFIC_PID'].isin(excluded_pids)]
    mrt_perf = mrt_perf[~mrt_perf['PROLIFIC_PID'].isin(excluded_pids)]
    mrt_reg = mrt_reg[~mrt_reg['excluded']]
    va_effects = va_effects[~va_effects['excluded']]
    vviq2 = vviq2[~vviq2['excluded']]
    
    print(f"Participants remaining after exclusions: {len(vviq2)}")
    
    # Calculate overall MRT accuracy and average effects
    print("\nCalculating derived measures...")
    
    # Calculate overall MRT accuracy for each participant
    mrt_overall = mrt_perf.groupby('PROLIFIC_PID')['accuracy'].mean().reset_index()
    mrt_overall.rename(columns={'accuracy': 'mrt_overall_accuracy'}, inplace=True)
    
    # Calculate average set size and delay effects
    va_effects['avg_set_size_effect'] = (va_effects['set_size_effect_delay1'] + 
                                         va_effects['set_size_effect_delay3']) / 2
    va_effects['avg_delay_effect'] = (va_effects['delay_effect_size3'] + 
                                      va_effects['delay_effect_size5']) / 2
    
    # Create a base merged dataset with key measures
    merged = pd.merge(vviq2[['PROLIFIC_PID', 'total_score', 'total_score_z']], 
                      va_effects[['PROLIFIC_PID', 'set_size_effect_delay1', 'set_size_effect_delay3',
                                 'delay_effect_size3', 'delay_effect_size5', 
                                 'avg_set_size_effect', 'avg_delay_effect']], 
                      on='PROLIFIC_PID', how='inner')
    
    merged = pd.merge(merged, mrt_reg[['PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 
                                     'rt_by_angle_r_squared']], 
                     on='PROLIFIC_PID', how='inner')
    
    merged = pd.merge(merged, mrt_overall, on='PROLIFIC_PID', how='inner')
    
    print(f"Final merged dataset contains {len(merged)} participants")
    print("\nMerged dataset preview:")
    print(merged.head(2))
    
    # Prepare VA data in wide format for ANCOVA
    va_wide = va_perf.pivot_table(
        index='PROLIFIC_PID',
        columns=['set_size', 'delay'],
        values='d_prime'
    ).reset_index()
    
    # Rename columns to make them more readable
    va_wide.columns = ['PROLIFIC_PID'] + [f'ss{ss}_d{d}' for ss, d in va_wide.columns[1:]]
    
    # Merge with VVIQ2 scores
    va_wide = pd.merge(va_wide, vviq2[['PROLIFIC_PID', 'total_score', 'total_score_z']], 
                      on='PROLIFIC_PID', how='inner')
    
    print("\nVA wide format dataset preview:")
    print(va_wide.head(2))
    
    # Prepare MRT accuracy data in wide format for ANCOVA
    mrt_acc_wide = mrt_perf.pivot_table(
        index='PROLIFIC_PID',
        columns='angular_disparity',
        values='accuracy'
    ).reset_index()
    
    # Rename columns
    mrt_acc_wide.columns = ['PROLIFIC_PID'] + [f'acc_ang{int(ang)}' for ang in mrt_acc_wide.columns[1:]]
    
    # Merge with VVIQ2 scores
    mrt_acc_wide = pd.merge(mrt_acc_wide, vviq2[['PROLIFIC_PID', 'total_score', 'total_score_z']], 
                           on='PROLIFIC_PID', how='inner')
    
    print("\nMRT accuracy wide format dataset preview:")
    print(mrt_acc_wide.head(2))
    
    # Prepare MRT RT data in wide format for ANCOVA
    mrt_rt_wide = mrt_perf.pivot_table(
        index='PROLIFIC_PID',
        columns='angular_disparity',
        values='mean_rt_correct'
    ).reset_index()
    
    # Rename columns
    mrt_rt_wide.columns = ['PROLIFIC_PID'] + [f'rt_ang{int(ang)}' for ang in mrt_rt_wide.columns[1:]]
    
    # Merge with VVIQ2 scores
    mrt_rt_wide = pd.merge(mrt_rt_wide, vviq2[['PROLIFIC_PID', 'total_score', 'total_score_z']], 
                          on='PROLIFIC_PID', how='inner')
    
    print("\nMRT RT wide format dataset preview:")
    print(mrt_rt_wide.head(2))
    
    # Check for missing values in key variables
    print("\nChecking for missing values in merged dataset:")
    print(merged.isnull().sum())
    
    # Drop any rows with missing values in key variables
    merged_complete = merged.dropna()
    print(f"Complete cases in merged dataset: {len(merged_complete)} of {len(merged)}")
    
    return merged_complete, va_wide, mrt_acc_wide, mrt_rt_wide

def calculate_correlation_with_ci(data, x_var, y_var, alpha=0.05):
    """
    Calculate Pearson correlation with confidence intervals.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing the variables.
    x_var : str
        Name of the first variable.
    y_var : str
        Name of the second variable.
    alpha : float, optional
        Significance level (default: 0.05).
        
    Returns:
    --------
    dict
        Dictionary containing correlation results.
    """
    # Calculate Pearson correlation
    r, p = stats.pearsonr(data[x_var], data[y_var])
    
    # Calculate 95% confidence interval using Fisher's z-transformation
    n = len(data)
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z_crit = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = z-z_crit*se, z+z_crit*se
    lo, hi = np.tanh(lo_z), np.tanh(hi_z)
    
    # Check if correlation meets predicted threshold (r ≥ 0.25)
    hypothesis_supported = abs(r) >= 0.25
    
    return {
        'pearson_r': r,
        'p_value': p,
        'r_lower_ci': lo,
        'r_upper_ci': hi,
        'hypothesis_supported': hypothesis_supported,
        'n': n
    }

def test_hypothesis_3a(data):
    """
    Test Hypothesis 3a - VVIQ2 and set size effects.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged dataset containing VVIQ2 scores and VA effects.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing correlation results.
    """
    print("\nTesting Hypothesis 3a - VVIQ2 and set size effects...")
    
    # Define variables to correlate
    correlations = []
    
    # Correlate VVIQ2 with set size effects
    for effect_var, effect_desc in [
        ('set_size_effect_delay1', 'Set size effect at delay 1s'),
        ('set_size_effect_delay3', 'Set size effect at delay 3s'),
        ('avg_set_size_effect', 'Average set size effect')
    ]:
        # Calculate correlation with total VVIQ2 score
        result = calculate_correlation_with_ci(data, 'total_score', effect_var)
        
        # Add to results
        correlations.append({
            'VA_measure': effect_var,
            'VA_condition': effect_desc,
            'VVIQ2_measure': 'total_score',
            **result
        })
        
        print(f"Correlation between VVIQ2 total score and {effect_desc}:")
        print(f"  r = {result['pearson_r']:.3f}, p = {result['p_value']:.3f}")
        print(f"  95% CI: [{result['r_lower_ci']:.3f}, {result['r_upper_ci']:.3f}]")
        print(f"  Hypothesis supported: {result['hypothesis_supported']}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(correlations)
    
    # Plot scatterplots with loess curves
    for effect_var, effect_desc in [
        ('set_size_effect_delay1', 'Set size effect at delay 1s'),
        ('set_size_effect_delay3', 'Set size effect at delay 3s'),
        ('avg_set_size_effect', 'Average set size effect')
    ]:
        plt.figure(figsize=(8, 6))
        sns.regplot(x='total_score', y=effect_var, data=data, 
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        sns.regplot(x='total_score', y=effect_var, data=data, 
                   scatter=False, lowess=True, line_kws={'color': 'blue'})
        
        plt.title(f'VVIQ2 Total Score vs {effect_desc}')
        plt.xlabel('VVIQ2 Total Score')
        plt.ylabel(effect_desc)
        
        r, p = stats.pearsonr(data['total_score'], data[effect_var])
        plt.annotate(f'r = {r:.3f}, p = {p:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'outputs/VVIQ2_vs_{effect_var}_{timestamp}.png')
        plt.close()
    
    return results_df

def test_hypothesis_3b(data):
    """
    Test Hypothesis 3b - VVIQ2 and Mental Rotation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged dataset containing VVIQ2 scores and MRT metrics.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing correlation results.
    """
    print("\nTesting Hypothesis 3b - VVIQ2 and Mental Rotation...")
    
    # Define variables to correlate
    correlations = []
    
    # Correlate VVIQ2 with MRT metrics
    for mrt_var, mrt_desc in [
        ('rt_by_angle_slope', 'RT-by-angle slope'),
        ('mrt_overall_accuracy', 'Overall Mental Rotation accuracy')
    ]:
        # Calculate correlation with total VVIQ2 score
        result = calculate_correlation_with_ci(data, 'total_score', mrt_var)
        
        # Add to results
        correlations.append({
            'MRT_measure': mrt_var,
            'MRT_condition': mrt_desc,
            'VVIQ2_measure': 'total_score',
            **result
        })
        
        print(f"Correlation between VVIQ2 total score and {mrt_desc}:")
        print(f"  r = {result['pearson_r']:.3f}, p = {result['p_value']:.3f}")
        print(f"  95% CI: [{result['r_lower_ci']:.3f}, {result['r_upper_ci']:.3f}]")
        print(f"  Hypothesis supported: {result['hypothesis_supported']}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(correlations)
    
    # Plot scatterplots with loess curves
    for mrt_var, mrt_desc in [
        ('rt_by_angle_slope', 'RT-by-angle slope'),
        ('mrt_overall_accuracy', 'Overall Mental Rotation accuracy')
    ]:
        plt.figure(figsize=(8, 6))
        sns.regplot(x='total_score', y=mrt_var, data=data, 
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        sns.regplot(x='total_score', y=mrt_var, data=data, 
                   scatter=False, lowess=True, line_kws={'color': 'blue'})
        
        plt.title(f'VVIQ2 Total Score vs {mrt_desc}')
        plt.xlabel('VVIQ2 Total Score')
        plt.ylabel(mrt_desc)
        
        r, p = stats.pearsonr(data['total_score'], data[mrt_var])
        plt.annotate(f'r = {r:.3f}, p = {p:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'outputs/VVIQ2_vs_{mrt_var}_{timestamp}.png')
        plt.close()
    
    return results_df

def conduct_va_ancova(va_wide):
    """
    Conduct repeated-measures ANCOVA for Visual Arrays Task.
    
    Parameters:
    -----------
    va_wide : pandas.DataFrame
        VA data in wide format with VVIQ2 scores.
        
    Returns:
    --------
    tuple
        (ancova_results, simple_slopes_results)
    """
    print("\nConducting repeated-measures ANCOVA for Visual Arrays Task...")
    
    # Activate R
    pandas2ri.activate()
    
    # Import required R packages
    base = importr('base')
    stats_r = importr('stats')
    ez = importr('ez')
    emmeans = importr('emmeans')
    
    # Convert wide format to long format for R
    va_long = pd.melt(
        va_wide,
        id_vars=['PROLIFIC_PID', 'total_score', 'total_score_z'],
        value_vars=[col for col in va_wide.columns if col.startswith('ss')],
        var_name='condition',
        value_name='d_prime'
    )
    
    # Extract set size and delay from condition column
    va_long['set_size'] = va_long['condition'].str.extract(r'ss(\d)').astype(int)
    va_long['delay'] = va_long['condition'].str.extract(r'd(\d)').astype(int)
    
    print("\nVA long format dataset preview:")
    print(va_long.head(2))
    
    # Convert to R dataframe
    va_long_r = pandas2ri.py2rpy(va_long)
    
    # Create factors in R
    ro.r('''
    prepare_data <- function(data) {
        data$PROLIFIC_PID <- as.factor(data$PROLIFIC_PID)
        data$set_size <- as.factor(data$set_size)
        data$delay <- as.factor(data$delay)
        return(data)
    }
    ''')
    
    va_long_r = ro.r['prepare_data'](va_long_r)
    
    # Run the ANCOVA in R
    ro.r('''
    run_ancova <- function(data) {
        # Add error handling to prevent crashes
        tryCatch({
            # Run the ANCOVA using ezANOVA
            model <- ezANOVA(
                data = data,
                dv = .(d_prime),
                wid = .(PROLIFIC_PID),
                within = .(set_size, delay),
                between = .(total_score_z),
                type = 3,
                detailed = TRUE
            )
            
            # Format the results
            result <- data.frame(
                effect = character(),
                F_value = numeric(),
                df_numerator = numeric(),
                df_denominator = numeric(),
                p_value = numeric(),
                partial_eta_squared = numeric(),
                partial_eta_squared_lower_ci = numeric(),
                partial_eta_squared_upper_ci = numeric(),
                significance_indicator = character(),
                stringsAsFactors = FALSE
            )
            
            # Extract main effects and interactions
            for (i in 1:nrow(model$ANOVA)) {
                effect_name <- rownames(model$ANOVA)[i]
                F_val <- model$ANOVA$F[i]
                df_num <- model$ANOVA$DFn[i]
                df_den <- model$ANOVA$DFd[i]
                p_val <- model$ANOVA$p[i]
                ges <- model$ANOVA$ges[i]
                
                # Calculate confidence intervals for partial eta squared
                # Using Fisher's z transformation approximation
                z <- 0.5 * log((1 + sqrt(ges)) / (1 - sqrt(ges)))
                se <- 1 / sqrt(df_num + df_den - 1)
                z_lower <- z - 1.96 * se
                z_upper <- z + 1.96 * se
                ges_lower <- (exp(2 * z_lower) - 1)^2 / (exp(2 * z_lower) + 1)^2
                ges_upper <- (exp(2 * z_upper) - 1)^2 / (exp(2 * z_upper) + 1)^2
                
                # Determine significance indicator
                sig_indicator <- ""
                if (p_val < 0.001) sig_indicator <- "***"
                else if (p_val < 0.01) sig_indicator <- "**"
                else if (p_val < 0.05) sig_indicator <- "*"
                else if (p_val < 0.1) sig_indicator <- "."
                
                result <- rbind(result, data.frame(
                    effect = effect_name,
                    F_value = F_val,
                    df_numerator = df_num,
                    df_denominator = df_den,
                    p_value = p_val,
                    partial_eta_squared = ges,
                    partial_eta_squared_lower_ci = ges_lower,
                    partial_eta_squared_upper_ci = ges_upper,
                    significance_indicator = sig_indicator
                ))
            }
            
            return(result)
        }, error = function(e) {
            # Print error message
            print(paste("Error in R VA ANCOVA analysis:", e$message))
            
            # Return empty dataframe with correct structure
            empty_result <- data.frame(
                effect = "Error",
                F_value = NA,
                df_numerator = NA,
                df_denominator = NA,
                p_value = NA,
                partial_eta_squared = NA,
                partial_eta_squared_lower_ci = NA,
                partial_eta_squared_upper_ci = NA,
                significance_indicator = "",
                stringsAsFactors = FALSE
            )
            return(empty_result)
        })
    }
    ''')
    
    # Run the ANCOVA
    ancova_results_r = ro.r['run_ancova'](va_long_r)
    
    # Convert back to pandas DataFrame
    ancova_results = pandas2ri.rpy2py(ancova_results_r)
    
    print("\nANCOVA Results:")
    print(ancova_results)
    
    # Check for significant interactions involving VVIQ2
    vviq2_interactions = ancova_results[
        (ancova_results['effect'].str.contains('total_score_z')) & 
        (ancova_results['p_value'] < 0.05)
    ]
    
    # If there are significant interactions, conduct simple slopes analyses
    simple_slopes_results = pd.DataFrame(columns=[
        'task', 'effect', 'VVIQ2_level', 'estimate', 'std_error', 
        't_value', 'p_value', 'significance_indicator'
    ])
    
    if not vviq2_interactions.empty:
        print("\nConducting simple slopes analyses for significant VVIQ2 interactions...")
        
        # Define R function for simple slopes analysis
        ro.r('''
        simple_slopes_analysis <- function(data) {
            # Create emmeans model
            model <- lm(d_prime ~ set_size * delay * total_score_z, data = data)
            
            # Define levels for VVIQ2 (-1 SD, Mean, +1 SD)
            vviq_levels <- c(-1, 0, 1)
            
            # Create empty dataframe for results
            results <- data.frame(
                task = character(),
                effect = character(),
                VVIQ2_level = character(),
                estimate = numeric(),
                std_error = numeric(),
                t_value = numeric(),
                p_value = numeric(),
                significance_indicator = character(),
                stringsAsFactors = FALSE
            )
            
            # For each VVIQ2 level
            for (vviq_level in vviq_levels) {
                level_name <- ifelse(vviq_level == -1, "Low VVIQ2 (-1 SD)", 
                                   ifelse(vviq_level == 0, "Medium VVIQ2 (Mean)", 
                                          "High VVIQ2 (+1 SD)"))
                
                # Set size effect at each delay
                for (d in c(1, 3)) {
                    # Create reference grid
                    rg <- ref_grid(model, at = list(total_score_z = vviq_level, delay = as.character(d)))
                    
                    # Get EMMs for set size
                    emm_ss <- emmeans(rg, ~ set_size)
                    
                    # Get contrasts
                    cont <- contrast(emm_ss, method = "pairwise")
                    
                    # Extract results
                    for (i in 1:nrow(cont)) {
                        p_val <- summary(cont)$p.value[i]
                        
                        # Determine significance indicator
                        sig_indicator <- ""
                        if (p_val < 0.001) sig_indicator <- "***"
                        else if (p_val < 0.01) sig_indicator <- "**"
                        else if (p_val < 0.05) sig_indicator <- "*"
                        else if (p_val < 0.1) sig_indicator <- "."
                        
                        results <- rbind(results, data.frame(
                            task = "VA",
                            effect = paste0("Set size effect at delay ", d, "s"),
                            VVIQ2_level = level_name,
                            estimate = summary(cont)$estimate[i],
                            std_error = summary(cont)$SE[i],
                            t_value = summary(cont)$t.ratio[i],
                            p_value = p_val,
                            significance_indicator = sig_indicator
                        ))
                    }
                }
                
                # Delay effect at each set size
                for (ss in c(3, 5)) {
                    # Create reference grid
                    rg <- ref_grid(model, at = list(total_score_z = vviq_level, set_size = as.character(ss)))
                    
                    # Get EMMs for delay
                    emm_delay <- emmeans(rg, ~ delay)
                    
                    # Get contrasts
                    cont <- contrast(emm_delay, method = "pairwise")
                    
                    # Extract results
                    for (i in 1:nrow(cont)) {
                        p_val <- summary(cont)$p.value[i]
                        
                        # Determine significance indicator
                        sig_indicator <- ""
                        if (p_val < 0.001) sig_indicator <- "***"
                        else if (p_val < 0.01) sig_indicator <- "**"
                        else if (p_val < 0.05) sig_indicator <- "*"
                        else if (p_val < 0.1) sig_indicator <- "."
                        
                        results <- rbind(results, data.frame(
                            task = "VA",
                            effect = paste0("Delay effect at set size ", ss),
                            VVIQ2_level = level_name,
                            estimate = summary(cont)$estimate[i],
                            std_error = summary(cont)$SE[i],
                            t_value = summary(cont)$t.ratio[i],
                            p_value = p_val,
                            significance_indicator = sig_indicator
                        ))
                    }
                }
            }
            
            return(results)
        }
        ''')
        
        # Run simple slopes analysis with error handling
        try:
            simple_slopes_r = ro.r['simple_slopes_analysis'](va_long_r)
        
            # Convert back to pandas DataFrame
            simple_slopes_results = pandas2ri.rpy2py(simple_slopes_r)
        
            print("\nSimple Slopes Results:")
            print(simple_slopes_results)
        except Exception as e:
            print(f"\nError in simple slopes analysis: {str(e)}")
            # Create empty DataFrame with correct structure
            simple_slopes_results = pd.DataFrame(columns=[
                'task', 'effect', 'VVIQ2_level', 'estimate', 'std_error', 
                't_value', 'p_value', 'significance_indicator'
            ])
            print("Created empty simple slopes results DataFrame due to error")
    
    return ancova_results, simple_slopes_results

def conduct_mrt_ancova(mrt_acc_wide, mrt_rt_wide):
    """
    Conduct repeated-measures ANCOVA for Mental Rotation Task.
    
    Parameters:
    -----------
    mrt_acc_wide : pandas.DataFrame
        MRT accuracy data in wide format with VVIQ2 scores.
    mrt_rt_wide : pandas.DataFrame
        MRT RT data in wide format with VVIQ2 scores.
        
    Returns:
    --------
    tuple
        (ancova_results, simple_slopes_results)
    """
    print("\nConducting repeated-measures ANCOVA for Mental Rotation Task...")
    
    # Activate R
    pandas2ri.activate()
    
    # Import required R packages
    base = importr('base')
    stats_r = importr('stats')
    ez = importr('ez')
    emmeans = importr('emmeans')
    
    # Convert wide format to long format for R
    # For accuracy
    mrt_acc_long = pd.melt(
        mrt_acc_wide,
        id_vars=['PROLIFIC_PID', 'total_score', 'total_score_z'],
        value_vars=[col for col in mrt_acc_wide.columns if col.startswith('acc_ang')],
        var_name='condition',
        value_name='accuracy'
    )
    
    # Extract angular disparity from condition column
    mrt_acc_long['angular_disparity'] = mrt_acc_long['condition'].str.extract(r'acc_ang(\d+)').astype(int)
    
    print("\nMRT accuracy long format dataset preview:")
    print(mrt_acc_long.head(2))
    
    # For RT
    mrt_rt_long = pd.melt(
        mrt_rt_wide,
        id_vars=['PROLIFIC_PID', 'total_score', 'total_score_z'],
        value_vars=[col for col in mrt_rt_wide.columns if col.startswith('rt_ang')],
        var_name='condition',
        value_name='rt'
    )
    
    # Extract angular disparity from condition column
    mrt_rt_long['angular_disparity'] = mrt_rt_long['condition'].str.extract(r'rt_ang(\d+)').astype(int)
    
    print("\nMRT RT long format dataset preview:")
    print(mrt_rt_long.head(2))
    
    # Convert to R dataframes
    mrt_acc_long_r = pandas2ri.py2rpy(mrt_acc_long)
    mrt_rt_long_r = pandas2ri.py2rpy(mrt_rt_long)
    
    # Create factors in R
    ro.r('''
    prepare_mrt_data <- function(data) {
        data$PROLIFIC_PID <- as.factor(data$PROLIFIC_PID)
        data$angular_disparity <- as.factor(data$angular_disparity)
        return(data)
    }
    ''')
    
    mrt_acc_long_r = ro.r['prepare_mrt_data'](mrt_acc_long_r)
    mrt_rt_long_r = ro.r['prepare_mrt_data'](mrt_rt_long_r)
    
    # Run the ANCOVA in R
    ro.r('''
    run_mrt_ancova <- function(data, dv_name) {
        # Add error handling to prevent crashes
        tryCatch({
            # Run the ANCOVA using ezANOVA
            if (dv_name == "accuracy") {
                model <- ezANOVA(
                    data = data,
                    dv = .(accuracy),
                    wid = .(PROLIFIC_PID),
                    within = .(angular_disparity),
                    between = .(total_score_z),
                    type = 3,
                    detailed = TRUE
                )
            } else {
                model <- ezANOVA(
                    data = data,
                    dv = .(rt),
                    wid = .(PROLIFIC_PID),
                    within = .(angular_disparity),
                    between = .(total_score_z),
                    type = 3,
                    detailed = TRUE
                )
            }
            
            # Format the results
            result <- data.frame(
                outcome_variable = character(),
                effect = character(),
                F_value = numeric(),
                df_numerator = numeric(),
                df_denominator = numeric(),
                p_value = numeric(),
                partial_eta_squared = numeric(),
                partial_eta_squared_lower_ci = numeric(),
                partial_eta_squared_upper_ci = numeric(),
                significance_indicator = character(),
                stringsAsFactors = FALSE
            )
            
            # Extract main effects and interactions
            for (i in 1:nrow(model$ANOVA)) {
                effect_name <- rownames(model$ANOVA)[i]
                F_val <- model$ANOVA$F[i]
                df_num <- model$ANOVA$DFn[i]
                df_den <- model$ANOVA$DFd[i]
                p_val <- model$ANOVA$p[i]
                ges <- model$ANOVA$ges[i]
                
                # Calculate confidence intervals for partial eta squared
                # Using Fisher's z transformation approximation
                z <- 0.5 * log((1 + sqrt(ges)) / (1 - sqrt(ges)))
                se <- 1 / sqrt(df_num + df_den - 1)
                z_lower <- z - 1.96 * se
                z_upper <- z + 1.96 * se
                ges_lower <- (exp(2 * z_lower) - 1)^2 / (exp(2 * z_lower) + 1)^2
                ges_upper <- (exp(2 * z_upper) - 1)^2 / (exp(2 * z_upper) + 1)^2
                
                # Determine significance indicator
                sig_indicator <- ""
                if (p_val < 0.001) sig_indicator <- "***"
                else if (p_val < 0.01) sig_indicator <- "**"
                else if (p_val < 0.05) sig_indicator <- "*"
                else if (p_val < 0.1) sig_indicator <- "."
                
                result <- rbind(result, data.frame(
                    outcome_variable = as.character(dv_name),
                    effect = effect_name,
                    F_value = F_val,
                    df_numerator = df_num,
                    df_denominator = df_den,
                    p_value = p_val,
                    partial_eta_squared = ges,
                    partial_eta_squared_lower_ci = ges_lower,
                    partial_eta_squared_upper_ci = ges_upper,
                    significance_indicator = sig_indicator
                ))
            }
            
            return(result)
        }, error = function(e) {
            # Print error message
            print(paste("Error in R ANCOVA analysis:", e$message))
            
            # Return empty dataframe with correct structure
            empty_result <- data.frame(
                outcome_variable = as.character(dv_name),
                effect = "Error",
                F_value = NA,
                df_numerator = NA,
                df_denominator = NA,
                p_value = NA,
                partial_eta_squared = NA,
                partial_eta_squared_lower_ci = NA,
                partial_eta_squared_upper_ci = NA,
                significance_indicator = "",
                stringsAsFactors = FALSE
            )
            return(empty_result)
        })
    }
    ''')
    
    # Run the ANCOVA for accuracy
    acc_ancova_results_r = ro.r['run_mrt_ancova'](mrt_acc_long_r, "accuracy")
    
    # Run the ANCOVA for RT
    rt_ancova_results_r = ro.r['run_mrt_ancova'](mrt_rt_long_r, "rt")
    
    # Convert back to pandas DataFrame
    acc_ancova_results = pandas2ri.rpy2py(acc_ancova_results_r)
    rt_ancova_results = pandas2ri.rpy2py(rt_ancova_results_r)
    
    # Combine results
    ancova_results = pd.concat([acc_ancova_results, rt_ancova_results])
    
    print("\nMRT ANCOVA Results:")
    print(ancova_results)
    
    # Check for significant interactions involving VVIQ2
    vviq2_interactions = ancova_results[
        (ancova_results['effect'].str.contains('total_score_z')) & 
        (ancova_results['p_value'] < 0.05)
    ]
    
    # If there are significant interactions, conduct simple slopes analyses
    simple_slopes_results = pd.DataFrame()
    
    if not vviq2_interactions.empty:
        print("\nConducting simple slopes analyses for significant VVIQ2 interactions...")
        
        # Define R function for simple slopes analysis
        ro.r('''
        mrt_simple_slopes_analysis <- function(acc_data, rt_data) {
            # Create empty dataframe for results
            results <- data.frame(
                task = character(),
                effect = character(),
                VVIQ2_level = character(),
                estimate = numeric(),
                std_error = numeric(),
                t_value = numeric(),
                p_value = numeric(),
                significance_indicator = character(),
                stringsAsFactors = FALSE
            )
            
            # Accuracy model
            acc_model <- lm(accuracy ~ angular_disparity * total_score_z, data = acc_data)
            
            # RT model
            rt_model <- lm(rt ~ angular_disparity * total_score_z, data = rt_data)
            
            # Define levels for VVIQ2 (-1 SD, Mean, +1 SD)
            vviq_levels <- c(-1, 0, 1)
            
            # For each VVIQ2 level
            for (vviq_level in vviq_levels) {
                level_name <- ifelse(vviq_level == -1, "Low VVIQ2 (-1 SD)", 
                                   ifelse(vviq_level == 0, "Medium VVIQ2 (Mean)", 
                                          "High VVIQ2 (+1 SD)"))
                
                # Accuracy contrasts
                acc_rg <- ref_grid(acc_model, at = list(total_score_z = vviq_level))
                acc_emm <- emmeans(acc_rg, ~ angular_disparity)
                acc_cont <- contrast(acc_emm, method = "pairwise")
                
                # Extract accuracy results
                for (i in 1:nrow(acc_cont)) {
                    p_val <- summary(acc_cont)$p.value[i]
                    
                    # Determine significance indicator
                    sig_indicator <- ""
                    if (p_val < 0.001) sig_indicator <- "***"
                    else if (p_val < 0.01) sig_indicator <- "**"
                    else if (p_val < 0.05) sig_indicator <- "*"
                    else if (p_val < 0.1) sig_indicator <- "."
                    
                    results <- rbind(results, data.frame(
                        task = "MRT",
                        effect = paste0("Accuracy: ", summary(acc_cont)$contrast[i]),
                        VVIQ2_level = level_name,
                        estimate = summary(acc_cont)$estimate[i],
                        std_error = summary(acc_cont)$SE[i],
                        t_value = summary(acc_cont)$t.ratio[i],
                        p_value = p_val,
                        significance_indicator = sig_indicator
                    ))
                }
                
                # RT contrasts
                rt_rg <- ref_grid(rt_model, at = list(total_score_z = vviq_level))
                rt_emm <- emmeans(rt_rg, ~ angular_disparity)
                rt_cont <- contrast(rt_emm, method = "pairwise")
                
                # Extract RT results
                for (i in 1:nrow(rt_cont)) {
                    p_val <- summary(rt_cont)$p.value[i]
                    
                    # Determine significance indicator
                    sig_indicator <- ""
                    if (p_val < 0.001) sig_indicator <- "***"
                    else if (p_val < 0.01) sig_indicator <- "**"
                    else if (p_val < 0.05) sig_indicator <- "*"
                    else if (p_val < 0.1) sig_indicator <- "."
                    
                    results <- rbind(results, data.frame(
                        task = "MRT",
                        effect = paste0("RT: ", summary(rt_cont)$contrast[i]),
                        VVIQ2_level = level_name,
                        estimate = summary(rt_cont)$estimate[i],
                        std_error = summary(rt_cont)$SE[i],
                        t_value = summary(rt_cont)$t.ratio[i],
                        p_value = p_val,
                        significance_indicator = sig_indicator
                    ))
                }
            }
            
            return(results)
        }
        ''')
        
        # Run simple slopes analysis with error handling
        try:
            simple_slopes_r = ro.r['mrt_simple_slopes_analysis'](mrt_acc_long_r, mrt_rt_long_r)
        
            # Convert back to pandas DataFrame
            simple_slopes_results = pandas2ri.rpy2py(simple_slopes_r)
        
            print("\nMRT Simple Slopes Results:")
            print(simple_slopes_results)
        except Exception as e:
            print(f"\nError in MRT simple slopes analysis: {str(e)}")
            # Create empty DataFrame with correct structure
            simple_slopes_results = pd.DataFrame(columns=[
                'task', 'effect', 'VVIQ2_level', 'estimate', 'std_error', 
                't_value', 'p_value', 'significance_indicator'
            ])
            print("Created empty MRT simple slopes results DataFrame due to error")
    
    return ancova_results, simple_slopes_results

def explore_vviq2_subscales(data, vviq2):
    """
    Explore VVIQ2 subscale relationships with VA and MRT measures.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Merged dataset with VA and MRT measures.
    vviq2 : pandas.DataFrame
        VVIQ2 scores data with subscales.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing correlation results for subscales.
    """
    try:
        print("\nExploring VVIQ2 subscale relationships...")
        
        # Merge data with VVIQ2 subscales
        subscale_cols = [
            'familiar_person_score', 'sunrise_score', 'shop_front_score', 
            'countryside_score', 'driving_score', 'beach_score', 
            'railway_station_score', 'garden_score'
        ]
        
        merged = pd.merge(
            data, 
            vviq2[['PROLIFIC_PID'] + subscale_cols], 
            on='PROLIFIC_PID', 
            how='inner'
        )
        
        print(f"Data merged with VVIQ2 subscales: {len(merged)} participants")
        
        # Define measures to correlate with subscales
        va_measures = [
            ('set_size_effect_delay1', 'VA set size effect (delay 1s)'),
            ('set_size_effect_delay3', 'VA set size effect (delay 3s)'),
            ('avg_set_size_effect', 'VA average set size effect')
        ]
        
        mrt_measures = [
            ('rt_by_angle_slope', 'MRT RT-by-angle slope'),
            ('mrt_overall_accuracy', 'MRT overall accuracy')
        ]
        
        # Calculate correlations
        correlations = []
        
        # For each subscale
        for subscale in subscale_cols:
            subscale_name = subscale.replace('_score', '')
            
            # Correlate with VA measures
            for va_var, va_desc in va_measures:
                result = calculate_correlation_with_ci(merged, subscale, va_var)
                
                correlations.append({
                    'VA_measure': va_var,
                    'VA_condition': va_desc,
                    'VVIQ2_measure': subscale,
                    **result
                })
            
            # Correlate with MRT measures
            for mrt_var, mrt_desc in mrt_measures:
                result = calculate_correlation_with_ci(merged, subscale, mrt_var)
                
                correlations.append({
                    'MRT_measure': mrt_var,
                    'MRT_condition': mrt_desc,
                    'VVIQ2_measure': subscale,
                    **result
                })
        
        # Create a DataFrame with the results
        results_df = pd.DataFrame(correlations)
        
        # Apply FDR correction for multiple comparisons
        va_subscale_results = results_df[results_df['VA_measure'].notna()].copy()
        mrt_subscale_results = results_df[results_df['MRT_measure'].notna()].copy()
        
        # FDR correction for VA correlations
        if not va_subscale_results.empty:
            try:
                _, va_subscale_results['p_value_fdr'] = fdrcorrection(va_subscale_results['p_value'])
                
                # Update hypothesis support based on FDR-corrected p-values
                va_subscale_results['hypothesis_supported_fdr'] = (
                    (va_subscale_results['p_value_fdr'] < 0.05) & 
                    (abs(va_subscale_results['pearson_r']) >= 0.25)
                )
                print(f"Applied FDR correction to {len(va_subscale_results)} VA correlations")
            except Exception as fdr_err:
                print(f"Error in VA FDR correction: {str(fdr_err)}")
                va_subscale_results['p_value_fdr'] = va_subscale_results['p_value']
                va_subscale_results['hypothesis_supported_fdr'] = va_subscale_results['hypothesis_supported']
        
        # FDR correction for MRT correlations
        if not mrt_subscale_results.empty:
            try:
                _, mrt_subscale_results['p_value_fdr'] = fdrcorrection(mrt_subscale_results['p_value'])
                
                # Update hypothesis support based on FDR-corrected p-values
                mrt_subscale_results['hypothesis_supported_fdr'] = (
                    (mrt_subscale_results['p_value_fdr'] < 0.05) & 
                    (abs(mrt_subscale_results['pearson_r']) >= 0.25)
                )
                print(f"Applied FDR correction to {len(mrt_subscale_results)} MRT correlations")
            except Exception as fdr_err:
                print(f"Error in MRT FDR correction: {str(fdr_err)}")
                mrt_subscale_results['p_value_fdr'] = mrt_subscale_results['p_value']
                mrt_subscale_results['hypothesis_supported_fdr'] = mrt_subscale_results['hypothesis_supported']
        
        # Combine results
        fdr_results = pd.concat([va_subscale_results, mrt_subscale_results])
        
        print("\nVVIQ2 Subscale Correlation Results (with FDR correction):")
        print(fdr_results[['VVIQ2_measure', 'VA_measure', 'MRT_measure', 'pearson_r', 
                           'p_value', 'p_value_fdr', 'hypothesis_supported', 
                           'hypothesis_supported_fdr']].head(10))
        
        return fdr_results
        
    except Exception as e:
        print(f"Error in VVIQ2 subscale analysis: {str(e)}")
        # Return empty dataframe with correct structure
        empty_df = pd.DataFrame(columns=[
            'VVIQ2_measure', 'VA_measure', 'VA_condition', 'MRT_measure', 'MRT_condition',
            'pearson_r', 'p_value', 'r_lower_ci', 'r_upper_ci', 'hypothesis_supported', 'n',
            'p_value_fdr', 'hypothesis_supported_fdr'
        ])
        print("Created empty subscale results DataFrame due to error")
        return empty_df

def save_results(va_correlations, mrt_correlations, va_ancova, mrt_ancova, simple_slopes):
    """
    Save results to output CSV files with timestamp.
    
    Parameters:
    -----------
    va_correlations : pandas.DataFrame
        Results of VVIQ2 and VA correlations.
    mrt_correlations : pandas.DataFrame
        Results of VVIQ2 and MRT correlations.
    va_ancova : pandas.DataFrame
        Results of VA ANCOVA.
    mrt_ancova : pandas.DataFrame
        Results of MRT ANCOVA.
    simple_slopes : pandas.DataFrame
        Results of simple slopes analyses.
    """
    print("\nSaving results to output files...")
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Ensure all DataFrames have at least one row to avoid empty file issues
    if va_correlations.empty:
        va_correlations = pd.DataFrame({
            'VA_measure': ['placeholder'], 'VA_condition': ['placeholder'], 'VVIQ2_measure': ['placeholder'],
            'pearson_r': [0], 'p_value': [1], 'r_lower_ci': [0], 'r_upper_ci': [0], 
            'hypothesis_supported': [False], 'n': [0]
        })
        print("Created placeholder row for empty VA correlations DataFrame")
    
    if mrt_correlations.empty:
        mrt_correlations = pd.DataFrame({
            'MRT_measure': ['placeholder'], 'MRT_condition': ['placeholder'], 'VVIQ2_measure': ['placeholder'],
            'pearson_r': [0], 'p_value': [1], 'r_lower_ci': [0], 'r_upper_ci': [0], 
            'hypothesis_supported': [False], 'n': [0]
        })
        print("Created placeholder row for empty MRT correlations DataFrame")
    
    if va_ancova.empty:
        va_ancova = pd.DataFrame({
            'effect': ['placeholder'], 'F_value': [0], 'df_numerator': [0], 'df_denominator': [0],
            'p_value': [1], 'partial_eta_squared': [0], 'partial_eta_squared_lower_ci': [0],
            'partial_eta_squared_upper_ci': [0], 'significance_indicator': ['']
        })
        print("Created placeholder row for empty VA ANCOVA DataFrame")
    
    if mrt_ancova.empty:
        mrt_ancova = pd.DataFrame({
            'outcome_variable': ['placeholder'], 'effect': ['placeholder'], 'F_value': [0], 
            'df_numerator': [0], 'df_denominator': [0], 'p_value': [1], 'partial_eta_squared': [0],
            'partial_eta_squared_lower_ci': [0], 'partial_eta_squared_upper_ci': [0], 
            'significance_indicator': ['']
        })
        print("Created placeholder row for empty MRT ANCOVA DataFrame")
    
    if simple_slopes.empty:
        simple_slopes = pd.DataFrame({
            'task': ['placeholder'], 'effect': ['placeholder'], 'VVIQ2_level': ['placeholder'],
            'estimate': [0], 'std_error': [0], 't_value': [0], 'p_value': [1], 
            'significance_indicator': ['']
        })
        print("Created placeholder row for empty simple slopes DataFrame")
    
    try:
        # Save VA correlations
        va_correlations.to_csv(f'outputs/VVIQ2_VA_correlations_{timestamp}.csv', index=False)
        print(f"Saved VA correlations to outputs/VVIQ2_VA_correlations_{timestamp}.csv")
        print(f"Verified: File exists at outputs/VVIQ2_VA_correlations_{timestamp}.csv: {os.path.exists(f'outputs/VVIQ2_VA_correlations_{timestamp}.csv')}")
        
        # Save MRT correlations
        mrt_correlations.to_csv(f'outputs/VVIQ2_MRT_correlations_{timestamp}.csv', index=False)
        print(f"Saved MRT correlations to outputs/VVIQ2_MRT_correlations_{timestamp}.csv")
        print(f"Verified: File exists at outputs/VVIQ2_MRT_correlations_{timestamp}.csv: {os.path.exists(f'outputs/VVIQ2_MRT_correlations_{timestamp}.csv')}")
        
        # Save VA ANCOVA results
        va_ancova.to_csv(f'outputs/VA_ANCOVA_VVIQ2_results_{timestamp}.csv', index=False)
        print(f"Saved VA ANCOVA results to outputs/VA_ANCOVA_VVIQ2_results_{timestamp}.csv")
        print(f"Verified: File exists at outputs/VA_ANCOVA_VVIQ2_results_{timestamp}.csv: {os.path.exists(f'outputs/VA_ANCOVA_VVIQ2_results_{timestamp}.csv')}")
        
        # Save MRT ANCOVA results
        mrt_ancova.to_csv(f'outputs/MRT_ANCOVA_VVIQ2_results_{timestamp}.csv', index=False)
        print(f"Saved MRT ANCOVA results to outputs/MRT_ANCOVA_VVIQ2_results_{timestamp}.csv")
        print(f"Verified: File exists at outputs/MRT_ANCOVA_VVIQ2_results_{timestamp}.csv: {os.path.exists(f'outputs/MRT_ANCOVA_VVIQ2_results_{timestamp}.csv')}")
        
        # Save simple slopes results
        simple_slopes.to_csv(f'outputs/VVIQ2_simple_slopes_{timestamp}.csv', index=False)
        print(f"Saved simple slopes results to outputs/VVIQ2_simple_slopes_{timestamp}.csv")
        print(f"Verified: File exists at outputs/VVIQ2_simple_slopes_{timestamp}.csv: {os.path.exists(f'outputs/VVIQ2_simple_slopes_{timestamp}.csv')}")
    except Exception as e:
        print(f"Error during file saving: {str(e)}")
        raise

def main():
    """
    Main function to execute the analysis workflow.
    """
    try:
        # Create output directory if it doesn't exist
        create_output_directory()
        
        # Load data
        print("Loading data files...")
        va_perf = load_va_performance_metrics()
        mrt_perf = load_mrt_performance_metrics()
        mrt_reg = load_mrt_regression_metrics()
        va_effects = load_va_condition_effects()
        vviq2 = load_vviq2_scores()
        
        # Merge and prepare data
        merged_data, va_wide, mrt_acc_wide, mrt_rt_wide = merge_and_prepare_data(
            va_perf, mrt_perf, mrt_reg, va_effects, vviq2
        )
        
        # Test Hypothesis 3a - VVIQ2 and set size effects
        va_correlations = test_hypothesis_3a(merged_data)
        
        # Test Hypothesis 3b - VVIQ2 and Mental Rotation
        mrt_correlations = test_hypothesis_3b(merged_data)
        
        # Create empty DataFrames for results that might fail
        va_ancova_results = pd.DataFrame(columns=['effect', 'F_value', 'df_numerator', 'df_denominator', 
                                                'p_value', 'partial_eta_squared', 'partial_eta_squared_lower_ci', 
                                                'partial_eta_squared_upper_ci', 'significance_indicator'])
        va_simple_slopes = pd.DataFrame(columns=['task', 'effect', 'VVIQ2_level', 'estimate', 'std_error', 
                                                't_value', 'p_value', 'significance_indicator'])
        
        # Conduct repeated-measures ANCOVA for Visual Arrays Task
        try:
            va_ancova_results, va_simple_slopes = conduct_va_ancova(va_wide)
            print("VA ANCOVA completed successfully")
        except Exception as e:
            print(f"Error in VA ANCOVA: {str(e)}")
        
        # Create empty DataFrames for MRT results that might fail
        mrt_ancova_results = pd.DataFrame(columns=['outcome_variable', 'effect', 'F_value', 'df_numerator', 
                                                'df_denominator', 'p_value', 'partial_eta_squared', 
                                                'partial_eta_squared_lower_ci', 'partial_eta_squared_upper_ci', 
                                                'significance_indicator'])
        mrt_simple_slopes = pd.DataFrame(columns=['task', 'effect', 'VVIQ2_level', 'estimate', 'std_error', 
                                                't_value', 'p_value', 'significance_indicator'])
        
        # Conduct repeated-measures ANCOVA for Mental Rotation Task
        try:
            mrt_ancova_results, mrt_simple_slopes = conduct_mrt_ancova(mrt_acc_wide, mrt_rt_wide)
            print("MRT ANCOVA completed successfully")
        except Exception as e:
            print(f"Error in MRT ANCOVA: {str(e)}")
        
        # Combine simple slopes results
        all_simple_slopes = pd.concat([va_simple_slopes, mrt_simple_slopes])
        
        # If no simple slopes were calculated, create an empty DataFrame with the required columns
        if all_simple_slopes.empty:
            all_simple_slopes = pd.DataFrame(columns=[
                'task', 'effect', 'VVIQ2_level', 'estimate', 'std_error', 
                't_value', 'p_value', 'significance_indicator'
            ])
        
        # Create empty subscale results DataFrame with correct structure
        empty_subscale_df = pd.DataFrame(columns=[
            'VVIQ2_measure', 'VA_measure', 'VA_condition', 'MRT_measure', 'MRT_condition',
            'pearson_r', 'p_value', 'r_lower_ci', 'r_upper_ci', 'hypothesis_supported', 'n',
            'p_value_fdr', 'hypothesis_supported_fdr'
        ])
        
        # Explore VVIQ2 subscale relationships
        try:
            subscale_results = explore_vviq2_subscales(merged_data, vviq2)
            if subscale_results.empty:
                subscale_results = empty_subscale_df
            print("VVIQ2 subscale analysis completed successfully")
        except Exception as e:
            print(f"Error in VVIQ2 subscale analysis: {str(e)}")
            subscale_results = empty_subscale_df
            print("Using empty subscale results DataFrame due to error")
        
        # Save results to output files
        try:
            save_results(
                va_correlations, 
                mrt_correlations, 
                va_ancova_results, 
                mrt_ancova_results, 
                all_simple_slopes
            )
            print("Results saved successfully")
            
            # Save subscale results separately to avoid issues with the main save_results function
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            subscale_file = f'outputs/VVIQ2_subscale_correlations_{timestamp}.csv'
            subscale_results.to_csv(subscale_file, index=False)
            print(f"Saved VVIQ2 subscale correlations to {subscale_file}")
            print(f"Verified: File exists at {subscale_file}: {os.path.exists(subscale_file)}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            print("Attempting to create backup empty files")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Create backup empty files with correct structure
            for filename, df_name in [
                (f'outputs/VVIQ2_VA_correlations_{timestamp}.csv', 'VA correlations'),
                (f'outputs/VVIQ2_MRT_correlations_{timestamp}.csv', 'MRT correlations'),
                (f'outputs/VA_ANCOVA_VVIQ2_results_{timestamp}.csv', 'VA ANCOVA results'),
                (f'outputs/MRT_ANCOVA_VVIQ2_results_{timestamp}.csv', 'MRT ANCOVA results'),
                (f'outputs/VVIQ2_simple_slopes_{timestamp}.csv', 'Simple slopes results'),
                (f'outputs/VVIQ2_subscale_correlations_{timestamp}.csv', 'VVIQ2 subscale correlations')
            ]:
                try:
                    pd.DataFrame().to_csv(filename, index=False)
                    print(f"Created backup empty file for {df_name}: {filename}")
                    print(f"Verified: File exists at {filename}: {os.path.exists(filename)}")
                except Exception as backup_err:
                    print(f"Failed to create backup file for {df_name}: {str(backup_err)}")
        
        print("Finished execution")
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
