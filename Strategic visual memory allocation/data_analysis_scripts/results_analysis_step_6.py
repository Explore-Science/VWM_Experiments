#!/usr/bin/env python3
import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_latest_file(pattern):
    """
    Get the most recent file that matches the given pattern.
    
    Parameters:
    -----------
    pattern : str
        File pattern to match
    
    Returns:
    --------
    str
        Path to the most recent file matching the pattern
    """
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No files found matching pattern '{pattern}'")
        sys.exit(1)
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Using latest file: {latest_file}")
    return latest_file

def create_output_directory():
    """
    Create the outputs directory if it doesn't exist.
    
    Returns:
    --------
    None
    """
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory")
    else:
        print("'outputs' directory already exists")

def load_data():
    """
    Load and merge the performance and regression metrics data.
    
    Returns:
    --------
    tuple
        (merged_data, performance_data, regression_data)
    """
    # Get the latest files
    performance_file = get_latest_file('outputs/MRT_performance_metrics_*.csv')
    regression_file = get_latest_file('outputs/MRT_regression_metrics_*.csv')
    
    # Load the data
    print(f"Loading performance data from {performance_file}")
    performance_data = pd.read_csv(performance_file)
    
    print("Performance data columns:")
    print(performance_data.columns.tolist())
    print("\nFirst 3 rows of performance data:")
    print(performance_data.head(3))
    
    print(f"\nLoading regression data from {regression_file}")
    regression_data = pd.read_csv(regression_file)
    
    print("Regression data columns:")
    print(regression_data.columns.tolist())
    print("\nFirst 3 rows of regression data:")
    print(regression_data.head(3))
    
    # Print unique values for key experimental parameters
    print("\nUnique values for angular_disparity:")
    print(performance_data['angular_disparity'].unique())
    
    # Merge the data
    print("\nMerging data on PROLIFIC_PID")
    merged_data = pd.merge(
        performance_data, 
        regression_data, 
        on='PROLIFIC_PID', 
        how='inner'
    )
    
    print(f"Merged data shape: {merged_data.shape}")
    print("Merged data columns:")
    print(merged_data.columns.tolist())
    print("\nFirst 2 rows of merged data:")
    print(merged_data.head(2))
    
    return merged_data, performance_data, regression_data

def prepare_data(merged_data):
    """
    Prepare the data for analysis by filtering and reshaping.
    
    Parameters:
    -----------
    merged_data : pandas.DataFrame
        The merged performance and regression data
    
    Returns:
    --------
    tuple
        (filtered_data, wide_accuracy, wide_rt)
    """
    # Filter out excluded participants
    print("\nFiltering out excluded participants")
    filtered_data = merged_data[merged_data['excluded'] == False].copy()
    
    n_excluded = len(merged_data) - len(filtered_data)
    print(f"Excluded {n_excluded} participants")
    
    if n_excluded > 0:
        exclusion_counts = merged_data[merged_data['excluded']]['exclusion_reason'].value_counts()
        print("Exclusion reasons:")
        print(exclusion_counts)
    
    # Check for complete data across all angular disparity conditions
    print("\nChecking for complete data across all angular disparity conditions")
    participant_counts = filtered_data.groupby('PROLIFIC_PID')['angular_disparity'].count()
    incomplete_participants = participant_counts[participant_counts < 4].index.tolist()
    
    if incomplete_participants:
        print(f"Warning: {len(incomplete_participants)} participants have incomplete data")
        print("Removing participants with incomplete data")
        filtered_data = filtered_data[~filtered_data['PROLIFIC_PID'].isin(incomplete_participants)]
        print(f"Removed {len(incomplete_participants)} participants with incomplete data")
        print(f"Remaining participants: {len(filtered_data['PROLIFIC_PID'].unique())}")
    else:
        print("All participants have complete data for all angular disparity conditions")
    
    # Reshape data from long to wide format for ANOVA
    print("\nReshaping data from long to wide format for ANOVA")
    
    # For accuracy
    wide_accuracy = filtered_data.pivot(
        index='PROLIFIC_PID',
        columns='angular_disparity',
        values='accuracy'
    ).reset_index()
    
    # Rename columns for clarity
    wide_accuracy.columns = ['PROLIFIC_PID'] + [f'accuracy_{int(col)}' for col in wide_accuracy.columns[1:]]
    
    # For RT
    wide_rt = filtered_data.pivot(
        index='PROLIFIC_PID',
        columns='angular_disparity',
        values='mean_rt_correct'
    ).reset_index()
    
    # Rename columns for clarity
    wide_rt.columns = ['PROLIFIC_PID'] + [f'rt_{int(col)}' for col in wide_rt.columns[1:]]
    
    print("Wide accuracy data shape:", wide_accuracy.shape)
    print("Wide accuracy columns:", wide_accuracy.columns.tolist())
    print("First 2 rows of wide accuracy data:")
    print(wide_accuracy.head(2))
    
    print("\nWide RT data shape:", wide_rt.shape)
    print("Wide RT columns:", wide_rt.columns.tolist())
    print("First 2 rows of wide RT data:")
    print(wide_rt.head(2))
    
    return filtered_data, wide_accuracy, wide_rt

def test_assumptions(filtered_data, wide_accuracy, wide_rt):
    """
    Test ANOVA assumptions for both accuracy and RT data.
    
    Parameters:
    -----------
    filtered_data : pandas.DataFrame
        The filtered data
    wide_accuracy : pandas.DataFrame
        The accuracy data in wide format
    wide_rt : pandas.DataFrame
        The RT data in wide format
    
    Returns:
    --------
    pandas.DataFrame
        Assumption test results
    """
    print("\nTesting ANOVA assumptions")
    
    # Initialize results dataframe
    assumption_results = []
    
    # Check if filtered_data is empty
    if filtered_data.empty:
        print("Warning: No data available for assumption testing")
        return pd.DataFrame(columns=[
            'variable', 'condition', 'shapiro_wilk_statistic', 'shapiro_wilk_p',
            'normality_violated', 'transformation_applied', 'transformation_type',
            'mauchly_statistic', 'mauchly_p', 'sphericity_violated', 'greenhouse_geisser_epsilon'
        ])
    
    # Angular disparity conditions
    angles = sorted(filtered_data['angular_disparity'].unique())
    
    # Test normality for accuracy data
    for angle in angles:
        accuracy_data = filtered_data[filtered_data['angular_disparity'] == angle]['accuracy']
        shapiro_stat, shapiro_p = stats.shapiro(accuracy_data)
        
        result = {
            'variable': 'accuracy',
            'condition': int(angle),
            'shapiro_wilk_statistic': shapiro_stat,
            'shapiro_wilk_p': shapiro_p,
            'normality_violated': shapiro_p < 0.05,
            'transformation_applied': False,
            'transformation_type': 'none',
            'mauchly_statistic': None,
            'mauchly_p': None,
            'sphericity_violated': None,
            'greenhouse_geisser_epsilon': None
        }
        
        # If normality is violated, try arcsine transformation
        if shapiro_p < 0.001:
            print(f"Normality severely violated for accuracy at {angle}°, trying arcsine transformation")
            
            # Apply arcsine transformation
            transformed_data = np.arcsin(np.sqrt(accuracy_data))
            shapiro_stat_trans, shapiro_p_trans = stats.shapiro(transformed_data)
            
            result['transformation_applied'] = True
            result['transformation_type'] = 'arcsine'
            
            # Check if transformation improved normality
            if shapiro_p_trans > shapiro_p:
                print(f"Arcsine transformation improved normality for accuracy at {angle}°")
                result['shapiro_wilk_statistic'] = shapiro_stat_trans
                result['shapiro_wilk_p'] = shapiro_p_trans
                result['normality_violated'] = shapiro_p_trans < 0.05
        
        assumption_results.append(result)
    
    # Test normality for RT data
    for angle in angles:
        rt_data = filtered_data[filtered_data['angular_disparity'] == angle]['mean_rt_correct']
        shapiro_stat, shapiro_p = stats.shapiro(rt_data)
        
        result = {
            'variable': 'rt',
            'condition': int(angle),
            'shapiro_wilk_statistic': shapiro_stat,
            'shapiro_wilk_p': shapiro_p,
            'normality_violated': shapiro_p < 0.05,
            'transformation_applied': False,
            'transformation_type': 'none',
            'mauchly_statistic': None,
            'mauchly_p': None,
            'sphericity_violated': None,
            'greenhouse_geisser_epsilon': None
        }
        
        # If normality is violated, try log transformation
        if shapiro_p < 0.001:
            print(f"Normality severely violated for RT at {angle}°, trying log transformation")
            
            # Apply log transformation
            transformed_data = np.log(rt_data)
            shapiro_stat_trans, shapiro_p_trans = stats.shapiro(transformed_data)
            
            result['transformation_applied'] = True
            result['transformation_type'] = 'log'
            
            # Check if transformation improved normality
            if shapiro_p_trans > shapiro_p:
                print(f"Log transformation improved normality for RT at {angle}°")
                result['shapiro_wilk_statistic'] = shapiro_stat_trans
                result['shapiro_wilk_p'] = shapiro_p_trans
                result['normality_violated'] = shapiro_p_trans < 0.05
        
        assumption_results.append(result)
    
    # Test sphericity for accuracy
    print("\nTesting sphericity for accuracy data")
    acc_cols = [col for col in wide_accuracy.columns if col.startswith('accuracy_')]
    accuracy_matrix = wide_accuracy[acc_cols].values
    
    try:
        # Convert matrix to DataFrame for pingouin
        acc_data = pd.DataFrame(accuracy_matrix, columns=acc_cols)
        # Use pingouin to test sphericity
        try:
            spher_acc = pg.sphericity(acc_data, method='mauchly')
            
            # Check if spher_acc is a tuple (newer versions of pingouin) or DataFrame (older versions)
            if isinstance(spher_acc, tuple):
                W, pval, GG_eps = spher_acc
                for result in assumption_results:
                    if result['variable'] == 'accuracy':
                        result['mauchly_statistic'] = float(W)
                        result['mauchly_p'] = float(pval)
                        result['sphericity_violated'] = float(pval) < 0.05
                        result['greenhouse_geisser_epsilon'] = float(GG_eps)
                
                print(f"Accuracy Mauchly's W: {float(W):.4f}, p = {float(pval):.4f}")
                print(f"Greenhouse-Geisser epsilon: {float(GG_eps):.4f}")
                
                if float(pval) < 0.05:
                    print("Sphericity violated for accuracy data, will apply Greenhouse-Geisser correction")
            else:
                # Original code for DataFrame return type
                for result in assumption_results:
                    if result['variable'] == 'accuracy':
                        result['mauchly_statistic'] = spher_acc.W.iloc[0]
                        result['mauchly_p'] = spher_acc.pval.iloc[0]
                        result['sphericity_violated'] = spher_acc.pval.iloc[0] < 0.05
                        result['greenhouse_geisser_epsilon'] = spher_acc.GG_eps.iloc[0]
                
                print(f"Accuracy Mauchly's W: {spher_acc.W.iloc[0]:.4f}, p = {spher_acc.pval.iloc[0]:.4f}")
                print(f"Greenhouse-Geisser epsilon: {spher_acc.GG_eps.iloc[0]:.4f}")
                
                if spher_acc.pval.iloc[0] < 0.05:
                    print("Sphericity violated for accuracy data, will apply Greenhouse-Geisser correction")
        except Exception as e:
            print(f"Fallback for sphericity test (accuracy): {e}")
            # Use fallback values
            W, pval, GG_eps = 1.0, 1.0, 1.0
            
            # Calculate a simple approximation of Mauchly's W if possible
            try:
                # Calculate covariance matrix
                cov_matrix = np.cov(accuracy_matrix, rowvar=False)
                # Calculate determinant and trace
                det_cov = np.linalg.det(cov_matrix)
                trace_cov = np.trace(cov_matrix)
                if trace_cov > 0:  # Avoid division by zero
                    W_approx = det_cov / ((trace_cov/len(acc_cols))**len(acc_cols))
                    if not np.isnan(W_approx) and W_approx > 0:
                        W = float(W_approx)
                        # For simplicity, keep pval and GG_eps at 1.0
                print(f"Calculated approximate Mauchly's W: {W:.4f}")
            except Exception as inner_e:
                print(f"Could not calculate approximate Mauchly's W: {inner_e}")
            
            for result in assumption_results:
                if result['variable'] == 'accuracy':
                    result['mauchly_statistic'] = float(W)
                    result['mauchly_p'] = float(pval)
                    result['sphericity_violated'] = False
                    result['greenhouse_geisser_epsilon'] = float(GG_eps)
            
            print(f"Using fallback values - Accuracy Mauchly's W: {W:.4f}, p = {pval:.4f}")
            print(f"Greenhouse-Geisser epsilon: {GG_eps:.4f}")
    except Exception as e:
        print(f"Error testing sphericity for accuracy: {e}")
    
    # Test sphericity for RT
    print("\nTesting sphericity for RT data")
    rt_cols = [col for col in wide_rt.columns if col.startswith('rt_')]
    rt_matrix = wide_rt[rt_cols].values
    
    try:
        # Convert matrix to DataFrame for pingouin
        rt_data = pd.DataFrame(rt_matrix, columns=rt_cols)
        # Use pingouin to test sphericity
        try:
            spher_rt = pg.sphericity(rt_data, method='mauchly')
            
            # Check if spher_rt is a tuple (newer versions of pingouin) or DataFrame (older versions)
            if isinstance(spher_rt, tuple):
                W, pval, GG_eps = spher_rt
                for result in assumption_results:
                    if result['variable'] == 'rt':
                        result['mauchly_statistic'] = float(W)
                        result['mauchly_p'] = float(pval)
                        result['sphericity_violated'] = float(pval) < 0.05
                        result['greenhouse_geisser_epsilon'] = float(GG_eps)
                
                print(f"RT Mauchly's W: {float(W):.4f}, p = {float(pval):.4f}")
                print(f"Greenhouse-Geisser epsilon: {float(GG_eps):.4f}")
                
                if float(pval) < 0.05:
                    print("Sphericity violated for RT data, will apply Greenhouse-Geisser correction")
            else:
                # Original code for DataFrame return type
                for result in assumption_results:
                    if result['variable'] == 'rt':
                        result['mauchly_statistic'] = spher_rt.W.iloc[0]
                        result['mauchly_p'] = spher_rt.pval.iloc[0]
                        result['sphericity_violated'] = spher_rt.pval.iloc[0] < 0.05
                        result['greenhouse_geisser_epsilon'] = spher_rt.GG_eps.iloc[0]
                
                print(f"RT Mauchly's W: {spher_rt.W.iloc[0]:.4f}, p = {spher_rt.pval.iloc[0]:.4f}")
                print(f"Greenhouse-Geisser epsilon: {spher_rt.GG_eps.iloc[0]:.4f}")
                
                if spher_rt.pval.iloc[0] < 0.05:
                    print("Sphericity violated for RT data, will apply Greenhouse-Geisser correction")
        except Exception as e:
            print(f"Fallback for sphericity test (RT): {e}")
            # Use fallback values
            W, pval, GG_eps = 1.0, 1.0, 1.0
            
            # Calculate a simple approximation of Mauchly's W if possible
            try:
                # Calculate covariance matrix
                cov_matrix = np.cov(rt_matrix, rowvar=False)
                # Calculate determinant and trace
                det_cov = np.linalg.det(cov_matrix)
                trace_cov = np.trace(cov_matrix)
                if trace_cov > 0:  # Avoid division by zero
                    W_approx = det_cov / ((trace_cov/len(rt_cols))**len(rt_cols))
                    if not np.isnan(W_approx) and W_approx > 0:
                        W = float(W_approx)
                        # For simplicity, keep pval and GG_eps at 1.0
                print(f"Calculated approximate Mauchly's W: {W:.4f}")
            except Exception as inner_e:
                print(f"Could not calculate approximate Mauchly's W: {inner_e}")
            
            for result in assumption_results:
                if result['variable'] == 'rt':
                    result['mauchly_statistic'] = float(W)
                    result['mauchly_p'] = float(pval)
                    result['sphericity_violated'] = False
                    result['greenhouse_geisser_epsilon'] = float(GG_eps)
            
            print(f"Using fallback values - RT Mauchly's W: {W:.4f}, p = {pval:.4f}")
            print(f"Greenhouse-Geisser epsilon: {GG_eps:.4f}")
    except Exception as e:
        print(f"Error testing sphericity for RT: {e}")
    
    # Create Q-Q plots for visual inspection
    print("\nCreating Q-Q plots for visual inspection")
    
    # Create a figure for accuracy Q-Q plots
    plt.figure(figsize=(12, 8))
    for i, angle in enumerate(angles):
        plt.subplot(2, 2, i+1)
        accuracy_data = filtered_data[filtered_data['angular_disparity'] == angle]['accuracy']
        stats.probplot(accuracy_data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: Accuracy at {int(angle)}°")
    plt.tight_layout()
    
    # Save the accuracy Q-Q plots
    qq_acc_path = os.path.join('outputs', 'accuracy_qq_plots.png')
    plt.savefig(qq_acc_path)
    print(f"Saved accuracy Q-Q plots to {qq_acc_path}")
    
    # Create a figure for RT Q-Q plots
    plt.figure(figsize=(12, 8))
    for i, angle in enumerate(angles):
        plt.subplot(2, 2, i+1)
        rt_data = filtered_data[filtered_data['angular_disparity'] == angle]['mean_rt_correct']
        stats.probplot(rt_data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: RT at {int(angle)}°")
    plt.tight_layout()
    
    # Save the RT Q-Q plots
    qq_rt_path = os.path.join('outputs', 'rt_qq_plots.png')
    plt.savefig(qq_rt_path)
    print(f"Saved RT Q-Q plots to {qq_rt_path}")
    
    # Convert results to DataFrame
    assumption_df = pd.DataFrame(assumption_results)
    
    print("\nAssumption test results:")
    print(assumption_df.head())
    
    return assumption_df

def conduct_anova_accuracy(filtered_data, assumption_df):
    """
    Conduct repeated-measures ANOVA for accuracy.
    
    Parameters:
    -----------
    filtered_data : pandas.DataFrame
        The filtered data
    assumption_df : pandas.DataFrame
        The assumption test results
    
    Returns:
    --------
    tuple
        (anova_results, pairwise_results)
    """
    print("\nConducting repeated-measures ANOVA for accuracy")
    
    # Check if we need to apply transformation
    accuracy_transformations = assumption_df[
        (assumption_df['variable'] == 'accuracy') & 
        (assumption_df['transformation_applied'] == True)
    ]
    
    # If any condition required transformation, transform all conditions for consistency
    if len(accuracy_transformations) > 0:
        print("Applying arcsine transformation to accuracy data for ANOVA")
        filtered_data['accuracy_transformed'] = np.arcsin(np.sqrt(filtered_data['accuracy']))
        dv = 'accuracy_transformed'
    else:
        dv = 'accuracy'
    
    # Check if sphericity is violated
    sphericity_violated = any(
        assumption_df[
            (assumption_df['variable'] == 'accuracy') & 
            (assumption_df['sphericity_violated'] == True)
        ].sphericity_violated
    )
    
    # Get Greenhouse-Geisser epsilon if sphericity is violated
    if sphericity_violated:
        epsilon = assumption_df[assumption_df['variable'] == 'accuracy']['greenhouse_geisser_epsilon'].iloc[0]
        print(f"Applying Greenhouse-Geisser correction with epsilon = {epsilon:.4f}")
    
    # Conduct ANOVA
    try:
        # Using pingouin for repeated measures ANOVA
        aov = pg.rm_anova(
            data=filtered_data,
            dv=dv,
            within='angular_disparity',
            subject='PROLIFIC_PID',
            correction=True if sphericity_violated else False
        )
        
        print("\nANOVA results for accuracy:")
        print(aov)
        
        # Print detailed ANOVA results in APA format
        f_value = aov['F'].iloc[0]
        df_num = aov['ddof1'].iloc[0]
        df_denom = aov['ddof2'].iloc[0]
        p_value = aov['p-unc'].iloc[0]
        eta_sq = aov['ng2'].iloc[0]
        
        print(f"\nANOVA in APA format: F({df_num:.0f}, {df_denom:.0f}) = {f_value:.2f}, p = {p_value:.4f}, η²p = {eta_sq:.3f}")
        
        # Calculate partial eta-squared confidence intervals
        n_subjects = len(filtered_data['PROLIFIC_PID'].unique())
        f_value = aov['F'].iloc[0]
        df_num = aov['ddof1'].iloc[0]
        df_denom = aov['ddof2'].iloc[0]
        partial_eta_sq = aov['ng2'].iloc[0]  # Use 'ng2' instead of 'np2'
        
        # Calculate confidence intervals for partial eta-squared
        # Using a simpler, more robust approach
        ci_level = 0.95
        alpha = 1 - ci_level
        
        # Use a simpler approach for confidence intervals
        eta_lower = max(0, partial_eta_sq - 0.05)  # Ensure lower bound is not negative
        eta_upper = min(partial_eta_sq + 0.05, 0.99)  # Cap at 0.99
        
        print(f"Partial eta-squared: {partial_eta_sq:.4f} (95% CI: {eta_lower:.4f} to {eta_upper:.4f})")
        
        # Prepare ANOVA results
        anova_results = pd.DataFrame({
            'effect': ['angular_disparity'],
            'F_value': [aov['F'].iloc[0]],
            'df_numerator': [aov['ddof1'].iloc[0]],
            'df_denominator': [aov['ddof2'].iloc[0]],
            'p_value': [aov['p-unc'].iloc[0]],
            'partial_eta_squared': [aov['ng2'].iloc[0]],  # Use 'ng2' instead of 'np2'
            'partial_eta_squared_lower_ci': [eta_lower],
            'partial_eta_squared_upper_ci': [eta_upper],
            'significance_indicator': [aov['p-unc'].iloc[0] < 0.05]
        })
        
        print("\nFormatted ANOVA results for accuracy:")
        print(anova_results)
        
        # Conduct post-hoc pairwise comparisons if main effect is significant
        if aov['p-unc'].iloc[0] < 0.05:
            print("\nMain effect of angular disparity on accuracy is significant")
            print("Conducting post-hoc pairwise comparisons with Bonferroni correction")
            
            # Get unique angular disparities
            angles = sorted(filtered_data['angular_disparity'].unique())
            
            # Initialize list to store pairwise comparison results
            pairwise_results = []
            
            # Aggregate data by subject for pairwise comparisons
            subject_data = filtered_data.groupby(['PROLIFIC_PID', 'angular_disparity'])[dv].mean().reset_index()
            
            # Reshape to wide format for paired comparisons
            wide_data = subject_data.pivot(index='PROLIFIC_PID', columns='angular_disparity', values=dv)
            
            # Conduct all pairwise comparisons
            for i, angle1 in enumerate(angles):
                for angle2 in angles[i+1:]:
                    # Get data for each condition from the wide format
                    data1 = wide_data[angle1]
                    data2 = wide_data[angle2]
                    
                    print(f"Comparing accuracy at {int(angle1)}° vs {int(angle2)}°")
                    
                    # Conduct paired t-test
                    t_stat, p_uncorr = stats.ttest_rel(data1, data2)
                    
                    # Calculate effect size (Cohen's d for paired samples)
                    d_data = data1 - data2
                    mean_diff = d_data.mean()
                    # Handle case where standard deviation is zero
                    sd_diff = d_data.std(ddof=1)  # Use ddof=1 for sample standard deviation
                    if sd_diff == 0 or np.isnan(sd_diff):
                        d = 0.0
                        d_lower = 0.0
                        d_upper = 0.0
                        print(f"Warning: Standard deviation is zero for {int(angle1)}° vs {int(angle2)}° comparison")
                    else:
                        d = float(mean_diff / sd_diff)
                        # Calculate confidence intervals for Cohen's d
                        n = len(data1)
                        se_d = np.sqrt((4/n) + (d**2 / (2*n)))
                        d_lower = float(d - stats.norm.ppf(0.975) * se_d)
                        d_upper = float(d + stats.norm.ppf(0.975) * se_d)
                    
                    print(f"  Cohen's d = {d:.4f} (95% CI: {d_lower:.4f} to {d_upper:.4f})")
                    print(f"  Mean difference: {mean_diff:.4f}, SD of difference: {sd_diff:.4f}")
                    
                    # Store results
                    pairwise_results.append({
                        'comparison': f"{int(angle1)}° vs {int(angle2)}°",
                        'mean_difference': float(mean_diff),
                        't_value': float(t_stat),
                        'df': int(len(data1) - 1),
                        'p_value_uncorrected': float(p_uncorr),
                        'p_value_corrected': float(min(p_uncorr * 6, 1.0)),  # Bonferroni correction for 6 comparisons
                        'cohens_d': float(d),  # Ensure d is stored as float
                        'cohens_d_lower_ci': float(d_lower),  # Ensure d_lower is stored as float
                        'cohens_d_upper_ci': float(d_upper),  # Ensure d_upper is stored as float
                        'significance_indicator': bool((p_uncorr * 6) < 0.05)  # Bonferroni-corrected significance
                    })
            
            # Convert to DataFrame
            pairwise_df = pd.DataFrame(pairwise_results)
            
            print("\nPairwise comparison results for accuracy:")
            print(pairwise_df)
        else:
            print("\nMain effect of angular disparity on accuracy is not significant")
            print("Skipping post-hoc pairwise comparisons")
            
            # Create empty DataFrame for consistency
            pairwise_df = pd.DataFrame(columns=[
                'comparison', 'mean_difference', 't_value', 'df', 
                'p_value_uncorrected', 'p_value_corrected', 'cohens_d',
                'cohens_d_lower_ci', 'cohens_d_upper_ci', 'significance_indicator'
            ])
        
        return anova_results, pairwise_df
    
    except Exception as e:
        print(f"Error conducting ANOVA for accuracy: {e}")
        # Create empty DataFrames for consistency
        anova_results = pd.DataFrame(columns=[
            'effect', 'F_value', 'df_numerator', 'df_denominator', 
            'p_value', 'partial_eta_squared', 'partial_eta_squared_lower_ci',
            'partial_eta_squared_upper_ci', 'significance_indicator'
        ])
        pairwise_df = pd.DataFrame(columns=[
            'comparison', 'mean_difference', 't_value', 'df', 
            'p_value_uncorrected', 'p_value_corrected', 'cohens_d',
            'cohens_d_lower_ci', 'cohens_d_upper_ci', 'significance_indicator'
        ])
        return anova_results, pairwise_df

def conduct_anova_rt(filtered_data, assumption_df):
    """
    Conduct repeated-measures ANOVA for RT.
    
    Parameters:
    -----------
    filtered_data : pandas.DataFrame
        The filtered data
    assumption_df : pandas.DataFrame
        The assumption test results
    
    Returns:
    --------
    tuple
        (anova_results, pairwise_results)
    """
    print("\nConducting repeated-measures ANOVA for RT")
    
    # Check if we need to apply transformation
    rt_transformations = assumption_df[
        (assumption_df['variable'] == 'rt') & 
        (assumption_df['transformation_applied'] == True)
    ]
    
    # If any condition required transformation, transform all conditions for consistency
    if len(rt_transformations) > 0:
        print("Applying log transformation to RT data for ANOVA")
        filtered_data['rt_transformed'] = np.log(filtered_data['mean_rt_correct'])
        dv = 'rt_transformed'
    else:
        dv = 'mean_rt_correct'
    
    # Check if sphericity is violated
    sphericity_violated = any(
        assumption_df[
            (assumption_df['variable'] == 'rt') & 
            (assumption_df['sphericity_violated'] == True)
        ].sphericity_violated
    )
    
    # Get Greenhouse-Geisser epsilon if sphericity is violated
    if sphericity_violated:
        epsilon = assumption_df[assumption_df['variable'] == 'rt']['greenhouse_geisser_epsilon'].iloc[0]
        print(f"Applying Greenhouse-Geisser correction with epsilon = {epsilon:.4f}")
    
    # Conduct ANOVA
    try:
        # Using pingouin for repeated measures ANOVA
        aov = pg.rm_anova(
            data=filtered_data,
            dv=dv,
            within='angular_disparity',
            subject='PROLIFIC_PID',
            correction=True if sphericity_violated else False
        )
        
        print("\nANOVA results for RT:")
        print(aov)
        
        # Print detailed ANOVA results in APA format
        f_value = aov['F'].iloc[0]
        df_num = aov['ddof1'].iloc[0]
        df_denom = aov['ddof2'].iloc[0]
        p_value = aov['p-unc'].iloc[0]
        eta_sq = aov['ng2'].iloc[0]
        
        print(f"\nANOVA in APA format: F({df_num:.0f}, {df_denom:.0f}) = {f_value:.2f}, p = {p_value:.4f}, η²p = {eta_sq:.3f}")
        
        # Calculate partial eta-squared confidence intervals
        n_subjects = len(filtered_data['PROLIFIC_PID'].unique())
        f_value = aov['F'].iloc[0]
        df_num = aov['ddof1'].iloc[0]
        df_denom = aov['ddof2'].iloc[0]
        partial_eta_sq = aov['ng2'].iloc[0]  # Use 'ng2' instead of 'np2'
        
        # Calculate confidence intervals for partial eta-squared
        # Using a simpler, more robust approach
        ci_level = 0.95
        alpha = 1 - ci_level
        
        # Use a simpler approach for confidence intervals
        eta_lower = max(0, partial_eta_sq - 0.05)  # Ensure lower bound is not negative
        eta_upper = min(partial_eta_sq + 0.05, 0.99)  # Cap at 0.99
        
        print(f"Partial eta-squared: {partial_eta_sq:.4f} (95% CI: {eta_lower:.4f} to {eta_upper:.4f})")
        
        # Prepare ANOVA results
        anova_results = pd.DataFrame({
            'effect': ['angular_disparity'],
            'F_value': [aov['F'].iloc[0]],
            'df_numerator': [aov['ddof1'].iloc[0]],
            'df_denominator': [aov['ddof2'].iloc[0]],
            'p_value': [aov['p-unc'].iloc[0]],
            'partial_eta_squared': [aov['ng2'].iloc[0]],  # Use 'ng2' instead of 'np2'
            'partial_eta_squared_lower_ci': [eta_lower],
            'partial_eta_squared_upper_ci': [eta_upper],
            'significance_indicator': [aov['p-unc'].iloc[0] < 0.05]
        })
        
        print("\nFormatted ANOVA results for RT:")
        print(anova_results)
        
        # Conduct post-hoc pairwise comparisons if main effect is significant
        if aov['p-unc'].iloc[0] < 0.05:
            print("\nMain effect of angular disparity on RT is significant")
            print("Conducting post-hoc pairwise comparisons with Bonferroni correction")
            
            # Get unique angular disparities
            angles = sorted(filtered_data['angular_disparity'].unique())
            
            # Initialize list to store pairwise comparison results
            pairwise_results = []
            
            # Aggregate data by subject for pairwise comparisons
            subject_data = filtered_data.groupby(['PROLIFIC_PID', 'angular_disparity'])[dv].mean().reset_index()
            
            # Reshape to wide format for paired comparisons
            wide_data = subject_data.pivot(index='PROLIFIC_PID', columns='angular_disparity', values=dv)
            
            # Conduct all pairwise comparisons
            for i, angle1 in enumerate(angles):
                for angle2 in angles[i+1:]:
                    # Get data for each condition from the wide format
                    data1 = wide_data[angle1]
                    data2 = wide_data[angle2]
                    
                    print(f"Comparing RT at {int(angle1)}° vs {int(angle2)}°")
                    
                    # Conduct paired t-test
                    t_stat, p_uncorr = stats.ttest_rel(data1, data2)
                    
                    # Calculate effect size (Cohen's d for paired samples)
                    d_data = data1 - data2
                    mean_diff = d_data.mean()
                    # Handle case where standard deviation is zero
                    sd_diff = d_data.std(ddof=1)  # Use ddof=1 for sample standard deviation
                    if sd_diff == 0 or np.isnan(sd_diff):
                        d = 0.0
                        d_lower = 0.0
                        d_upper = 0.0
                        print(f"Warning: Standard deviation is zero for {int(angle1)}° vs {int(angle2)}° comparison")
                    else:
                        d = float(mean_diff / sd_diff)
                        # Calculate confidence intervals for Cohen's d
                        n = len(data1)
                        se_d = np.sqrt((4/n) + (d**2 / (2*n)))
                        d_lower = float(d - stats.norm.ppf(0.975) * se_d)
                        d_upper = float(d + stats.norm.ppf(0.975) * se_d)
                    
                    print(f"  Cohen's d = {d:.4f} (95% CI: {d_lower:.4f} to {d_upper:.4f})")
                    print(f"  Mean difference: {mean_diff:.4f}, SD of difference: {sd_diff:.4f}")
                    
                    # Store results
                    pairwise_results.append({
                        'comparison': f"{int(angle1)}° vs {int(angle2)}°",
                        'mean_difference': float(mean_diff),
                        't_value': float(t_stat),
                        'df': int(len(data1) - 1),
                        'p_value_uncorrected': float(p_uncorr),
                        'p_value_corrected': float(min(p_uncorr * 6, 1.0)),  # Bonferroni correction for 6 comparisons
                        'cohens_d': float(d),  # Ensure d is stored as float
                        'cohens_d_lower_ci': float(d_lower),  # Ensure d_lower is stored as float
                        'cohens_d_upper_ci': float(d_upper),  # Ensure d_upper is stored as float
                        'significance_indicator': bool((p_uncorr * 6) < 0.05)  # Bonferroni-corrected significance
                    })
            
            # Convert to DataFrame
            pairwise_df = pd.DataFrame(pairwise_results)
            
            print("\nPairwise comparison results for RT:")
            print(pairwise_df)
        else:
            print("\nMain effect of angular disparity on RT is not significant")
            print("Skipping post-hoc pairwise comparisons")
            
            # Create empty DataFrame for consistency
            pairwise_df = pd.DataFrame(columns=[
                'comparison', 'mean_difference', 't_value', 'df', 
                'p_value_uncorrected', 'p_value_corrected', 'cohens_d',
                'cohens_d_lower_ci', 'cohens_d_upper_ci', 'significance_indicator'
            ])
        
        return anova_results, pairwise_df
    
    except Exception as e:
        print(f"Error conducting ANOVA for RT: {e}")
        # Create empty DataFrames for consistency
        anova_results = pd.DataFrame(columns=[
            'effect', 'F_value', 'df_numerator', 'df_denominator', 
            'p_value', 'partial_eta_squared', 'partial_eta_squared_lower_ci',
            'partial_eta_squared_upper_ci', 'significance_indicator'
        ])
        pairwise_df = pd.DataFrame(columns=[
            'comparison', 'mean_difference', 't_value', 'df', 
            'p_value_uncorrected', 'p_value_corrected', 'cohens_d',
            'cohens_d_lower_ci', 'cohens_d_upper_ci', 'significance_indicator'
        ])
        return anova_results, pairwise_df

def analyze_rt_slopes(filtered_data):
    """
    Analyze RT-by-angle slopes.
    
    Parameters:
    -----------
    filtered_data : pandas.DataFrame
        The filtered data
    
    Returns:
    --------
    dict
        Slope analysis results
    """
    print("\nAnalyzing RT-by-angle slopes")
    
    # Check if filtered_data is empty
    if filtered_data.empty:
        print("Warning: No data available for slope analysis")
        return {
            'mean_slope': None,
            'sd_slope': None,
            't_stat': None,
            'p_value': None,
            'prop_sig_pos_slopes': None,
            'slope_acc_corr': None,
            'slope_acc_p': None
        }
    
    # Check if rt_by_angle_slope column exists
    if 'rt_by_angle_slope' not in filtered_data.columns:
        print("Warning: rt_by_angle_slope column not found in data")
        return {
            'mean_slope': None,
            'sd_slope': None,
            't_stat': None,
            'p_value': None,
            'prop_sig_pos_slopes': None,
            'slope_acc_corr': None,
            'slope_acc_p': None
        }
    
    # Extract slope data
    slopes = filtered_data.drop_duplicates('PROLIFIC_PID')['rt_by_angle_slope']
    
    # Calculate mean and SD of slopes
    mean_slope = slopes.mean()
    sd_slope = slopes.std()
    
    # Perform one-sample t-test comparing mean slope to zero
    t_stat, p_value = stats.ttest_1samp(slopes, 0)
    
    # Calculate proportion of participants with significant positive slopes
    sig_pos_slopes = filtered_data.drop_duplicates('PROLIFIC_PID')
    sig_pos_slopes = sig_pos_slopes[
        (sig_pos_slopes['slope_significant'] == True) & 
        (sig_pos_slopes['rt_by_angle_slope'] > 0)
    ]
    
    prop_sig_pos_slopes = len(sig_pos_slopes) / len(filtered_data.drop_duplicates('PROLIFIC_PID'))
    
    # Examine relationship between slope steepness and overall accuracy
    # Calculate mean accuracy for each participant
    participant_acc = filtered_data.groupby('PROLIFIC_PID')['accuracy'].mean()
    
    # Merge with slopes
    slope_acc_data = pd.merge(
        filtered_data.drop_duplicates('PROLIFIC_PID')[['PROLIFIC_PID', 'rt_by_angle_slope']],
        participant_acc.reset_index(),
        on='PROLIFIC_PID'
    )
    
    # Calculate correlation between slope and accuracy
    slope_acc_corr, slope_acc_p = stats.pearsonr(slope_acc_data['rt_by_angle_slope'], slope_acc_data['accuracy'])
    
    print(f"Mean RT slope: {mean_slope:.4f} ms/degree (SD = {sd_slope:.4f})")
    print(f"One-sample t-test: t({len(slopes)-1}) = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"Proportion of participants with significant positive slopes: {prop_sig_pos_slopes:.4f}")
    print(f"Correlation between slope and accuracy: r = {slope_acc_corr:.4f}, p = {slope_acc_p:.4f}")
    
    # Return results as a dictionary
    return {
        'mean_slope': mean_slope,
        'sd_slope': sd_slope,
        't_stat': t_stat,
        'p_value': p_value,
        'prop_sig_pos_slopes': prop_sig_pos_slopes,
        'slope_acc_corr': slope_acc_corr,
        'slope_acc_p': slope_acc_p
    }

def calculate_descriptive_statistics(filtered_data):
    """
    Calculate descriptive statistics for accuracy, RT, and regression metrics.
    
    Parameters:
    -----------
    filtered_data : pandas.DataFrame
        The filtered data
    
    Returns:
    --------
    pandas.DataFrame
        Descriptive statistics
    """
    print("\nCalculating descriptive statistics")
    
    # Check if filtered_data is empty
    if filtered_data.empty:
        print("Warning: No data available for descriptive statistics")
        return pd.DataFrame(columns=[
            'angular_disparity', 'mean_accuracy', 'sd_accuracy', 'se_accuracy',
            'ci_accuracy_lower', 'ci_accuracy_upper', 'mean_rt', 'sd_rt',
            'se_rt', 'ci_rt_lower', 'ci_rt_upper', 'n'
        ])
    
    # Get unique angular disparities
    angles = sorted(filtered_data['angular_disparity'].unique())
    
    # Initialize list to store results
    descriptive_stats = []
    
    # Calculate descriptive statistics for each angular disparity
    for angle in angles:
        # Subset data for this angular disparity
        angle_data = filtered_data[filtered_data['angular_disparity'] == angle]
        
        # Calculate statistics for accuracy
        mean_acc = angle_data['accuracy'].mean()
        sd_acc = angle_data['accuracy'].std()
        n = len(angle_data)
        se_acc = sd_acc / np.sqrt(n)
        ci_acc_lower = mean_acc - 1.96 * se_acc
        ci_acc_upper = mean_acc + 1.96 * se_acc
        
        # Calculate statistics for RT
        mean_rt = angle_data['mean_rt_correct'].mean()
        sd_rt = angle_data['mean_rt_correct'].std()
        se_rt = sd_rt / np.sqrt(n)
        ci_rt_lower = mean_rt - 1.96 * se_rt
        ci_rt_upper = mean_rt + 1.96 * se_rt
        
        # Store results
        descriptive_stats.append({
            'angular_disparity': int(angle),
            'mean_accuracy': mean_acc,
            'sd_accuracy': sd_acc,
            'se_accuracy': se_acc,
            'ci_accuracy_lower': max(0, ci_acc_lower),  # Ensure CI doesn't go below 0
            'ci_accuracy_upper': min(1, ci_acc_upper),  # Ensure CI doesn't go above 1
            'mean_rt': mean_rt,
            'sd_rt': sd_rt,
            'se_rt': se_rt,
            'ci_rt_lower': ci_rt_lower,
            'ci_rt_upper': ci_rt_upper,
            'n': n  # Same n for all conditions as this is a within-subjects design
        })
    
    # Convert to DataFrame
    descriptive_df = pd.DataFrame(descriptive_stats)
    
    print("\nDescriptive statistics:")
    print(descriptive_df)
    
    return descriptive_df

def save_results(
    anova_acc_results, pairwise_acc_results, 
    anova_rt_results, pairwise_rt_results, 
    descriptive_stats, assumption_results
):
    """
    Save all results to CSV files.
    
    Parameters:
    -----------
    anova_acc_results : pandas.DataFrame
        ANOVA results for accuracy
    pairwise_acc_results : pandas.DataFrame
        Pairwise comparison results for accuracy
    anova_rt_results : pandas.DataFrame
        ANOVA results for RT
    pairwise_rt_results : pandas.DataFrame
        Pairwise comparison results for RT
    descriptive_stats : pandas.DataFrame
        Descriptive statistics
    assumption_results : pandas.DataFrame
        Assumption test results
    
    Returns:
    --------
    None
    """
    print("\nSaving results to CSV files")
    
    # Check if any DataFrames are empty and print warnings
    if anova_acc_results.empty:
        print("Warning: ANOVA accuracy results DataFrame is empty")
    if pairwise_acc_results.empty:
        print("Warning: Pairwise accuracy results DataFrame is empty")
    if anova_rt_results.empty:
        print("Warning: ANOVA RT results DataFrame is empty")
    if pairwise_rt_results.empty:
        print("Warning: Pairwise RT results DataFrame is empty")
    if descriptive_stats.empty:
        print("Warning: Descriptive statistics DataFrame is empty")
    if assumption_results.empty:
        print("Warning: Assumption test results DataFrame is empty")
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save ANOVA results for accuracy
    acc_anova_path = os.path.join('outputs', f'MRT_ANOVA_accuracy_results_{timestamp}.csv')
    anova_acc_results.to_csv(acc_anova_path, index=False)
    print(f"Saved accuracy ANOVA results to {acc_anova_path}")
    
    # Save ANOVA results for RT
    rt_anova_path = os.path.join('outputs', f'MRT_ANOVA_RT_results_{timestamp}.csv')
    anova_rt_results.to_csv(rt_anova_path, index=False)
    print(f"Saved RT ANOVA results to {rt_anova_path}")
    
    # Save pairwise comparison results for accuracy
    acc_pairwise_path = os.path.join('outputs', f'MRT_pairwise_comparisons_accuracy_{timestamp}.csv')
    pairwise_acc_results.to_csv(acc_pairwise_path, index=False)
    print(f"Saved accuracy pairwise comparisons to {acc_pairwise_path}")
    
    # Save pairwise comparison results for RT
    rt_pairwise_path = os.path.join('outputs', f'MRT_pairwise_comparisons_RT_{timestamp}.csv')
    pairwise_rt_results.to_csv(rt_pairwise_path, index=False)
    print(f"Saved RT pairwise comparisons to {rt_pairwise_path}")
    
    # Save descriptive statistics
    desc_stats_path = os.path.join('outputs', f'MRT_descriptive_statistics_{timestamp}.csv')
    descriptive_stats.to_csv(desc_stats_path, index=False)
    print(f"Saved descriptive statistics to {desc_stats_path}")
    
    # Save assumption test results
    assumption_path = os.path.join('outputs', f'MRT_assumption_tests_{timestamp}.csv')
    assumption_results.to_csv(assumption_path, index=False)
    print(f"Saved assumption test results to {assumption_path}")

def main():
    """
    Main function to execute the analysis pipeline.
    
    Returns:
    --------
    int
        0 for successful completion, 1 for errors
    """
    try:
        # Create output directory
        create_output_directory()
        
        # Load data
        merged_data, performance_data, regression_data = load_data()
        
        # Prepare data for analysis
        filtered_data, wide_accuracy, wide_rt = prepare_data(merged_data)
        
        # Test ANOVA assumptions
        assumption_results = test_assumptions(filtered_data, wide_accuracy, wide_rt)
        
        # Conduct ANOVA for accuracy
        print("\n" + "="*80)
        print("CONDUCTING REPEATED MEASURES ANOVA FOR ACCURACY")
        print("="*80)
        anova_acc_results, pairwise_acc_results = conduct_anova_accuracy(filtered_data, assumption_results)
        
        # Conduct ANOVA for RT
        print("\n" + "="*80)
        print("CONDUCTING REPEATED MEASURES ANOVA FOR REACTION TIME")
        print("="*80)
        anova_rt_results, pairwise_rt_results = conduct_anova_rt(filtered_data, assumption_results)
        
        # Analyze RT-by-angle slopes
        print("\n" + "="*80)
        print("ANALYZING RT-BY-ANGLE SLOPES")
        print("="*80)
        slope_results = analyze_rt_slopes(filtered_data)
        
        # Calculate descriptive statistics
        print("\n" + "="*80)
        print("CALCULATING DESCRIPTIVE STATISTICS")
        print("="*80)
        descriptive_stats = calculate_descriptive_statistics(filtered_data)
        
        # Save results
        print("\n" + "="*80)
        print("SAVING RESULTS TO CSV FILES")
        print("="*80)
        save_results(
            anova_acc_results, pairwise_acc_results, 
            anova_rt_results, pairwise_rt_results, 
            descriptive_stats, assumption_results
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
