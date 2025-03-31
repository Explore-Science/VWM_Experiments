import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime

def find_latest_file(pattern):
    """
    Find the most recent file matching the given pattern.
    
    Parameters:
    -----------
    pattern : str
        File pattern to match, including path
        
    Returns:
    --------
    str
        Path to the most recent file matching the pattern
    """
    matching_files = glob.glob(pattern)
    if not matching_files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    # Sort files by creation time, most recent first
    matching_files.sort(key=os.path.getctime, reverse=True)
    latest_file = matching_files[0]
    print(f"Latest file found: {latest_file}")
    return latest_file

def read_and_validate_csv(file_path, required_columns):
    """
    Read a CSV file and validate that it contains the required columns.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    required_columns : list
        List of column names that must be present in the CSV
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the CSV data, or None if validation fails
    """
    if file_path is None:
        print("No file path provided")
        return None
        
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Warning: File {file_path} is empty")
            return None
            
        print(f"\nRead file: {file_path}")
        print(f"Columns: {list(df.columns)}")
        print(f"Number of rows: {len(df)}")
        print(f"First 3 rows:\n{df.head(3)}")
        
        # Validate required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return None
            
        # Print unique values for key experimental parameters
        for col in df.columns:
            if col in ['effect', 'comparison']:
                print(f"Unique values for '{col}': {df[col].unique()}")
        
        # Print data statistics for important numerical columns
        numeric_cols = ['F_value', 'df_numerator', 'df_denominator', 'p_value', 
                        'partial_eta_squared', 't_value', 'df', 'cohens_d']
        for col in numeric_cols:
            if col in df.columns and len(df[col]) > 0:
                try:
                    print(f"Statistics for '{col}': min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
                except (TypeError, ValueError) as e:
                    print(f"Error calculating statistics for '{col}': {str(e)}")
                
        # Check for and report missing values in required columns
        missing_data = df[required_columns].isnull().sum()
        if missing_data.sum() > 0:
            print(f"Missing values in required columns:\n{missing_data[missing_data > 0]}")
            
        # Filter out rows with missing values in required columns
        original_row_count = len(df)
        df = df.dropna(subset=required_columns)
        filtered_row_count = len(df)
        if original_row_count > filtered_row_count:
            print(f"Dropped {original_row_count - filtered_row_count} rows with missing values")
            
        return df
        
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def extract_sample_size(df, test_type):
    """
    Extract sample size from degrees of freedom.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing test results
    test_type : str
        Type of test ('ANOVA' or 'Pairwise comparison')
        
    Returns:
    --------
    int
        Estimated sample size
    """
    if df is None:
        return None
        
    if test_type == 'ANOVA':
        # For ANOVA, df_denominator + 1 gives a good approximation of sample size
        if 'df_denominator' in df.columns and len(df) > 0:
            return int(df['df_denominator'].iloc[0] + 1)
        else:
            print(f"  Warning: Cannot extract sample size for ANOVA - missing df_denominator column or empty dataframe")
            return None
    else:  # For t-tests
        # For paired t-test, df + 1 gives the sample size
        if 'df' in df.columns and len(df) > 0:
            return int(df['df'].iloc[0] + 1)
        else:
            print(f"  Warning: Cannot extract sample size for t-test - missing df column or empty dataframe")
            return None

def calculate_partial_eta_squared_ci(f_value, df_num, df_denom, partial_eta_sq=None, alpha=0.05):
    """
    Calculate confidence intervals for partial eta-squared using non-central F distribution.
    
    Parameters:
    -----------
    f_value : float
        F-value from ANOVA
    df_num : float
        Numerator degrees of freedom
    df_denom : float
        Denominator degrees of freedom
    partial_eta_sq : float, optional
        Pre-calculated partial eta-squared value (if None, will be calculated)
    alpha : float, optional
        Alpha level for confidence interval (default: 0.05)
        
    Returns:
    --------
    tuple
        (lower CI, upper CI) for partial eta-squared
    """
    # Calculate partial eta-squared from F-value if not provided
    if partial_eta_sq is None:
        partial_eta_sq = (f_value * df_num) / (f_value * df_num + df_denom)
    
    # Calculate non-centrality parameter
    ncp = f_value * df_num
    
    # For very large F-values or degrees of freedom, use approximation to avoid overflow
    if f_value > 1000 or df_denom > 1000 or ncp > 1000:
        print(f"  Using approximation for CI calculation (large values: F={f_value:.2f}, df_denom={df_denom}, ncp={ncp:.2f})")
        # Use approximation based on standard error
        se = np.sqrt((4 * partial_eta_sq * (1 - partial_eta_sq)**2) / 
                    (df_num + df_denom - 1))
        return (max(0, partial_eta_sq - 1.96 * se), min(1, partial_eta_sq + 1.96 * se))
    
    # Calculate confidence intervals for non-centrality parameter
    # Lower bound
    def func_lower(x):
        try:
            return stats.ncf.cdf(f_value, df_num, df_denom, x) - (1 - alpha/2)
        except:
            # If calculation fails, return a large value to guide optimization
            return 999
    
    # Upper bound
    def func_upper(x):
        try:
            return stats.ncf.cdf(f_value, df_num, df_denom, x) - alpha/2
        except:
            # If calculation fails, return a large negative value to guide optimization
            return -999
    
    # Find bounds using root-finding
    try:
        # Use reasonable starting points and bounds
        ncp_lower = max(0.001, ncp/10)  # Avoid starting at exactly 0
        ncp_upper = min(1000, ncp * 2)  # Avoid extremely large values
        
        # For lower CI
        try:
            from scipy.optimize import brentq
            ncp_lower_ci = brentq(func_lower, ncp_lower, ncp, maxiter=100)
        except Exception as e:
            print(f"  Lower CI calculation failed, using approximation: {str(e)}")
            # Fallback: approximate with a simple method
            ncp_lower_ci = max(0.001, ncp - 1.96 * np.sqrt(2 * ncp))
        
        # For upper CI
        try:
            ncp_upper_ci = brentq(func_upper, ncp, ncp_upper, maxiter=100)
        except Exception as e:
            print(f"  Upper CI calculation failed, using approximation: {str(e)}")
            # Fallback: approximate with a simple method
            ncp_upper_ci = ncp + 1.96 * np.sqrt(2 * ncp)
        
        # Convert non-centrality parameter CIs to partial eta-squared CIs
        lower_ci = (ncp_lower_ci) / (ncp_lower_ci + df_denom)
        upper_ci = (ncp_upper_ci) / (ncp_upper_ci + df_denom)
        
        # Ensure CIs are within bounds and make logical sense
        lower_ci = max(0, min(lower_ci, partial_eta_sq))
        upper_ci = min(1, max(upper_ci, partial_eta_sq))
        
        return (lower_ci, upper_ci)
    except Exception as e:
        print(f"  Error calculating partial eta-squared CI: {str(e)}")
        # Return approximate CI as fallback
        se = np.sqrt((4 * partial_eta_sq * (1 - partial_eta_sq)**2) / 
                    (df_num + df_denom - 1))
        return (max(0, partial_eta_sq - 1.96 * se), min(1, partial_eta_sq + 1.96 * se))

def calculate_cohens_d(t_value, df, sample_size=None):
    """
    Calculate Cohen's d from t-value.
    
    Parameters:
    -----------
    t_value : float
        t-value from t-test
    df : float
        Degrees of freedom
    sample_size : float, optional
        Sample size (if None, estimated from df)
        
    Returns:
    --------
    float
        Cohen's d
    """
    if sample_size is None:
        # For paired t-test, df = n - 1, so n = df + 1
        # For independent t-test, df ≈ 2n - 2, so n ≈ (df + 2) / 2
        # We'll use a conservative estimate
        sample_size = df + 1
    
    # Calculate Cohen's d
    d = t_value / np.sqrt(sample_size)
    return d

def calculate_cohens_d_ci(d, n, alpha=0.05):
    """
    Calculate confidence intervals for Cohen's d.
    
    Parameters:
    -----------
    d : float
        Cohen's d value
    n : float
        Sample size
    alpha : float, optional
        Alpha level for confidence interval (default: 0.05)
        
    Returns:
    --------
    tuple
        (lower CI, upper CI) for Cohen's d
    """
    # Handle invalid inputs
    try:
        d = float(d)
        n = float(n)
    except (TypeError, ValueError):
        print(f"  Invalid inputs for Cohen's d CI calculation: d={d}, n={n}")
        return (np.nan, np.nan)
        
    if pd.isna(d) or pd.isna(n) or n <= 0:
        print(f"  Invalid inputs for Cohen's d CI calculation: d={d}, n={n}")
        return (np.nan, np.nan)
    
    # Standard error for d
    se = np.sqrt((4 + d**2) / n)
    
    # Calculate CI
    try:
        z = stats.norm.ppf(1 - alpha/2)
        lower_ci = d - z * se
        upper_ci = d + z * se
        return (lower_ci, upper_ci)
    except Exception as e:
        print(f"  Error calculating Cohen's d CI: {str(e)}")
        return (np.nan, np.nan)

def interpret_effect_size(effect_type, value):
    """
    Interpret effect size based on conventional thresholds.
    
    Parameters:
    -----------
    effect_type : str
        Type of effect size ('partial_eta_squared', 'cohens_d', or 'r')
    value : float
        Effect size value
        
    Returns:
    --------
    str
        Interpretation of effect size
    """
    # Handle NaN values
    if pd.isna(value):
        return "Not available"
    
    # Ensure value is a float
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "Invalid value"
    
    if effect_type == 'partial_eta_squared':
        # Thresholds from Cohen (1988): 0.01 (small), 0.06 (medium), 0.14 (large)
        if value < 0.01:
            return "Very small"
        elif value < 0.06:
            return "Small"
        elif value < 0.14:
            return "Medium"
        else:
            return "Large"
    elif effect_type == 'cohens_d':
        # Thresholds from Cohen (1988): 0.2 (small), 0.5 (medium), 0.8 (large)
        if abs(value) < 0.2:
            return "Very small"
        elif abs(value) < 0.5:
            return "Small"
        elif abs(value) < 0.8:
            return "Medium"
        else:
            return "Large"
    elif effect_type == 'r':
        # Correlation coefficient thresholds
        if abs(value) < 0.1:
            return "Very small"
        elif abs(value) < 0.3:
            return "Small"
        elif abs(value) < 0.5:
            return "Medium"
        else:
            return "Large"
    elif effect_type == 'r_squared':
        # R-squared thresholds
        if value < 0.01:
            return "Very small"
        elif value < 0.09:
            return "Small"
        elif value < 0.25:
            return "Medium"
        else:
            return "Large"
    else:
        return f"Unknown effect type: {effect_type}"

def calculate_power(test_type, effect_size, df_num=None, df_denom=None, df=None, alpha=0.05):
    """
    Calculate observed power for a given effect size.
    
    Parameters:
    -----------
    test_type : str
        Type of test ('F' for ANOVA, 't' for t-test)
    effect_size : float
        Effect size (partial eta-squared for ANOVA, Cohen's d for t-test)
        For t-tests, the absolute value will be used for power calculation
    df_num : float, optional
        Numerator degrees of freedom (for ANOVA)
    df_denom : float, optional
        Denominator degrees of freedom (for ANOVA)
    df : float, optional
        Degrees of freedom (for t-test)
    alpha : float, optional
        Alpha level (default: 0.05)
        
    Returns:
    --------
    float
        Observed power
    """
    # Handle invalid effect sizes
    try:
        effect_size = float(effect_size)
    except (TypeError, ValueError):
        print(f"  Invalid effect size type for power calculation: {type(effect_size)}")
        return np.nan
        
    if pd.isna(effect_size):
        print(f"  NaN effect size for power calculation")
        return np.nan
    
    # For t-tests, we'll handle negative values differently (using absolute value)
    # For ANOVA, partial eta-squared should always be positive
    if test_type == 'F' and effect_size <= 0:
        print(f"  Non-positive effect size for power calculation: {effect_size}")
        return np.nan
    
    try:
        if test_type == 'F':
            # For ANOVA
            if df_num is None or df_denom is None or pd.isna(df_num) or pd.isna(df_denom):
                print("  Missing or invalid degrees of freedom for power calculation")
                return np.nan
                
            try:
                df_num = float(df_num)
                df_denom = float(df_denom)
            except (TypeError, ValueError):
                print(f"  Invalid df types: df_num={type(df_num)}, df_denom={type(df_denom)}")
                return np.nan
                
            # Convert partial eta-squared to f
            f_squared = effect_size / (1 - effect_size)
            f = np.sqrt(f_squared)
            
            # For very large effect sizes or df values, use approximation to avoid overflow
            if f_squared > 100 or df_denom > 1000 or effect_size > 0.99:
                print(f"  Using power approximation for large values: f²={f_squared:.2f}, df_denom={df_denom}, effect_size={effect_size:.4f}")
                return 0.9999  # Extremely high power
            
            # Calculate non-centrality parameter
            ncp = f_squared * df_denom
            
            # Calculate critical F value
            try:
                f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
            except (ValueError, TypeError, RuntimeError) as e:
                print(f"  Error calculating critical F value: {str(e)}. Using approximation.")
                # Use approximation based on effect size
                if effect_size > 0.14:  # Large effect
                    return 0.95
                elif effect_size > 0.06:  # Medium effect
                    return 0.8
                else:  # Small effect
                    return 0.5
            
            try:
                # Calculate power with error handling
                with np.errstate(all='raise'):  # Convert warnings to exceptions
                    power = 1 - stats.ncf.cdf(f_crit, df_num, df_denom, ncp)
                
                return power
            except (RuntimeWarning, RuntimeError, OverflowError, FloatingPointError) as e:
                print(f"  Error in power calculation: {str(e)}. Using approximation.")
                # Use approximation based on effect size
                if effect_size > 0.14:  # Large effect
                    return 0.95
                elif effect_size > 0.06:  # Medium effect
                    return 0.8
                else:  # Small effect
                    return 0.5
            
        elif test_type == 't':
            # For t-test
            if df is None:
                print("  Missing degrees of freedom for power calculation")
                return np.nan
                
            # Cohen's d is the effect size - use absolute value for power calculation
            # since power depends on magnitude, not direction of effect
            d = abs(effect_size)
            print(f"  Using absolute effect size |d|={d:.4f} for power calculation")
            
            # For very large effect sizes or df values, use approximation
            if d > 5 or df > 1000:
                print(f"  Using power approximation for large values: |d|={d:.4f}, df={df}")
                return 0.9999 if d > 0.8 else 0.95
            
            # Calculate non-centrality parameter (for one-sample or paired t-test)
            # For simplicity, we'll assume paired t-test (n = df + 1)
            n = df + 1
            ncp = d * np.sqrt(n)
            
            # Calculate critical t value (two-tailed)
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            try:
                # Calculate power with error handling
                power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
                
                return power
            except (RuntimeWarning, RuntimeError, OverflowError) as e:
                print(f"  Error in power calculation: {str(e)}. Using approximation.")
                # Use approximation based on effect size
                if abs(d) > 0.8:
                    return 0.95
                elif abs(d) > 0.5:
                    return 0.8
                else:
                    return 0.5
        else:
            print(f"  Unknown test type: {test_type}")
            return np.nan
    except Exception as e:
        print(f"  Unexpected error in power calculation: {str(e)}")
        return np.nan

def calculate_min_detectable_effect(test_type, df_num=None, df_denom=None, df=None, alpha=0.05, target_power=0.8):
    """
    Calculate minimum detectable effect size for given parameters.
    
    Parameters:
    -----------
    test_type : str
        Type of test ('F' for ANOVA, 't' for t-test)
    df_num : float, optional
        Numerator degrees of freedom (for ANOVA)
    df_denom : float, optional
        Denominator degrees of freedom (for ANOVA)
    df : float, optional
        Degrees of freedom (for t-test)
    alpha : float, optional
        Alpha level (default: 0.05)
    target_power : float, optional
        Target power (default: 0.8)
        
    Returns:
    --------
    float
        Minimum detectable effect size
    """
    if test_type == 'F':
        try:
            # For ANOVA
            if df_num is None or df_denom is None or pd.isna(df_num) or pd.isna(df_denom):
                print("  Missing or invalid degrees of freedom for min effect calculation")
                return np.nan
                
            try:
                df_num = float(df_num)
                df_denom = float(df_denom)
            except (TypeError, ValueError):
                print(f"  Invalid df types: df_num={type(df_num)}, df_denom={type(df_denom)}")
                return np.nan
                
            # Calculate critical F value
            try:
                f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
            except Exception as e:
                print(f"  Error calculating critical F value: {str(e)}")
                return 0.06  # Return medium effect size as fallback
            
            # Function to find the non-centrality parameter for target power
            def power_diff(ncp):
                try:
                    power = 1 - stats.ncf.cdf(f_crit, df_num, df_denom, ncp)
                    return power - target_power
                except Exception:
                    # Return a large value to guide optimization away from problematic values
                    return 999
                
            # Find the non-centrality parameter using root-finding
            try:
                from scipy.optimize import brentq
                ncp = brentq(power_diff, 0.01, 100, maxiter=100)
                
                # Convert non-centrality parameter to partial eta-squared
                f_squared = ncp / df_denom
                partial_eta_squared = f_squared / (1 + f_squared)
                
                return partial_eta_squared
            except Exception as e:
                print(f"  Error in minimum detectable effect calculation: {str(e)}")
                # Fallback approximation
                # Using Cohen's f-squared values: small=0.02, medium=0.15, large=0.35
                f_squared_medium = 0.15
                partial_eta_squared = f_squared_medium / (1 + f_squared_medium)
                return partial_eta_squared
        except Exception as e:
            print(f"  Unexpected error in F-test min effect calculation: {str(e)}")
            return 0.06  # Return medium effect size as fallback
            
    elif test_type == 't':
        # For t-test
        try:
            if df is None or pd.isna(df):
                print("  Missing or invalid degrees of freedom for min effect calculation")
                return np.nan
                
            try:
                df = float(df)
            except (TypeError, ValueError):
                print(f"  Invalid df type: df={type(df)}")
                return np.nan
                
            # Calculate critical t value (two-tailed)
            try:
                t_crit = stats.t.ppf(1 - alpha/2, df)
            except Exception as e:
                print(f"  Error calculating critical t value: {str(e)}")
                return 0.5  # Return medium effect size as fallback
            
            # For simplicity, we'll assume paired t-test (n = df + 1)
            n = df + 1
            
            # Function to find the effect size (d) for target power
            def power_diff(d):
                try:
                    ncp = d * np.sqrt(n)
                    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
                    return power - target_power
                except Exception:
                    # Return a large value to guide optimization away from problematic values
                    return 999
                
            # Find the effect size using root-finding
            try:
                from scipy.optimize import brentq
                d = brentq(power_diff, 0.01, 2.0, maxiter=100)
                return d
            except Exception as e:
                print(f"  Error in minimum detectable effect calculation: {str(e)}")
                # Fallback to conventional medium effect size
                return 0.5
        except Exception as e:
            print(f"  Unexpected error in t-test min effect calculation: {str(e)}")
            return 0.5  # Return medium effect size as fallback
    else:
        print(f"Unknown test type: {test_type}")
        return np.nan

def process_anova_results(df, test_type, outcome_measure=None, sample_size=None):
    """
    Process ANOVA results to calculate effect sizes, CIs, power, etc.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ANOVA results
    test_type : str
        Type of test ('VA' for Visual Arrays, 'MRT' for Mental Rotation Task)
    outcome_measure : str, optional
        Outcome measure (for MRT: 'accuracy' or 'RT')
    sample_size : int, optional
        Sample size (if None, will be extracted from data)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed results
    """
    # Extract sample size from df if not provided
    if sample_size is None:
        sample_size = extract_sample_size(df, 'ANOVA')
    
    print(f"\nProcessing ANOVA results for {test_type}" + 
          (f" ({outcome_measure})" if outcome_measure else "") +
          f" with sample size: {sample_size}")
    
    # Initialize results dataframe
    results = []
    
    for _, row in df.iterrows():
        effect = row['effect']
        f_value = row['F_value']
        df_num = row['df_numerator']
        df_denom = row['df_denominator']
        p_value = row['p_value']
        
        # Use existing partial eta squared
        partial_eta_squared = row['partial_eta_squared']
        
        # Use existing CIs if available, otherwise calculate
        if 'partial_eta_squared_lower_ci' in row and 'partial_eta_squared_upper_ci' in row:
            partial_eta_squared_lower_ci = row['partial_eta_squared_lower_ci']
            partial_eta_squared_upper_ci = row['partial_eta_squared_upper_ci']
            print(f"  Using existing CIs for {effect}: [{partial_eta_squared_lower_ci:.3f}, {partial_eta_squared_upper_ci:.3f}]")
        else:
            partial_eta_squared_ci = calculate_partial_eta_squared_ci(f_value, df_num, df_denom, partial_eta_squared)
            partial_eta_squared_lower_ci = partial_eta_squared_ci[0]
            partial_eta_squared_upper_ci = partial_eta_squared_ci[1]
            print(f"  Calculated CIs for {effect}: [{partial_eta_squared_lower_ci:.3f}, {partial_eta_squared_upper_ci:.3f}]")
        
        # Interpret effect size
        effect_size_interpretation = interpret_effect_size('partial_eta_squared', partial_eta_squared)
        
        # Calculate observed power
        observed_power = calculate_power('F', partial_eta_squared, df_num, df_denom)
        
        # Calculate minimum detectable effect size
        min_detectable_effect = calculate_min_detectable_effect('F', df_num, df_denom)
        
        # Create result row
        result = {
            'effect': effect,
            'test_type': 'ANOVA',
            'F_value': f_value,
            't_value': np.nan,
            'df_numerator': df_num,
            'df_denominator': df_denom,
            'df': np.nan,
            'sample_size': sample_size,
            'partial_eta_squared': partial_eta_squared,
            'partial_eta_squared_lower_ci': partial_eta_squared_lower_ci,
            'partial_eta_squared_upper_ci': partial_eta_squared_upper_ci,
            'cohens_d': np.nan,
            'cohens_d_lower_ci': np.nan,
            'cohens_d_upper_ci': np.nan,
            'effect_size_interpretation': effect_size_interpretation,
            'observed_power': observed_power,
            'minimum_detectable_effect_size': min_detectable_effect
        }
        
        # Add outcome measure for MRT
        if outcome_measure:
            result['outcome_measure'] = outcome_measure
            
        results.append(result)
    
    return pd.DataFrame(results)

def process_pairwise_comparisons(df, test_type, outcome_measure=None, sample_size=None):
    """
    Process pairwise comparison results to calculate effect sizes, CIs, power, etc.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing pairwise comparison results
    test_type : str
        Type of test ('VA' for Visual Arrays, 'MRT' for Mental Rotation Task)
    outcome_measure : str, optional
        Outcome measure (for MRT: 'accuracy' or 'RT')
    sample_size : int, optional
        Sample size (if None, will be extracted from data)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed results
    """
    # Extract sample size from df if not provided
    if sample_size is None:
        sample_size = extract_sample_size(df, 'Pairwise comparison')
    
    print(f"\nProcessing pairwise comparisons for {test_type}" + 
          (f" ({outcome_measure})" if outcome_measure else "") +
          f" with sample size: {sample_size}")
    
    # Initialize results dataframe
    results = []
    
    for _, row in df.iterrows():
        comparison = row['comparison']
        mean_diff = row['mean_difference']
        t_value = row['t_value']
        df_val = row['df']
        p_value = row['p_value_corrected']
        
        # Use existing Cohen's d if available, otherwise calculate
        if 'cohens_d' in row:
            cohens_d = row['cohens_d']
            print(f"  Using existing Cohen's d for {comparison}: {cohens_d:.3f}")
        else:
            cohens_d = calculate_cohens_d(t_value, df_val, sample_size)
            print(f"  Calculated Cohen's d for {comparison}: {cohens_d:.3f}")
        
        # Use existing CIs if available, otherwise calculate
        if 'cohens_d_lower_ci' in row and 'cohens_d_upper_ci' in row:
            cohens_d_lower_ci = row['cohens_d_lower_ci']
            cohens_d_upper_ci = row['cohens_d_upper_ci']
            print(f"  Using existing CIs for {comparison}: [{cohens_d_lower_ci:.3f}, {cohens_d_upper_ci:.3f}]")
        else:
            cohens_d_ci = calculate_cohens_d_ci(cohens_d, sample_size)
            cohens_d_lower_ci = cohens_d_ci[0]
            cohens_d_upper_ci = cohens_d_ci[1]
            print(f"  Calculated CIs for {comparison}: [{cohens_d_lower_ci:.3f}, {cohens_d_upper_ci:.3f}]")
        
        # Interpret effect size
        effect_size_interpretation = interpret_effect_size('cohens_d', cohens_d)
        
        # Calculate observed power (note: power calculation will use absolute value internally)
        observed_power = calculate_power('t', cohens_d, df=df_val)
        print(f"  Calculated power for {comparison}: {observed_power:.4f} (based on effect magnitude |d|={abs(cohens_d):.4f})")
        
        # Calculate minimum detectable effect size
        min_detectable_effect = calculate_min_detectable_effect('t', df=df_val)
        
        # Create result row
        result = {
            'effect': comparison,
            'test_type': 'Pairwise comparison',
            'F_value': np.nan,
            't_value': t_value,
            'df_numerator': np.nan,
            'df_denominator': np.nan,
            'df': df_val,
            'sample_size': sample_size,
            'partial_eta_squared': np.nan,
            'partial_eta_squared_lower_ci': np.nan,
            'partial_eta_squared_upper_ci': np.nan,
            'cohens_d': cohens_d,
            'cohens_d_lower_ci': cohens_d_lower_ci,
            'cohens_d_upper_ci': cohens_d_upper_ci,
            'effect_size_interpretation': effect_size_interpretation,
            'observed_power': observed_power,
            'minimum_detectable_effect_size': min_detectable_effect
        }
        
        # Add outcome measure for MRT
        if outcome_measure:
            result['outcome_measure'] = outcome_measure
            
        results.append(result)
    
    return pd.DataFrame(results)

def create_effect_size_templates():
    """
    Create templates for reporting effect sizes.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with effect size templates
    """
    print("\nCreating effect size templates")
    
    templates = [
        {
            'analysis_type': 'ANOVA',
            'effect_size_type': 'partial_eta_squared',
            'small_threshold': 0.01,
            'medium_threshold': 0.06,
            'large_threshold': 0.14,
            'reporting_template': "The effect of {effect} was {interpretation} (ηp² = {value:.3f}, 95% CI [{lower:.3f}, {upper:.3f}]).",
            'confidence_interval_method': "Non-central F distribution"
        },
        {
            'analysis_type': 'Pairwise comparison',
            'effect_size_type': 'cohens_d',
            'small_threshold': 0.2,
            'medium_threshold': 0.5,
            'large_threshold': 0.8,
            'reporting_template': "The {effect} comparison showed a {interpretation} effect (d = {value:.3f}, 95% CI [{lower:.3f}, {upper:.3f}]).",
            'confidence_interval_method': "Standard error approximation"
        },
        {
            'analysis_type': 'Correlation',
            'effect_size_type': 'pearson_r',
            'small_threshold': 0.1,
            'medium_threshold': 0.3,
            'large_threshold': 0.5,
            'reporting_template': "The correlation between {var1} and {var2} was {interpretation} (r = {value:.3f}, 95% CI [{lower:.3f}, {upper:.3f}]).",
            'confidence_interval_method': "Fisher's Z transformation"
        },
        {
            'analysis_type': 'Regression',
            'effect_size_type': 'r_squared',
            'small_threshold': 0.01,
            'medium_threshold': 0.09,
            'large_threshold': 0.25,
            'reporting_template': "The model explained a {interpretation} amount of variance (R² = {value:.3f}, 95% CI [{lower:.3f}, {upper:.3f}]).",
            'confidence_interval_method': "Bootstrap"
        },
        {
            'analysis_type': 'Regression',
            'effect_size_type': 'standardized_beta',
            'small_threshold': 0.1,
            'medium_threshold': 0.3,
            'large_threshold': 0.5,
            'reporting_template': "The effect of {predictor} was {interpretation} (β = {value:.3f}, 95% CI [{lower:.3f}, {upper:.3f}]).",
            'confidence_interval_method': "Bootstrap"
        }
    ]
    
    return pd.DataFrame(templates)

def main():
    """Main function to execute the script."""
    print("Starting effect size calculation script")
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory")
    
    # Find the latest input files
    va_anova_file = find_latest_file('outputs/VA_ANOVA_results_*.csv')
    va_pairwise_file = find_latest_file('outputs/VA_pairwise_comparisons_*.csv')
    mrt_accuracy_anova_file = find_latest_file('outputs/MRT_ANOVA_accuracy_results_*.csv')
    mrt_rt_anova_file = find_latest_file('outputs/MRT_ANOVA_RT_results_*.csv')
    mrt_accuracy_pairwise_file = find_latest_file('outputs/MRT_pairwise_comparisons_accuracy_*.csv')
    mrt_rt_pairwise_file = find_latest_file('outputs/MRT_pairwise_comparisons_RT_*.csv')
    
    # Define required columns for each file
    va_anova_cols = ['effect', 'F_value', 'df_numerator', 'df_denominator', 
                     'p_value', 'partial_eta_squared', 'significance_indicator']
    va_pairwise_cols = ['comparison', 'mean_difference', 't_value', 'df', 
                        'p_value_corrected', 'significance_indicator']
    mrt_anova_cols = ['effect', 'F_value', 'df_numerator', 'df_denominator', 
                      'p_value', 'partial_eta_squared', 'significance_indicator']
    mrt_pairwise_cols = ['comparison', 'mean_difference', 't_value', 'df', 
                         'p_value_corrected', 'significance_indicator']
    
    # Optional columns that may exist in input files
    optional_anova_cols = ['partial_eta_squared_lower_ci', 'partial_eta_squared_upper_ci']
    optional_pairwise_cols = ['cohens_d', 'cohens_d_lower_ci', 'cohens_d_upper_ci', 'p_value_uncorrected']
    
    # Read and validate input files
    va_anova_df = read_and_validate_csv(va_anova_file, va_anova_cols)
    va_pairwise_df = read_and_validate_csv(va_pairwise_file, va_pairwise_cols)
    mrt_accuracy_anova_df = read_and_validate_csv(mrt_accuracy_anova_file, mrt_anova_cols)
    mrt_rt_anova_df = read_and_validate_csv(mrt_rt_anova_file, mrt_anova_cols)
    mrt_accuracy_pairwise_df = read_and_validate_csv(mrt_accuracy_pairwise_file, mrt_pairwise_cols)
    mrt_rt_pairwise_df = read_and_validate_csv(mrt_rt_pairwise_file, mrt_pairwise_cols)
    
    # Extract sample sizes from data
    va_sample_size = extract_sample_size(va_anova_df, 'ANOVA') if va_anova_df is not None else None
    va_pairwise_sample_size = extract_sample_size(va_pairwise_df, 'Pairwise comparison') if va_pairwise_df is not None else None
    mrt_accuracy_sample_size = extract_sample_size(mrt_accuracy_anova_df, 'ANOVA') if mrt_accuracy_anova_df is not None else None
    mrt_rt_sample_size = extract_sample_size(mrt_rt_anova_df, 'ANOVA') if mrt_rt_anova_df is not None else None
    
    print(f"\nExtracted sample sizes:")
    print(f"  VA ANOVA: {va_sample_size}")
    print(f"  VA Pairwise: {va_pairwise_sample_size}")
    print(f"  MRT Accuracy ANOVA: {mrt_accuracy_sample_size}")
    print(f"  MRT RT ANOVA: {mrt_rt_sample_size}")
    
    # Process VA ANOVA results
    va_results = []
    if va_anova_df is not None:
        va_anova_results = process_anova_results(va_anova_df, 'VA', sample_size=va_sample_size)
        va_results.append(va_anova_results)
    
    # Process VA pairwise comparisons
    if va_pairwise_df is not None:
        va_pairwise_results = process_pairwise_comparisons(va_pairwise_df, 'VA', sample_size=va_pairwise_sample_size)
        va_results.append(va_pairwise_results)
    
    # Combine VA results
    if va_results:
        va_combined_results = pd.concat(va_results, ignore_index=True)
        print("\nVA combined results:")
        print(va_combined_results.head(2))
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        va_output_file = f"outputs/VA_effect_sizes_{timestamp}.csv"
        
        # Save VA results
        va_combined_results.to_csv(va_output_file, index=False)
        print(f"Saved VA effect sizes to {va_output_file}")
    else:
        print("No VA results to save")
    
    # Process MRT ANOVA and pairwise comparison results
    mrt_results = []
    
    # Process MRT accuracy ANOVA
    if mrt_accuracy_anova_df is not None:
        mrt_accuracy_anova_results = process_anova_results(
            mrt_accuracy_anova_df, 'MRT', outcome_measure='accuracy', sample_size=mrt_accuracy_sample_size)
        mrt_results.append(mrt_accuracy_anova_results)
    
    # Process MRT RT ANOVA
    if mrt_rt_anova_df is not None:
        mrt_rt_anova_results = process_anova_results(
            mrt_rt_anova_df, 'MRT', outcome_measure='RT', sample_size=mrt_rt_sample_size)
        mrt_results.append(mrt_rt_anova_results)
    
    # Process MRT accuracy pairwise comparisons
    if mrt_accuracy_pairwise_df is not None:
        mrt_accuracy_pairwise_results = process_pairwise_comparisons(
            mrt_accuracy_pairwise_df, 'MRT', outcome_measure='accuracy', 
            sample_size=extract_sample_size(mrt_accuracy_pairwise_df, 'Pairwise comparison'))
        mrt_results.append(mrt_accuracy_pairwise_results)
    
    # Process MRT RT pairwise comparisons
    if mrt_rt_pairwise_df is not None:
        mrt_rt_pairwise_results = process_pairwise_comparisons(
            mrt_rt_pairwise_df, 'MRT', outcome_measure='RT', 
            sample_size=extract_sample_size(mrt_rt_pairwise_df, 'Pairwise comparison'))
        mrt_results.append(mrt_rt_pairwise_results)
    
    # Combine MRT results
    if mrt_results:
        mrt_combined_results = pd.concat(mrt_results, ignore_index=True)
        print("\nMRT combined results:")
        print(mrt_combined_results.head(2))
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mrt_output_file = f"outputs/MRT_effect_sizes_{timestamp}.csv"
        
        # Save MRT results
        mrt_combined_results.to_csv(mrt_output_file, index=False)
        print(f"Saved MRT effect sizes to {mrt_output_file}")
    else:
        print("No MRT results to save")
    
    # Create effect size templates
    templates_df = create_effect_size_templates()
    print("\nEffect size templates:")
    print(templates_df.head(2))
    
    # Save templates
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    templates_output_file = f"outputs/effect_size_templates_{timestamp}.csv"
    templates_df.to_csv(templates_output_file, index=False)
    print(f"Saved effect size templates to {templates_output_file}")
    
    print("Finished execution")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
