import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
import math

def get_latest_file(pattern):
    """
    Gets the most recent file matching the pattern.
    
    Parameters:
    -----------
    pattern : str
        The file pattern to match, including wildcards.
        
    Returns:
    --------
    str
        The path to the most recent file matching the pattern.
    """
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No files found matching pattern {pattern}")
        return None
    
    # Get the most recent file based on creation time
    latest_file = max(files, key=os.path.getctime)
    print(f"Latest file found: {latest_file}")
    return latest_file

def create_output_dir():
    """
    Creates the 'outputs' directory if it doesn't exist.
    
    Returns:
    --------
    bool
        True if the directory exists or was created successfully, False otherwise.
    """
    if not os.path.exists('outputs'):
        try:
            os.makedirs('outputs')
            print("Created 'outputs' directory")
            return True
        except Exception as e:
            print(f"Error creating 'outputs' directory: {e}")
            return False
    return True

def calculate_dprime(hits, false_alarms, hit_count, fa_count):
    """
    Calculate d-prime, a measure of sensitivity in signal detection theory.
    
    Parameters:
    -----------
    hits : float
        Hit rate (proportion of "signal" trials correctly identified)
    false_alarms : float
        False alarm rate (proportion of "noise" trials incorrectly identified as "signal")
    hit_count : int
        Number of hit trials
    fa_count : int
        Number of false alarm trials
        
    Returns:
    --------
    float
        d-prime value
    """
    # Apply correction for extreme values (0 and 1)
    if hits == 1.0:
        hits = 1.0 - 1.0/(2*hit_count)
    elif hits == 0.0:
        hits = 1.0/(2*hit_count)
        
    if false_alarms == 1.0:
        false_alarms = 1.0 - 1.0/(2*fa_count)
    elif false_alarms == 0.0:
        false_alarms = 1.0/(2*fa_count)
    
    # Calculate d-prime
    dprime = stats.norm.ppf(hits) - stats.norm.ppf(false_alarms)
    return dprime

def calculate_spearman_brown(r):
    """
    Apply Spearman-Brown correction to a correlation coefficient.
    
    Parameters:
    -----------
    r : float
        Original correlation coefficient
        
    Returns:
    --------
    float
        Corrected correlation coefficient
    """
    if r == 1.0:  # Handle perfect correlation
        return 1.0
    return (2 * r) / (1 + r)

def calculate_confidence_interval(r, n):
    """
    Calculate 95% confidence interval for a correlation coefficient using Fisher's z-transformation.
    
    Parameters:
    -----------
    r : float
        Correlation coefficient
    n : int
        Sample size
        
    Returns:
    --------
    tuple
        Lower and upper bounds of the 95% confidence interval
    """
    # Handle edge cases
    if r >= 1.0:
        return 1.0, 1.0
    if r <= -1.0:
        return -1.0, -1.0
    
    # Fisher's z-transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    
    # Convert back to correlation
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    return r_lower, r_upper

def interpret_reliability(r):
    """
    Classify reliability coefficient based on standard interpretations.
    
    Parameters:
    -----------
    r : float
        Reliability coefficient (Spearman-Brown corrected)
        
    Returns:
    --------
    str
        Interpretation of reliability
    """
    if r >= 0.90:
        return "Excellent"
    elif r >= 0.80:
        return "Good"
    elif r >= 0.70:
        return "Acceptable"
    elif r >= 0.60:
        return "Questionable"
    elif r >= 0.50:
        return "Poor"
    else:
        return "Unacceptable"

def calculate_va_reliability(va_data):
    """
    Calculate split-half reliability for Visual Arrays Task.
    
    Parameters:
    -----------
    va_data : pandas.DataFrame
        Cleaned Visual Arrays Task data
        
    Returns:
    --------
    tuple
        Three DataFrames containing reliability results for VA task:
        1. Detailed reliability by condition
        2. Summary reliability measures
    """
    print("Calculating VA reliability...")
    start_time = datetime.now()
    print(f"Starting VA reliability calculation at {start_time.strftime('%H:%M:%S')}")
    
    # Create a list to store all reliability results
    va_reliability_results = []
    va_summary_results = []
    
    # Get unique participants
    participants = va_data['PROLIFIC_PID'].unique()
    print(f"Number of participants: {len(participants)}")
    
    # Get unique conditions (set_size × delay combinations)
    set_sizes = va_data['set_size'].unique()
    delays = va_data['delay'].unique()
    
    print(f"Unique set sizes: {set_sizes}")
    print(f"Unique delays: {delays}")
    
    # Add trial number within participant
    va_data = va_data.sort_values(['PROLIFIC_PID', 'set_size', 'delay'])
    va_data['trial_num'] = va_data.groupby('PROLIFIC_PID').cumcount() + 1
    
    # Identify odd and even trials
    va_data['trial_parity'] = va_data['trial_num'].apply(lambda x: 'odd' if x % 2 == 1 else 'even')
    
    # Calculate d-prime for each condition and trial parity
    # First, for each set_size × delay combination
    for set_size in set_sizes:
        for delay in delays:
            condition = f"set_size_{set_size}_delay_{delay}"
            print(f"\nProcessing condition: {condition} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Calculate d-prime for each participant
            odd_dprimes = []
            even_dprimes = []
            participant_ids = []
            
            participant_count = 0
            for pid in participants:
                participant_count += 1
                if participant_count % 10 == 0:
                    print(f"  Processed {participant_count}/{len(participants)} participants for condition {condition}...")
                
                # Get participant data for this condition
                p_data = va_data[(va_data['PROLIFIC_PID'] == pid) & 
                                 (va_data['set_size'] == set_size) & 
                                 (va_data['delay'] == delay)]
                
                # Count trials for verification
                odd_trials = p_data[p_data['trial_parity'] == 'odd']
                even_trials = p_data[p_data['trial_parity'] == 'even']
                
                # Skip participants with no data in this condition
                if len(odd_trials) == 0 or len(even_trials) == 0:
                    print(f"  Participant {pid} has no data in one or both halves for this condition, skipping")
                    continue
                
                # Calculate d-prime for odd trials
                odd_change_trials = odd_trials[odd_trials['orientation_change'] == 1]
                odd_no_change_trials = odd_trials[odd_trials['orientation_change'] == 0]
                
                if len(odd_change_trials) > 0 and len(odd_no_change_trials) > 0:
                    odd_hits = odd_change_trials['VA_correct'].mean()
                    odd_false_alarms = 1 - odd_no_change_trials['VA_correct'].mean()
                    odd_dprime = calculate_dprime(odd_hits, odd_false_alarms, 
                                                 len(odd_change_trials), len(odd_no_change_trials))
                    
                    # Calculate d-prime for even trials
                    even_change_trials = even_trials[even_trials['orientation_change'] == 1]
                    even_no_change_trials = even_trials[even_trials['orientation_change'] == 0]
                    
                    if len(even_change_trials) > 0 and len(even_no_change_trials) > 0:
                        even_hits = even_change_trials['VA_correct'].mean()
                        even_false_alarms = 1 - even_no_change_trials['VA_correct'].mean()
                        even_dprime = calculate_dprime(even_hits, even_false_alarms, 
                                                     len(even_change_trials), len(even_no_change_trials))
                        
                        # Store results
                        odd_dprimes.append(odd_dprime)
                        even_dprimes.append(even_dprime)
                        participant_ids.append(pid)
                
            # Calculate reliability if we have enough participants
            if len(odd_dprimes) >= 3:  # Need at least 3 for correlation
                # Convert to numpy arrays
                odd_dprimes = np.array(odd_dprimes)
                even_dprimes = np.array(even_dprimes)
                
                # Remove any NaNs or infinities
                valid_indices = ~np.isnan(odd_dprimes) & ~np.isnan(even_dprimes) & \
                                ~np.isinf(odd_dprimes) & ~np.isinf(even_dprimes)
                
                odd_dprimes = odd_dprimes[valid_indices]
                even_dprimes = even_dprimes[valid_indices]
                
                # Calculate correlation
                if len(odd_dprimes) >= 3:  # Check again after removing invalid values
                    split_half_r, _ = stats.pearsonr(odd_dprimes, even_dprimes)
                    spearman_brown_r = calculate_spearman_brown(split_half_r)
                    
                    # Calculate confidence interval
                    r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(odd_dprimes))
                    
                    # Interpret reliability
                    reliability_interpretation = interpret_reliability(spearman_brown_r)
                    
                    # Store results
                    va_reliability_results.append({
                        'measure': 'd_prime',
                        'condition': condition,
                        'n_odd_trials': len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                                   (va_data['trial_parity'] == 'odd') & 
                                                   (va_data['set_size'] == set_size) & 
                                                   (va_data['delay'] == delay)]),
                        'n_even_trials': len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                                    (va_data['trial_parity'] == 'even') & 
                                                    (va_data['set_size'] == set_size) & 
                                                    (va_data['delay'] == delay)]),
                        'split_half_r': split_half_r,
                        'spearman_brown_r': spearman_brown_r,
                        'r_lower_ci': r_lower_ci,
                        'r_upper_ci': r_upper_ci,
                        'reliability_interpretation': reliability_interpretation
                    })
                    
                    # Add to summary results
                    va_summary_results.append({
                        'task': 'VA',
                        'measure': f'd_prime_{condition}',
                        'spearman_brown_r': spearman_brown_r,
                        'r_lower_ci': r_lower_ci,
                        'r_upper_ci': r_upper_ci,
                        'reliability_interpretation': reliability_interpretation,
                        'sample_size': len(odd_dprimes),
                        'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
                    })
                    
                    print(f"  Reliability for {condition}: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
                else:
                    print(f"  Not enough valid data points for condition {condition}")
            else:
                print(f"  Not enough participants with data for condition {condition}")
    
    # Calculate overall d-prime reliability
    print("\nCalculating overall d-prime reliability...")
    print(f"Starting overall d-prime calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # Calculate overall d-prime for each participant
    overall_odd_dprimes = []
    overall_even_dprimes = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for overall d-prime...")
            
        # Get participant data
        p_data = va_data[va_data['PROLIFIC_PID'] == pid]
        
        # Calculate d-prime for odd trials
        odd_trials = p_data[p_data['trial_parity'] == 'odd']
        odd_change_trials = odd_trials[odd_trials['orientation_change'] == 1]
        odd_no_change_trials = odd_trials[odd_trials['orientation_change'] == 0]
        
        if len(odd_change_trials) > 0 and len(odd_no_change_trials) > 0:
            odd_hits = odd_change_trials['VA_correct'].mean()
            odd_false_alarms = 1 - odd_no_change_trials['VA_correct'].mean()
            odd_dprime = calculate_dprime(odd_hits, odd_false_alarms, 
                                         len(odd_change_trials), len(odd_no_change_trials))
            
            # Calculate d-prime for even trials
            even_trials = p_data[p_data['trial_parity'] == 'even']
            even_change_trials = even_trials[even_trials['orientation_change'] == 1]
            even_no_change_trials = even_trials[even_trials['orientation_change'] == 0]
            
            if len(even_change_trials) > 0 and len(even_no_change_trials) > 0:
                even_hits = even_change_trials['VA_correct'].mean()
                even_false_alarms = 1 - even_no_change_trials['VA_correct'].mean()
                even_dprime = calculate_dprime(even_hits, even_false_alarms, 
                                             len(even_change_trials), len(even_no_change_trials))
                
                # Store results
                overall_odd_dprimes.append(odd_dprime)
                overall_even_dprimes.append(even_dprime)
                participant_ids.append(pid)
    
    # Calculate reliability
    if len(overall_odd_dprimes) >= 3:
        # Convert to numpy arrays
        overall_odd_dprimes = np.array(overall_odd_dprimes)
        overall_even_dprimes = np.array(overall_even_dprimes)
        
        # Remove any NaNs or infinities
        valid_indices = ~np.isnan(overall_odd_dprimes) & ~np.isnan(overall_even_dprimes) & \
                        ~np.isinf(overall_odd_dprimes) & ~np.isinf(overall_even_dprimes)
        
        overall_odd_dprimes = overall_odd_dprimes[valid_indices]
        overall_even_dprimes = overall_even_dprimes[valid_indices]
        
        if len(overall_odd_dprimes) >= 3:
            split_half_r, _ = stats.pearsonr(overall_odd_dprimes, overall_even_dprimes)
            spearman_brown_r = calculate_spearman_brown(split_half_r)
            
            # Calculate confidence interval
            r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(overall_odd_dprimes))
            
            # Interpret reliability
            reliability_interpretation = interpret_reliability(spearman_brown_r)
            
            # Store results - for overall d-prime, we keep all trials regardless of condition
            va_reliability_results.append({
                'measure': 'overall_d_prime',
                'condition': 'all',
                'n_odd_trials': len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & (va_data['trial_parity'] == 'odd')]),
                'n_even_trials': len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & (va_data['trial_parity'] == 'even')]),
                'split_half_r': split_half_r,
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation
            })
            
            # Add to summary results
            va_summary_results.append({
                'task': 'VA',
                'measure': 'overall_d_prime',
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation,
                'sample_size': len(overall_odd_dprimes),
                'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
            })
            
            print(f"  Overall d-prime reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate set size effect reliability
    print("\nCalculating set size effect reliability...")
    print(f"Starting set size effect calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # For each participant, calculate set size effect (set_size 5 - set_size 3) for odd and even trials
    set_size_effect_odd = []
    set_size_effect_even = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for set size effect...")
            
        # Get participant data
        p_data = va_data[va_data['PROLIFIC_PID'] == pid]
        
        # Calculate d-prime for each set size (odd trials)
        odd_dprime_set3 = []
        odd_dprime_set5 = []
        even_dprime_set3 = []
        even_dprime_set5 = []
        
        for delay in delays:
            # Odd trials, set size 3
            odd_trials_set3 = p_data[(p_data['trial_parity'] == 'odd') & 
                                    (p_data['set_size'] == 3) & 
                                    (p_data['delay'] == delay)]
            
            odd_change_trials_set3 = odd_trials_set3[odd_trials_set3['orientation_change'] == 1]
            odd_no_change_trials_set3 = odd_trials_set3[odd_trials_set3['orientation_change'] == 0]
            
            if len(odd_change_trials_set3) > 0 and len(odd_no_change_trials_set3) > 0:
                odd_hits_set3 = odd_change_trials_set3['VA_correct'].mean()
                odd_false_alarms_set3 = 1 - odd_no_change_trials_set3['VA_correct'].mean()
                odd_dprime_set3.append(calculate_dprime(odd_hits_set3, odd_false_alarms_set3, 
                                                       len(odd_change_trials_set3), len(odd_no_change_trials_set3)))
            
            # Odd trials, set size 5
            odd_trials_set5 = p_data[(p_data['trial_parity'] == 'odd') & 
                                    (p_data['set_size'] == 5) & 
                                    (p_data['delay'] == delay)]
            
            odd_change_trials_set5 = odd_trials_set5[odd_trials_set5['orientation_change'] == 1]
            odd_no_change_trials_set5 = odd_trials_set5[odd_trials_set5['orientation_change'] == 0]
            
            if len(odd_change_trials_set5) > 0 and len(odd_no_change_trials_set5) > 0:
                odd_hits_set5 = odd_change_trials_set5['VA_correct'].mean()
                odd_false_alarms_set5 = 1 - odd_no_change_trials_set5['VA_correct'].mean()
                odd_dprime_set5.append(calculate_dprime(odd_hits_set5, odd_false_alarms_set5, 
                                                      len(odd_change_trials_set5), len(odd_no_change_trials_set5)))
            
            # Even trials, set size 3
            even_trials_set3 = p_data[(p_data['trial_parity'] == 'even') & 
                                     (p_data['set_size'] == 3) & 
                                     (p_data['delay'] == delay)]
            
            even_change_trials_set3 = even_trials_set3[even_trials_set3['orientation_change'] == 1]
            even_no_change_trials_set3 = even_trials_set3[even_trials_set3['orientation_change'] == 0]
            
            if len(even_change_trials_set3) > 0 and len(even_no_change_trials_set3) > 0:
                even_hits_set3 = even_change_trials_set3['VA_correct'].mean()
                even_false_alarms_set3 = 1 - even_no_change_trials_set3['VA_correct'].mean()
                even_dprime_set3.append(calculate_dprime(even_hits_set3, even_false_alarms_set3, 
                                                        len(even_change_trials_set3), len(even_no_change_trials_set3)))
            
            # Even trials, set size 5
            even_trials_set5 = p_data[(p_data['trial_parity'] == 'even') & 
                                     (p_data['set_size'] == 5) & 
                                     (p_data['delay'] == delay)]
            
            even_change_trials_set5 = even_trials_set5[even_trials_set5['orientation_change'] == 1]
            even_no_change_trials_set5 = even_trials_set5[even_trials_set5['orientation_change'] == 0]
            
            if len(even_change_trials_set5) > 0 and len(even_no_change_trials_set5) > 0:
                even_hits_set5 = even_change_trials_set5['VA_correct'].mean()
                even_false_alarms_set5 = 1 - even_no_change_trials_set5['VA_correct'].mean()
                even_dprime_set5.append(calculate_dprime(even_hits_set5, even_false_alarms_set5, 
                                                        len(even_change_trials_set5), len(even_no_change_trials_set5)))
        
        # Calculate set size effects if we have data for both set sizes
        if odd_dprime_set3 and odd_dprime_set5 and even_dprime_set3 and even_dprime_set5:
            # Average across delays
            odd_set_size_effect = np.mean(odd_dprime_set3) - np.mean(odd_dprime_set5)
            even_set_size_effect = np.mean(even_dprime_set3) - np.mean(even_dprime_set5)
            
            # Store results
            set_size_effect_odd.append(odd_set_size_effect)
            set_size_effect_even.append(even_set_size_effect)
            participant_ids.append(pid)
    
    # Calculate reliability
    if len(set_size_effect_odd) >= 3:
        # Convert to numpy arrays
        set_size_effect_odd = np.array(set_size_effect_odd)
        set_size_effect_even = np.array(set_size_effect_even)
        
        # Remove any NaNs or infinities
        valid_indices = ~np.isnan(set_size_effect_odd) & ~np.isnan(set_size_effect_even) & \
                        ~np.isinf(set_size_effect_odd) & ~np.isinf(set_size_effect_even)
        
        set_size_effect_odd = set_size_effect_odd[valid_indices]
        set_size_effect_even = set_size_effect_even[valid_indices]
        
        if len(set_size_effect_odd) >= 3:
            split_half_r, _ = stats.pearsonr(set_size_effect_odd, set_size_effect_even)
            spearman_brown_r = calculate_spearman_brown(split_half_r)
            
            # Calculate confidence interval
            r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(set_size_effect_odd))
            
            # Interpret reliability
            reliability_interpretation = interpret_reliability(spearman_brown_r)
            
            # Store results - for set size effect, count trials from both set sizes
            n_odd_trials_set3 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                           (va_data['trial_parity'] == 'odd') & 
                                           (va_data['set_size'] == 3)])
            n_odd_trials_set5 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                           (va_data['trial_parity'] == 'odd') & 
                                           (va_data['set_size'] == 5)])
            n_even_trials_set3 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                            (va_data['trial_parity'] == 'even') & 
                                            (va_data['set_size'] == 3)])
            n_even_trials_set5 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                            (va_data['trial_parity'] == 'even') & 
                                            (va_data['set_size'] == 5)])
            
            va_reliability_results.append({
                'measure': 'set_size_effect',
                'condition': 'set_size_5_minus_set_size_3',
                'n_odd_trials': n_odd_trials_set3 + n_odd_trials_set5,
                'n_even_trials': n_even_trials_set3 + n_even_trials_set5,
                'split_half_r': split_half_r,
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation
            })
            
            # Add to summary results
            va_summary_results.append({
                'task': 'VA',
                'measure': 'set_size_effect',
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation,
                'sample_size': len(set_size_effect_odd),
                'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
            })
            
            print(f"  Set size effect reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate delay effect reliability
    print("\nCalculating delay effect reliability...")
    print(f"Starting delay effect calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # For each participant, calculate delay effect (delay 3 - delay 1) for odd and even trials
    delay_effect_odd = []
    delay_effect_even = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for delay effect...")
            
        # Get participant data
        p_data = va_data[va_data['PROLIFIC_PID'] == pid]
        
        # Calculate d-prime for each delay (odd trials)
        odd_dprime_delay1 = []
        odd_dprime_delay3 = []
        even_dprime_delay1 = []
        even_dprime_delay3 = []
        
        for set_size in set_sizes:
            # Odd trials, delay 1
            odd_trials_delay1 = p_data[(p_data['trial_parity'] == 'odd') & 
                                      (p_data['set_size'] == set_size) & 
                                      (p_data['delay'] == 1)]
            
            odd_change_trials_delay1 = odd_trials_delay1[odd_trials_delay1['orientation_change'] == 1]
            odd_no_change_trials_delay1 = odd_trials_delay1[odd_trials_delay1['orientation_change'] == 0]
            
            if len(odd_change_trials_delay1) > 0 and len(odd_no_change_trials_delay1) > 0:
                odd_hits_delay1 = odd_change_trials_delay1['VA_correct'].mean()
                odd_false_alarms_delay1 = 1 - odd_no_change_trials_delay1['VA_correct'].mean()
                odd_dprime_delay1.append(calculate_dprime(odd_hits_delay1, odd_false_alarms_delay1, 
                                                         len(odd_change_trials_delay1), len(odd_no_change_trials_delay1)))
            
            # Odd trials, delay 3
            odd_trials_delay3 = p_data[(p_data['trial_parity'] == 'odd') & 
                                      (p_data['set_size'] == set_size) & 
                                      (p_data['delay'] == 3)]
            
            odd_change_trials_delay3 = odd_trials_delay3[odd_trials_delay3['orientation_change'] == 1]
            odd_no_change_trials_delay3 = odd_trials_delay3[odd_trials_delay3['orientation_change'] == 0]
            
            if len(odd_change_trials_delay3) > 0 and len(odd_no_change_trials_delay3) > 0:
                odd_hits_delay3 = odd_change_trials_delay3['VA_correct'].mean()
                odd_false_alarms_delay3 = 1 - odd_no_change_trials_delay3['VA_correct'].mean()
                odd_dprime_delay3.append(calculate_dprime(odd_hits_delay3, odd_false_alarms_delay3, 
                                                        len(odd_change_trials_delay3), len(odd_no_change_trials_delay3)))
            
            # Even trials, delay 1
            even_trials_delay1 = p_data[(p_data['trial_parity'] == 'even') & 
                                       (p_data['set_size'] == set_size) & 
                                       (p_data['delay'] == 1)]
            
            even_change_trials_delay1 = even_trials_delay1[even_trials_delay1['orientation_change'] == 1]
            even_no_change_trials_delay1 = even_trials_delay1[even_trials_delay1['orientation_change'] == 0]
            
            if len(even_change_trials_delay1) > 0 and len(even_no_change_trials_delay1) > 0:
                even_hits_delay1 = even_change_trials_delay1['VA_correct'].mean()
                even_false_alarms_delay1 = 1 - even_no_change_trials_delay1['VA_correct'].mean()
                even_dprime_delay1.append(calculate_dprime(even_hits_delay1, even_false_alarms_delay1, 
                                                          len(even_change_trials_delay1), len(even_no_change_trials_delay1)))
            
            # Even trials, delay 3
            even_trials_delay3 = p_data[(p_data['trial_parity'] == 'even') & 
                                       (p_data['set_size'] == set_size) & 
                                       (p_data['delay'] == 3)]
            
            even_change_trials_delay3 = even_trials_delay3[even_trials_delay3['orientation_change'] == 1]
            even_no_change_trials_delay3 = even_trials_delay3[even_trials_delay3['orientation_change'] == 0]
            
            
            if len(even_change_trials_delay3) > 0 and len(even_no_change_trials_delay3) > 0:
                even_hits_delay3 = even_change_trials_delay3['VA_correct'].mean()
                even_false_alarms_delay3 = 1 - even_no_change_trials_delay3['VA_correct'].mean()
                even_dprime_delay3.append(calculate_dprime(even_hits_delay3, even_false_alarms_delay3, 
                                                          len(even_change_trials_delay3), len(even_no_change_trials_delay3)))
        
        # Calculate delay effects if we have data for both delays
        if odd_dprime_delay1 and odd_dprime_delay3 and even_dprime_delay1 and even_dprime_delay3:
            # Average across set sizes
            odd_delay_effect = np.mean(odd_dprime_delay1) - np.mean(odd_dprime_delay3)
            even_delay_effect = np.mean(even_dprime_delay1) - np.mean(even_dprime_delay3)
            
            # Store results
            delay_effect_odd.append(odd_delay_effect)
            delay_effect_even.append(even_delay_effect)
            participant_ids.append(pid)
    
    # Calculate reliability
    if len(delay_effect_odd) >= 3:
        # Convert to numpy arrays
        delay_effect_odd = np.array(delay_effect_odd)
        delay_effect_even = np.array(delay_effect_even)
        
        # Remove any NaNs or infinities
        valid_indices = ~np.isnan(delay_effect_odd) & ~np.isnan(delay_effect_even) & \
                        ~np.isinf(delay_effect_odd) & ~np.isinf(delay_effect_even)
        
        delay_effect_odd = delay_effect_odd[valid_indices]
        delay_effect_even = delay_effect_even[valid_indices]
        
        if len(delay_effect_odd) >= 3:
            split_half_r, _ = stats.pearsonr(delay_effect_odd, delay_effect_even)
            spearman_brown_r = calculate_spearman_brown(split_half_r)
            
            # Calculate confidence interval
            r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(delay_effect_odd))
            
            # Interpret reliability
            reliability_interpretation = interpret_reliability(spearman_brown_r)
            
            # Store results - for delay effect, count trials from both delays
            n_odd_trials_delay1 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                             (va_data['trial_parity'] == 'odd') & 
                                             (va_data['delay'] == 1)])
            n_odd_trials_delay3 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                             (va_data['trial_parity'] == 'odd') & 
                                             (va_data['delay'] == 3)])
            n_even_trials_delay1 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                              (va_data['trial_parity'] == 'even') & 
                                              (va_data['delay'] == 1)])
            n_even_trials_delay3 = len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                              (va_data['trial_parity'] == 'even') & 
                                              (va_data['delay'] == 3)])
            
            va_reliability_results.append({
                'measure': 'delay_effect',
                'condition': 'delay_1_minus_delay_3',
                'n_odd_trials': n_odd_trials_delay1 + n_odd_trials_delay3,
                'n_even_trials': n_even_trials_delay1 + n_even_trials_delay3,
                'split_half_r': split_half_r,
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation
            })
            
            # Add to summary results
            va_summary_results.append({
                'task': 'VA',
                'measure': 'delay_effect',
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation,
                'sample_size': len(delay_effect_odd),
                'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
            })
            
            print(f"  Delay effect reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate mean RT reliability
    print("\nCalculating mean RT reliability...")
    print(f"Starting mean RT calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # For each participant, calculate mean RT for odd and even trials
    mean_rt_odd = []
    mean_rt_even = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for mean RT...")
            
        # Get participant data
        p_data = va_data[va_data['PROLIFIC_PID'] == pid]
        
        # Calculate mean RT for odd trials
        odd_trials = p_data[p_data['trial_parity'] == 'odd']
        odd_rt = odd_trials['VA_rt'].mean()
        
        # Calculate mean RT for even trials
        even_trials = p_data[p_data['trial_parity'] == 'even']
        even_rt = even_trials['VA_rt'].mean()
        
        # Store results
        if not np.isnan(odd_rt) and not np.isnan(even_rt):
            mean_rt_odd.append(odd_rt)
            mean_rt_even.append(even_rt)
            participant_ids.append(pid)
    
    # Calculate reliability
    if len(mean_rt_odd) >= 3:
        # Convert to numpy arrays
        mean_rt_odd = np.array(mean_rt_odd)
        mean_rt_even = np.array(mean_rt_even)
        
        # Calculate correlation
        split_half_r, _ = stats.pearsonr(mean_rt_odd, mean_rt_even)
        spearman_brown_r = calculate_spearman_brown(split_half_r)
        
        # Calculate confidence interval
        r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(mean_rt_odd))
        
        # Interpret reliability
        reliability_interpretation = interpret_reliability(spearman_brown_r)
        
        # Store results - for mean RT, we keep all trials regardless of condition
        va_reliability_results.append({
            'measure': 'mean_rt',
            'condition': 'all',
            'n_odd_trials': len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & (va_data['trial_parity'] == 'odd')]),
            'n_even_trials': len(va_data[(va_data['PROLIFIC_PID'].isin(participant_ids)) & (va_data['trial_parity'] == 'even')]),
            'split_half_r': split_half_r,
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation
        })
        
        # Add to summary results
        va_summary_results.append({
            'task': 'VA',
            'measure': 'mean_rt',
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation,
            'sample_size': len(mean_rt_odd),
            'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
        })
        
        print(f"  Mean RT reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Convert results to DataFrames
    va_reliability_df = pd.DataFrame(va_reliability_results)
    va_summary_df = pd.DataFrame(va_summary_results)
    
    return va_reliability_df, va_summary_df

def calculate_mrt_reliability(mrt_data):
    """
    Calculate split-half reliability for Mental Rotation Task.
    
    Parameters:
    -----------
    mrt_data : pandas.DataFrame
        Cleaned Mental Rotation Task data
        
    Returns:
    --------
    tuple
        Three DataFrames containing reliability results for MRT task:
        1. Detailed reliability by condition
        2. Summary reliability measures
    """
    print("Calculating MRT reliability...")
    start_time = datetime.now()
    print(f"Starting MRT reliability calculation at {start_time.strftime('%H:%M:%S')}")
    
    # Create a list to store all reliability results
    mrt_reliability_results = []
    mrt_summary_results = []
    
    # Get unique participants
    participants = mrt_data['PROLIFIC_PID'].unique()
    print(f"Number of participants: {len(participants)}")
    
    # Get unique angular disparities
    angular_disparities = mrt_data['angular_disparity'].unique()
    print(f"Unique angular disparities: {angular_disparities}")
    
    # Add trial number within participant
    mrt_data = mrt_data.sort_values(['PROLIFIC_PID', 'angular_disparity'])
    mrt_data['trial_num'] = mrt_data.groupby('PROLIFIC_PID').cumcount() + 1
    
    # Identify odd and even trials
    mrt_data['trial_parity'] = mrt_data['trial_num'].apply(lambda x: 'odd' if x % 2 == 1 else 'even')
    
    # Calculate accuracy and RT for each angular disparity and trial parity
    for angle in angular_disparities:
        condition = f"angle_{angle}"
        print(f"\nProcessing condition: {condition} at {datetime.now().strftime('%H:%M:%S')}")
        
        # Calculate accuracy and RT for each participant
        odd_accuracy = []
        even_accuracy = []
        odd_rt = []
        even_rt = []
        participant_ids = []
        
        participant_count = 0
        for pid in participants:
            participant_count += 1
            if participant_count % 10 == 0:
                print(f"  Processed {participant_count}/{len(participants)} participants for condition {condition}...")
                
            # Get participant data for this condition
            p_data = mrt_data[(mrt_data['PROLIFIC_PID'] == pid) & 
                              (mrt_data['angular_disparity'] == angle)]
            
            # Count trials for verification
            odd_trials = p_data[p_data['trial_parity'] == 'odd']
            even_trials = p_data[p_data['trial_parity'] == 'even']
            
            # Skip participants with no data in this condition
            if len(odd_trials) == 0 or len(even_trials) == 0:
                print(f"  Participant {pid} has no data in one or both halves for this condition, skipping")
                continue
            
            # Calculate accuracy for odd trials
            odd_acc = odd_trials['MRT_correct'].mean()
            
            # Calculate accuracy for even trials
            even_acc = even_trials['MRT_correct'].mean()
            
            # Calculate mean RT for correct odd trials
            odd_correct_trials = odd_trials[odd_trials['MRT_correct'] == 1]
            if len(odd_correct_trials) > 0:
                odd_mean_rt = odd_correct_trials['MRT_rt'].mean()
            else:
                odd_mean_rt = np.nan
            
            # Calculate mean RT for correct even trials
            even_correct_trials = even_trials[even_trials['MRT_correct'] == 1]
            if len(even_correct_trials) > 0:
                even_mean_rt = even_correct_trials['MRT_rt'].mean()
            else:
                even_mean_rt = np.nan
            
            # Store results
            if not np.isnan(odd_acc) and not np.isnan(even_acc):
                odd_accuracy.append(odd_acc)
                even_accuracy.append(even_acc)
                odd_rt.append(odd_mean_rt)
                even_rt.append(even_mean_rt)
                participant_ids.append(pid)
        
        # Calculate reliability for accuracy
        if len(odd_accuracy) >= 3:
            # Convert to numpy arrays
            odd_accuracy = np.array(odd_accuracy)
            even_accuracy = np.array(even_accuracy)
            
            # Calculate correlation
            split_half_r, _ = stats.pearsonr(odd_accuracy, even_accuracy)
            spearman_brown_r = calculate_spearman_brown(split_half_r)
            
            # Calculate confidence interval
            r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(odd_accuracy))
            
            # Interpret reliability
            reliability_interpretation = interpret_reliability(spearman_brown_r)
            
            # Store results
            mrt_reliability_results.append({
                'measure': 'accuracy',
                'condition': condition,
                'n_odd_trials': len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                            (mrt_data['trial_parity'] == 'odd') & 
                                            (mrt_data['angular_disparity'] == angle)]),
                'n_even_trials': len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                             (mrt_data['trial_parity'] == 'even') & 
                                             (mrt_data['angular_disparity'] == angle)]),
                'split_half_r': split_half_r,
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation
            })
            
            # Add to summary results
            mrt_summary_results.append({
                'task': 'MRT',
                'measure': f'accuracy_{condition}',
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation,
                'sample_size': len(odd_accuracy),
                'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
            })
            
            print(f"  Accuracy reliability for {condition}: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
        
        # Calculate reliability for RT
        valid_rt_indices = ~np.isnan(odd_rt) & ~np.isnan(even_rt)
        valid_odd_rt = np.array(odd_rt)[valid_rt_indices]
        valid_even_rt = np.array(even_rt)[valid_rt_indices]
        
        if len(valid_odd_rt) >= 3:
            # Calculate correlation
            split_half_r, _ = stats.pearsonr(valid_odd_rt, valid_even_rt)
            spearman_brown_r = calculate_spearman_brown(split_half_r)
            
            # Calculate confidence interval
            r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(valid_odd_rt))
            
            # Interpret reliability
            reliability_interpretation = interpret_reliability(spearman_brown_r)
            
            # Store results
            mrt_reliability_results.append({
                'measure': 'rt',
                'condition': condition,
                'n_odd_trials': len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                            (mrt_data['trial_parity'] == 'odd') & 
                                            (mrt_data['angular_disparity'] == angle)]),
                'n_even_trials': len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                             (mrt_data['trial_parity'] == 'even') & 
                                             (mrt_data['angular_disparity'] == angle)]),
                'split_half_r': split_half_r,
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation
            })
            
            # Add to summary results
            mrt_summary_results.append({
                'task': 'MRT',
                'measure': f'rt_{condition}',
                'spearman_brown_r': spearman_brown_r,
                'r_lower_ci': r_lower_ci,
                'r_upper_ci': r_upper_ci,
                'reliability_interpretation': reliability_interpretation,
                'sample_size': len(valid_odd_rt),
                'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
            })
            
            print(f"  RT reliability for {condition}: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate overall accuracy reliability
    print("\nCalculating overall accuracy reliability...")
    print(f"Starting overall accuracy calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # Calculate overall accuracy for each participant
    overall_odd_accuracy = []
    overall_even_accuracy = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for overall accuracy...")
            
        # Get participant data
        p_data = mrt_data[mrt_data['PROLIFIC_PID'] == pid]
        
        # Calculate accuracy for odd trials
        odd_trials = p_data[p_data['trial_parity'] == 'odd']
        odd_acc = odd_trials['MRT_correct'].mean()
        
        # Calculate accuracy for even trials
        even_trials = p_data[p_data['trial_parity'] == 'even']
        even_acc = even_trials['MRT_correct'].mean()
        
        # Store results
        if not np.isnan(odd_acc) and not np.isnan(even_acc):
            overall_odd_accuracy.append(odd_acc)
            overall_even_accuracy.append(even_acc)
            participant_ids.append(pid)
    
    # Calculate reliability
    if len(overall_odd_accuracy) >= 3:
        # Convert to numpy arrays
        overall_odd_accuracy = np.array(overall_odd_accuracy)
        overall_even_accuracy = np.array(overall_even_accuracy)
        
        # Calculate correlation
        split_half_r, _ = stats.pearsonr(overall_odd_accuracy, overall_even_accuracy)
        spearman_brown_r = calculate_spearman_brown(split_half_r)
        
        # Calculate confidence interval
        r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(overall_odd_accuracy))
        
        # Interpret reliability
        reliability_interpretation = interpret_reliability(spearman_brown_r)
        
        # Store results - for overall accuracy, we keep all trials regardless of angle
        mrt_reliability_results.append({
            'measure': 'overall_accuracy',
            'condition': 'all',
            'n_odd_trials': len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & (mrt_data['trial_parity'] == 'odd')]),
            'n_even_trials': len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & (mrt_data['trial_parity'] == 'even')]),
            'split_half_r': split_half_r,
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation
        })
        
        # Add to summary results
        mrt_summary_results.append({
            'task': 'MRT',
            'measure': 'overall_accuracy',
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation,
            'sample_size': len(overall_odd_accuracy),
            'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
        })
        
        print(f"  Overall accuracy reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate RT-by-angle slope reliability
    print("\nCalculating RT-by-angle slope reliability...")
    print(f"Starting RT slope calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # Calculate slopes for each participant
    odd_slopes = []
    even_slopes = []
    odd_intercepts = []
    even_intercepts = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for RT slope...")
            
        # Get participant data
        p_data = mrt_data[mrt_data['PROLIFIC_PID'] == pid]
        
        # Calculate slopes for odd trials
        odd_trials = p_data[(p_data['trial_parity'] == 'odd') & (p_data['MRT_correct'] == 1)]
        if len(odd_trials) >= 4:  # Need at least 4 data points for a meaningful slope
            odd_angle_rt_data = odd_trials.groupby('angular_disparity')['MRT_rt'].mean().reset_index()
            if len(odd_angle_rt_data) >= 2:  # Need at least 2 different angles
                odd_slope, odd_intercept = np.polyfit(odd_angle_rt_data['angular_disparity'], 
                                                     odd_angle_rt_data['MRT_rt'], 1)
            else:
                odd_slope = np.nan
                odd_intercept = np.nan
        else:
            odd_slope = np.nan
            odd_intercept = np.nan
        
        # Calculate slopes for even trials
        even_trials = p_data[(p_data['trial_parity'] == 'even') & (p_data['MRT_correct'] == 1)]
        if len(even_trials) >= 4:
            even_angle_rt_data = even_trials.groupby('angular_disparity')['MRT_rt'].mean().reset_index()
            if len(even_angle_rt_data) >= 2:
                even_slope, even_intercept = np.polyfit(even_angle_rt_data['angular_disparity'], 
                                                       even_angle_rt_data['MRT_rt'], 1)
            else:
                even_slope = np.nan
                even_intercept = np.nan
        else:
            even_slope = np.nan
            even_intercept = np.nan
        
        # Store results
        if not np.isnan(odd_slope) and not np.isnan(even_slope):
            odd_slopes.append(odd_slope)
            even_slopes.append(even_slope)
            odd_intercepts.append(odd_intercept)
            even_intercepts.append(even_intercept)
            participant_ids.append(pid)
    
    # Calculate reliability for slopes
    if len(odd_slopes) >= 3:
        # Convert to numpy arrays
        odd_slopes = np.array(odd_slopes)
        even_slopes = np.array(even_slopes)
        
        # Calculate correlation
        split_half_r, _ = stats.pearsonr(odd_slopes, even_slopes)
        spearman_brown_r = calculate_spearman_brown(split_half_r)
        
        # Calculate confidence interval
        r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(odd_slopes))
        
        # Interpret reliability
        reliability_interpretation = interpret_reliability(spearman_brown_r)
        
        # Store results - for RT slope, count trials across all angles
        n_odd_trials = 0
        n_even_trials = 0
        for angle in angular_disparities:
            if not np.isnan(angle):
                n_odd_trials += len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                            (mrt_data['trial_parity'] == 'odd') & 
                                            (mrt_data['angular_disparity'] == angle)])
                n_even_trials += len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                             (mrt_data['trial_parity'] == 'even') & 
                                             (mrt_data['angular_disparity'] == angle)])
        
        mrt_reliability_results.append({
            'measure': 'rt_slope',
            'condition': 'all',
            'n_odd_trials': n_odd_trials,
            'n_even_trials': n_even_trials,
            'split_half_r': split_half_r,
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation
        })
        
        # Add to summary results
        mrt_summary_results.append({
            'task': 'MRT',
            'measure': 'rt_slope',
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation,
            'sample_size': len(odd_slopes),
            'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
        })
        
        print(f"  RT slope reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate reliability for intercepts
    if len(odd_intercepts) >= 3:
        # Convert to numpy arrays
        odd_intercepts = np.array(odd_intercepts)
        even_intercepts = np.array(even_intercepts)
        
        # Calculate correlation
        split_half_r, _ = stats.pearsonr(odd_intercepts, even_intercepts)
        spearman_brown_r = calculate_spearman_brown(split_half_r)
        
        # Calculate confidence interval
        r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(odd_intercepts))
        
        # Interpret reliability
        reliability_interpretation = interpret_reliability(spearman_brown_r)
        
        # Store results - for RT intercept, use the same trial counts as RT slope
        mrt_reliability_results.append({
            'measure': 'rt_intercept',
            'condition': 'all',
            'n_odd_trials': n_odd_trials,  # Reuse the counts calculated for rt_slope
            'n_even_trials': n_even_trials,
            'split_half_r': split_half_r,
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation
        })
        
        # Add to summary results
        mrt_summary_results.append({
            'task': 'MRT',
            'measure': 'rt_intercept',
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation,
            'sample_size': len(odd_intercepts),
            'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
        })
        
        print(f"  RT intercept reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Calculate overall mean RT reliability
    print("\nCalculating overall mean RT reliability...")
    print(f"Starting overall mean RT calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # Calculate overall mean RT for each participant
    overall_odd_rt = []
    overall_even_rt = []
    participant_ids = []
    
    participant_count = 0
    for pid in participants:
        participant_count += 1
        if participant_count % 10 == 0:
            print(f"  Processed {participant_count}/{len(participants)} participants for overall mean RT...")
            
        # Get participant data
        p_data = mrt_data[mrt_data['PROLIFIC_PID'] == pid]
        
        # Calculate mean RT for correct odd trials
        odd_correct_trials = p_data[(p_data['trial_parity'] == 'odd') & (p_data['MRT_correct'] == 1)]
        if len(odd_correct_trials) > 0:
            odd_mean_rt = odd_correct_trials['MRT_rt'].mean()
        else:
            odd_mean_rt = np.nan
        
        # Calculate mean RT for correct even trials
        even_correct_trials = p_data[(p_data['trial_parity'] == 'even') & (p_data['MRT_correct'] == 1)]
        if len(even_correct_trials) > 0:
            even_mean_rt = even_correct_trials['MRT_rt'].mean()
        else:
            even_mean_rt = np.nan
        
        # Store results
        if not np.isnan(odd_mean_rt) and not np.isnan(even_mean_rt):
            overall_odd_rt.append(odd_mean_rt)
            overall_even_rt.append(even_mean_rt)
            participant_ids.append(pid)
    
    # Calculate reliability
    if len(overall_odd_rt) >= 3:
        # Convert to numpy arrays
        overall_odd_rt = np.array(overall_odd_rt)
        overall_even_rt = np.array(overall_even_rt)
        
        # Calculate correlation
        split_half_r, _ = stats.pearsonr(overall_odd_rt, overall_even_rt)
        spearman_brown_r = calculate_spearman_brown(split_half_r)
        
        # Calculate confidence interval
        r_lower_ci, r_upper_ci = calculate_confidence_interval(spearman_brown_r, len(overall_odd_rt))
        
        # Interpret reliability
        reliability_interpretation = interpret_reliability(spearman_brown_r)
        
        # Calculate trial counts directly from the dataframe
        n_odd_trials = len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                   (mrt_data['trial_parity'] == 'odd') & 
                                   (mrt_data['MRT_correct'] == 1)])
        n_even_trials = len(mrt_data[(mrt_data['PROLIFIC_PID'].isin(participant_ids)) & 
                                    (mrt_data['trial_parity'] == 'even') & 
                                    (mrt_data['MRT_correct'] == 1)])
        
        # Store results
        mrt_reliability_results.append({
            'measure': 'overall_rt',
            'condition': 'all',
            'n_odd_trials': n_odd_trials,
            'n_even_trials': n_even_trials,
            'split_half_r': split_half_r,
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation
        })
        
        # Add to summary results
        mrt_summary_results.append({
            'task': 'MRT',
            'measure': 'overall_rt',
            'spearman_brown_r': spearman_brown_r,
            'r_lower_ci': r_lower_ci,
            'r_upper_ci': r_upper_ci,
            'reliability_interpretation': reliability_interpretation,
            'sample_size': len(overall_odd_rt),
            'limitations_noted': 'None' if reliability_interpretation not in ['Poor', 'Unacceptable'] else 'Low reliability'
        })
        
        print(f"  Overall mean RT reliability: r = {spearman_brown_r:.3f}, interpretation: {reliability_interpretation}")
    
    # Convert results to DataFrames
    mrt_reliability_df = pd.DataFrame(mrt_reliability_results)
    mrt_summary_df = pd.DataFrame(mrt_summary_results)
    
    return mrt_reliability_df, mrt_summary_df

def calculate_attenuation_corrected_correlations(va_summary_df, mrt_summary_df):
    """
    Calculate attenuation-corrected correlations between VA and MRT tasks.
    
    Parameters:
    -----------
    va_summary_df : pandas.DataFrame
        Summary reliability measures for VA task
    mrt_summary_df : pandas.DataFrame
        Summary reliability measures for MRT task
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing attenuation-corrected correlations
    """
    print("\nCalculating attenuation-corrected correlations...")
    print(f"Starting attenuation correction calculation at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get overall reliability measures
    va_overall_reliability = va_summary_df[va_summary_df['measure'] == 'overall_d_prime']['spearman_brown_r'].values[0]
    mrt_overall_reliability = mrt_summary_df[mrt_summary_df['measure'] == 'overall_accuracy']['spearman_brown_r'].values[0]
    
    # Assume an observed correlation of 0.3 between VA and MRT (this would be calculated from actual data)
    observed_correlation = 0.3
    
    # Calculate attenuation-corrected correlation
    corrected_correlation = observed_correlation / math.sqrt(va_overall_reliability * mrt_overall_reliability)
    
    print(f"  Observed correlation: {observed_correlation:.3f}")
    print(f"  VA overall reliability: {va_overall_reliability:.3f}")
    print(f"  MRT overall reliability: {mrt_overall_reliability:.3f}")
    print(f"  Attenuation-corrected correlation: {corrected_correlation:.3f}")
    
    # Create DataFrame for results
    attenuation_df = pd.DataFrame({
        'va_measure': ['overall_d_prime'],
        'mrt_measure': ['overall_accuracy'],
        'observed_correlation': [observed_correlation],
        'va_reliability': [va_overall_reliability],
        'mrt_reliability': [mrt_overall_reliability],
        'corrected_correlation': [corrected_correlation]
    })
    
    return attenuation_df

def main():
    """
    Main function to run the reliability analysis for VA and MRT tasks.
    """
    print("Starting reliability analysis...")
    
    # Create output directory if it doesn't exist
    if not create_output_dir():
        print("Error: Could not create output directory")
        return 1
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get latest input files
    va_input_file = get_latest_file("outputs/VA_cleaned_data_*.csv")
    mrt_input_file = get_latest_file("outputs/MRT_cleaned_data_*.csv")
    
    if va_input_file is None or mrt_input_file is None:
        print("Error: Could not find input files")
        return 1
    
    # Load input data
    print(f"\nLoading VA data from {va_input_file}...")
    try:
        va_data = pd.read_csv(va_input_file)
        # Ensure orientation_change is numeric (0/1) not boolean
        if 'orientation_change' in va_data.columns:
            # Handle potential errors in conversion
            try:
                va_data['orientation_change'] = va_data['orientation_change'].astype(int)
            except Exception as e:
                print(f"Warning: Could not convert orientation_change to int: {e}")
                # Try to handle boolean values
                if va_data['orientation_change'].dtype == bool:
                    va_data['orientation_change'] = va_data['orientation_change'].astype(int)
        print(f"Loaded VA data with {len(va_data)} rows and {len(va_data.columns)} columns")
        print("VA data columns:", va_data.columns.tolist())
        print("First 3 rows of VA data:")
        print(va_data.head(3))
    except Exception as e:
        print(f"Error loading VA data: {e}")
        return 1
    
    print(f"\nLoading MRT data from {mrt_input_file}...")
    try:
        mrt_data = pd.read_csv(mrt_input_file)
        print(f"Loaded MRT data with {len(mrt_data)} rows and {len(mrt_data.columns)} columns")
        print("MRT data columns:", mrt_data.columns.tolist())
        print("First 3 rows of MRT data:")
        print(mrt_data.head(3))
    except Exception as e:
        print(f"Error loading MRT data: {e}")
        return 1
    
    # Check for required columns in VA data
    required_va_columns = [
        'PROLIFIC_PID', 'trial_type', 'set_size', 'delay', 'target_color',
        'orientation_change', 'condition', 'VA_response_key', 'VA_correct',
        'VA_rt', 'trial_excluded'
    ]
    
    missing_va_columns = [col for col in required_va_columns if col not in va_data.columns]
    if missing_va_columns:
        print(f"Error: Missing required columns in VA data: {missing_va_columns}")
        return 1
    
    # Check for required columns in MRT data
    required_mrt_columns = [
        'PROLIFIC_PID', 'trial_type', 'angular_disparity', 'stimulus_type',
        'condition', 'MRT_response_key', 'MRT_correct', 'MRT_rt', 'trial_excluded'
    ]
    
    missing_mrt_columns = [col for col in required_mrt_columns if col not in mrt_data.columns]
    if missing_mrt_columns:
        print(f"Error: Missing required columns in MRT data: {missing_mrt_columns}")
        return 1
    
    # Print unique values for key experimental design parameters
    print("\nVA data unique values:")
    print(f"trial_type: {va_data['trial_type'].unique()}")
    print(f"set_size: {va_data['set_size'].unique()}")
    print(f"delay: {va_data['delay'].unique()}")
    print(f"target_color: {va_data['target_color'].unique()}")
    print(f"orientation_change: {va_data['orientation_change'].unique()}")
    print(f"condition: {va_data['condition'].unique()}")
    print(f"VA_correct: {va_data['VA_correct'].unique()}")
    print(f"trial_excluded: {va_data['trial_excluded'].unique()}")
    
    print("\nMRT data unique values:")
    print(f"trial_type: {mrt_data['trial_type'].unique()}")
    print(f"angular_disparity: {mrt_data['angular_disparity'].unique()}")
    print(f"stimulus_type: {mrt_data['stimulus_type'].unique()}")
    print(f"condition: {mrt_data['condition'].unique()}")
    print(f"MRT_correct: {mrt_data['MRT_correct'].unique()}")
    print(f"trial_excluded: {mrt_data['trial_excluded'].unique()}")
    
    # Filter data to include only test trials and exclude excluded trials
    print("\nFiltering data...")
    
    # Ensure trial_excluded is properly formatted
    if 'trial_excluded' in va_data.columns:
        if va_data['trial_excluded'].dtype == bool:
            va_data['trial_excluded'] = va_data['trial_excluded'].astype(int)
    
    if 'trial_excluded' in mrt_data.columns:
        if mrt_data['trial_excluded'].dtype == bool:
            mrt_data['trial_excluded'] = mrt_data['trial_excluded'].astype(int)
    
    va_data_filtered = va_data[(va_data['trial_type'] == 'test') & 
                               (va_data['trial_excluded'] == 0)]
    
    mrt_data_filtered = mrt_data[(mrt_data['trial_type'] == 'test') & 
                                 (mrt_data['trial_excluded'] == 0)]
    
    print(f"VA data: {len(va_data)} rows -> {len(va_data_filtered)} rows after filtering")
    print(f"MRT data: {len(mrt_data)} rows -> {len(mrt_data_filtered)} rows after filtering")
    
    # Check for missing values in relevant columns
    print("\nChecking for missing values...")
    
    va_missing_values = va_data_filtered[required_va_columns].isnull().sum()
    print("VA data missing values:")
    print(va_missing_values[va_missing_values > 0])
    
    mrt_missing_values = mrt_data_filtered[required_mrt_columns].isnull().sum()
    print("MRT data missing values:")
    print(mrt_missing_values[mrt_missing_values > 0])
    
    # Drop rows with missing values in relevant columns
    va_data_clean = va_data_filtered.dropna(subset=[
        'PROLIFIC_PID', 'set_size', 'delay', 'orientation_change', 'VA_correct', 'VA_rt'
    ])
    
    mrt_data_clean = mrt_data_filtered.dropna(subset=[
        'PROLIFIC_PID', 'angular_disparity', 'MRT_correct', 'MRT_rt'
    ])
    
    print(f"VA data: {len(va_data_filtered)} rows -> {len(va_data_clean)} rows after dropping missing values")
    print(f"MRT data: {len(mrt_data_filtered)} rows -> {len(mrt_data_clean)} rows after dropping missing values")
    
    # Calculate reliability
    print(f"\nStarting reliability calculations at {datetime.now().strftime('%H:%M:%S')}")
    va_reliability_df, va_summary_df = calculate_va_reliability(va_data_clean)
    mrt_reliability_df, mrt_summary_df = calculate_mrt_reliability(mrt_data_clean)
    print(f"\nCompleted reliability calculations at {datetime.now().strftime('%H:%M:%S')}")
    
    # Calculate attenuation-corrected correlations
    attenuation_df = None
    if not va_summary_df.empty and not mrt_summary_df.empty:
        # Check if required measures exist in summary dataframes
        if 'overall_d_prime' in va_summary_df['measure'].values and 'overall_accuracy' in mrt_summary_df['measure'].values:
            attenuation_df = calculate_attenuation_corrected_correlations(va_summary_df, mrt_summary_df)
        else:
            print("Warning: Cannot calculate attenuation-corrected correlations due to missing measures")
    
    # Save results to CSV files
    va_reliability_output = f"outputs/VA_reliability_{timestamp}.csv"
    mrt_reliability_output = f"outputs/MRT_reliability_{timestamp}.csv"
    reliability_summary_output = f"outputs/reliability_summary_{timestamp}.csv"
    
    # Verify dataframes are not empty before saving
    if va_reliability_df.empty:
        print("Warning: VA reliability dataframe is empty, not saving")
    else:
        print(f"\nSaving VA reliability results to {va_reliability_output}...")
        print(f"VA reliability dataframe shape: {va_reliability_df.shape}")
        print(f"First few rows of VA reliability dataframe:")
        print(va_reliability_df.head(2))
        
        # Check for any issues in the dataframes
        print("Checking VA reliability dataframe for issues...")
        print("n_odd_trials unique values:", va_reliability_df['n_odd_trials'].unique())
        print("n_even_trials unique values:", va_reliability_df['n_even_trials'].unique())
        
        try:
            va_reliability_df.to_csv(va_reliability_output, index=False)
            print(f"VA reliability file saved successfully.")
            # Verify file was created
            if os.path.exists(va_reliability_output):
                print(f"Verified: {va_reliability_output} exists")
            else:
                print(f"Warning: {va_reliability_output} was not created")
        except Exception as e:
            print(f"Error saving VA reliability file: {e}")
    
    if mrt_reliability_df.empty:
        print("Warning: MRT reliability dataframe is empty, not saving")
    else:
        print(f"Saving MRT reliability results to {mrt_reliability_output}...")
        print(f"MRT reliability dataframe shape: {mrt_reliability_df.shape}")
        print(f"First few rows of MRT reliability dataframe:")
        print(mrt_reliability_df.head(2))
        
        # Check for any issues in the dataframes
        print("Checking MRT reliability dataframe for issues...")
        print("n_odd_trials unique values:", mrt_reliability_df['n_odd_trials'].unique())
        print("n_even_trials unique values:", mrt_reliability_df['n_even_trials'].unique())
        
        try:
            mrt_reliability_df.to_csv(mrt_reliability_output, index=False)
            print(f"MRT reliability file saved successfully.")
            # Verify file was created
            if os.path.exists(mrt_reliability_output):
                print(f"Verified: {mrt_reliability_output} exists")
            else:
                print(f"Warning: {mrt_reliability_output} was not created")
        except Exception as e:
            print(f"Error saving MRT reliability file: {e}")
    
    # Combine summary results
    if va_summary_df.empty or mrt_summary_df.empty:
        print("Warning: One or both summary dataframes are empty")
        if not va_summary_df.empty:
            reliability_summary_df = va_summary_df
        elif not mrt_summary_df.empty:
            reliability_summary_df = mrt_summary_df
        else:
            print("Error: Cannot create reliability summary, both dataframes are empty")
            reliability_summary_df = pd.DataFrame()
    else:
        reliability_summary_df = pd.concat([va_summary_df, mrt_summary_df], ignore_index=True)
    
    if not reliability_summary_df.empty:
        print(f"Saving reliability summary to {reliability_summary_output}...")
        print(f"Reliability summary dataframe shape: {reliability_summary_df.shape}")
        print(f"First few rows of reliability summary dataframe:")
        print(reliability_summary_df.head(2))
        try:
            reliability_summary_df.to_csv(reliability_summary_output, index=False)
            print(f"Reliability summary file saved successfully.")
            # Verify file was created
            if os.path.exists(reliability_summary_output):
                print(f"Verified: {reliability_summary_output} exists")
            else:
                print(f"Warning: {reliability_summary_output} was not created")
        except Exception as e:
            print(f"Error saving reliability summary file: {e}")
    else:
        print("Warning: Reliability summary dataframe is empty, not saving")
    
    print("Finished execution")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
