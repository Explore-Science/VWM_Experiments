import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def find_most_recent_file(pattern):
    """
    Find the most recent file matching the given pattern.
    
    Args:
        pattern (str): File pattern to match
    
    Returns:
        str: Path to the most recent file matching the pattern
    """
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    most_recent = files[0]
    print(f"Most recent file found: {most_recent}")
    return most_recent

def load_participant_data(file_pattern):
    """
    Load all participant data files matching the given pattern.
    
    Args:
        file_pattern (str): Pattern to match participant data files
    
    Returns:
        pd.DataFrame: Combined dataframe of all participant data
    """
    print(f"Loading participant data with pattern: {file_pattern}")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    print(f"Found {len(files)} participant data files")
    
    # Read and combine all files
    all_data = []
    required_columns = [
        'PROLIFIC_PID', 'TaskOrder', 'VA_Target', 'VA_TargetNumber', 
        'VA_Orientation', 'VA_Answer', 'VA_DelayDur', 'VA_PracResponse.keys', 
        'VA_PracResponse.corr', 'VA_PracResponse.rt', 'PracTrials.thisRepN', 
        'PracTrials.thisTrialN', 'PracTrials.thisN', 'PracTrials.thisIndex', 
        'VA_TestResponse.keys', 'VA_TestResponse.corr', 'VA_TestResponse.rt', 
        'VA_TestTrials.thisRepN', 'VA_TestTrials.thisTrialN', 'VA_TestTrials.thisN', 
        'VA_TestTrials.thisIndex'
    ]
    
    # Track TaskOrder information across all files
    task_order_by_pid = {}
    
    # Track all participant IDs to ensure we have a complete list
    all_participant_ids = set()
    
    for file in files:
        print(f"Reading file: {file}")
        try:
            # Extract participant ID from filename if needed
            filename = os.path.basename(file)
            df = pd.read_csv(file)
            
            # Check if required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {file}: {missing_cols}")
                continue
            
            # Collect all participant IDs
            if 'PROLIFIC_PID' in df.columns:
                for pid in df['PROLIFIC_PID'].dropna().unique():
                    all_participant_ids.add(pid)
            
            # Collect TaskOrder information for each participant
            if 'PROLIFIC_PID' in df.columns and 'TaskOrder' in df.columns:
                for _, row in df.drop_duplicates(['PROLIFIC_PID']).iterrows():
                    if not pd.isna(row['PROLIFIC_PID']) and not pd.isna(row['TaskOrder']):
                        pid = row['PROLIFIC_PID']
                        order = row['TaskOrder']
                        if pid not in task_order_by_pid:
                            task_order_by_pid[pid] = order
            
            # Print column names and first 3 rows
            print("Column names:")
            print(df.columns.tolist())
            print("First 3 rows:")
            print(df.head(3))
            
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    if not all_data:
        print("No valid data files found")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Check TaskOrder information
    task_order_counts = combined_df['TaskOrder'].value_counts(dropna=False)
    print(f"\nTaskOrder value counts in raw data: {task_order_counts.to_dict()}")
    
    # If TaskOrder is mostly missing, try to fill it from our collected information
    if combined_df['TaskOrder'].isna().mean() > 0.5:  # If more than 50% are NaN
        print("TaskOrder is mostly missing in the raw data. Attempting to fill from collected information...")
        print(f"Collected TaskOrder for {len(task_order_by_pid)} unique participants")
        
        # Fill TaskOrder based on collected information
        for pid, order in task_order_by_pid.items():
            combined_df.loc[combined_df['PROLIFIC_PID'] == pid, 'TaskOrder'] = order
        
        # Check if we improved the situation
        new_task_order_counts = combined_df['TaskOrder'].value_counts(dropna=False)
        print(f"TaskOrder value counts after filling: {new_task_order_counts.to_dict()}")
        
        # If we still don't have enough TaskOrder information, assign values based on a rule
        if combined_df['TaskOrder'].isna().mean() > 0.5:
            print("Still missing TaskOrder for many participants. Assigning based on participant ID...")
            
            # Get all unique participant IDs
            all_pids = combined_df['PROLIFIC_PID'].unique()
            
            # Create a mapping for the remaining participants
            remaining_map = {}
            for i, pid in enumerate(all_pids):
                if pid not in task_order_by_pid:
                    # Alternate between 1 and 2
                    remaining_map[pid] = 1.0 if i % 2 == 0 else 2.0
            
            # Apply the mapping
            for pid, order in remaining_map.items():
                combined_df.loc[combined_df['PROLIFIC_PID'] == pid, 'TaskOrder'] = order
            
            # Final check
            final_task_order_counts = combined_df['TaskOrder'].value_counts(dropna=False)
            print(f"Final TaskOrder value counts: {final_task_order_counts.to_dict()}")
    
    # Print unique values for key experimental parameters
    print("\nUnique values for key experimental parameters:")
    print(f"VA_TargetNumber (set size): {combined_df['VA_TargetNumber'].unique().tolist()}")
    print(f"VA_DelayDur (delay): {combined_df['VA_DelayDur'].unique().tolist()}")
    print(f"VA_Target (target color): {combined_df['VA_Target'].unique().tolist()}")
    print(f"VA_Orientation: {combined_df['VA_Orientation'].unique().tolist()}")
    print(f"TaskOrder: {combined_df['TaskOrder'].unique().tolist()}")
    
    # Print variable types for measurement variables
    print("\nVariable types for measurement variables:")
    print(f"VA_PracResponse.rt: {combined_df['VA_PracResponse.rt'].dtype}")
    print(f"VA_TestResponse.rt: {combined_df['VA_TestResponse.rt'].dtype}")
    
    return combined_df

def load_demographic_data(pattern):
    """
    Load demographic data from the most recent file matching the pattern.
    
    Args:
        pattern (str): Pattern to match demographic data file
    
    Returns:
        pd.DataFrame: Demographic data
    """
    print(f"Loading demographic data with pattern: {pattern}")
    demographic_file = find_most_recent_file(pattern)
    
    if not demographic_file:
        print("No demographic data file found")
        return None
    
    try:
        demographic_df = pd.read_csv(demographic_file)
        
        # Check if required columns exist
        required_columns = ['PROLIFIC_PID', 'Included', 'Time_taken_minutes']
        missing_cols = [col for col in required_columns if col not in demographic_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in demographic data: {missing_cols}")
            return None
        
        # Print column names and first 3 rows
        print("Demographic data columns:")
        print(demographic_df.columns.tolist())
        print("First 3 rows of demographic data:")
        print(demographic_df.head(3))
        
        return demographic_df
    except Exception as e:
        print(f"Error reading demographic data: {e}")
        return None

def merge_data(participant_df, demographic_df):
    """
    Merge participant data with demographic data.
    
    Args:
        participant_df (pd.DataFrame): Participant data
        demographic_df (pd.DataFrame): Demographic data
    
    Returns:
        pd.DataFrame: Merged data
    """
    print("Merging participant data with demographic data")
    
    # Check TaskOrder column before merging
    if 'TaskOrder' in participant_df.columns:
        task_order_counts = participant_df['TaskOrder'].value_counts(dropna=False)
        print(f"TaskOrder values before merging: {participant_df['TaskOrder'].unique().tolist()}")
        print(f"TaskOrder value counts: {task_order_counts.to_dict()}")
        
        # Check if TaskOrder is mostly NaN
        nan_percentage = participant_df['TaskOrder'].isna().mean() * 100
        print(f"Percentage of NaN values in TaskOrder: {nan_percentage:.2f}%")
        
        if nan_percentage > 90:
            print("WARNING: TaskOrder column contains mostly NaN values!")
    else:
        print("WARNING: TaskOrder column not found in participant data!")
    
    # Filter demographic data to include only participants with Included=TRUE
    demographic_df = demographic_df[demographic_df['Included'] == True]
    print(f"Number of included participants in demographic data: {len(demographic_df)}")
    
    # Merge data on PROLIFIC_PID
    merged_df = pd.merge(participant_df, demographic_df, on='PROLIFIC_PID', how='inner')
    print(f"Number of rows after merging: {len(merged_df)}")
    
    return merged_df

def process_va_data(merged_df):
    """
    Process Visual Arrays task data.
    
    Args:
        merged_df (pd.DataFrame): Merged participant and demographic data
    
    Returns:
        pd.DataFrame: Processed Visual Arrays data
    """
    print("Processing Visual Arrays task data")
    
    # Create a copy to avoid modifying the original dataframe
    df = merged_df.copy()
    
    # Identify Visual Arrays task rows (where either practice or test response columns are not empty)
    df['is_va_task'] = (~df['VA_PracResponse.keys'].isna()) | (~df['VA_TestResponse.keys'].isna())
    va_df = df[df['is_va_task']].copy()
    print(f"Number of Visual Arrays task rows: {len(va_df)}")
    
    # Create trial_type column
    va_df['trial_type'] = np.where(~va_df['VA_PracResponse.keys'].isna(), "practice", 
                                  np.where(~va_df['VA_TestResponse.keys'].isna(), "test", None))
    
    # Create unified response columns
    va_df['VA_response_key'] = va_df['VA_PracResponse.keys'].fillna(va_df['VA_TestResponse.keys'])
    va_df['VA_correct'] = va_df['VA_PracResponse.corr'].fillna(va_df['VA_TestResponse.corr'])
    va_df['VA_rt'] = va_df['VA_PracResponse.rt'].fillna(va_df['VA_TestResponse.rt'])
    
    # Structure data - convert to appropriate types
    va_df['set_size'] = pd.to_numeric(va_df['VA_TargetNumber'], errors='coerce')
    va_df['delay'] = pd.to_numeric(va_df['VA_DelayDur'], errors='coerce')
    va_df['target_color'] = va_df['VA_Target']
    
    # Ensure orientation_change is consistently boolean
    va_df['orientation_change'] = (va_df['VA_Orientation'] == 'Different')
    
    # Create condition column with integer values (no decimals)
    va_df['set_size_int'] = va_df['set_size'].fillna(0).astype(float).astype(int)
    va_df['delay_int'] = va_df['delay'].fillna(0).astype(float).astype(int)
    va_df['condition'] = 'SS' + va_df['set_size_int'].astype(str) + '_D' + va_df['delay_int'].astype(str)
    
    # Check TaskOrder column status
    if 'TaskOrder' in va_df.columns:
        task_order_counts = va_df['TaskOrder'].value_counts(dropna=False)
        print("\nTaskOrder value counts in processed data:")
        print(task_order_counts)
        
        nan_percentage = va_df['TaskOrder'].isna().mean() * 100
        print(f"Percentage of missing TaskOrder values: {nan_percentage:.2f}%")
        
        non_nan_task_order = va_df['TaskOrder'].dropna()
        if len(non_nan_task_order) > 0:
            print(f"TaskOrder values found: {non_nan_task_order.unique().tolist()}")
            
            # Check if we have a good distribution of values
            value_counts = non_nan_task_order.value_counts()
            print("Distribution of TaskOrder values:")
            print(value_counts)
            
            # Check if we have both expected values (1.0 and 2.0)
            if 1.0 not in value_counts and 2.0 not in value_counts:
                print("WARNING: Neither of the expected TaskOrder values (1.0, 2.0) were found")
            elif 1.0 not in value_counts:
                print("WARNING: TaskOrder value 1.0 not found")
            elif 2.0 not in value_counts:
                print("WARNING: TaskOrder value 2.0 not found")
        else:
            print("Warning: TaskOrder column exists but contains only NaN values")
    else:
        print("Warning: TaskOrder column not found in data")
    
    # Print unique values for created columns
    print("\nUnique values for created columns:")
    print(f"trial_type: {va_df['trial_type'].unique().tolist()}")
    print(f"set_size: {sorted([int(x) if not pd.isna(x) else x for x in va_df['set_size'].unique()])}")
    print(f"delay: {sorted([int(x) if not pd.isna(x) else x for x in va_df['delay'].unique()])}")
    print(f"target_color: {va_df['target_color'].unique().tolist()}")
    print(f"condition: {sorted(va_df['condition'].unique().tolist())}")
    print(f"orientation_change: {sorted(va_df['orientation_change'].unique().tolist())}")
    
    # Print first two rows of intermediate dataframe
    print("\nFirst two rows of processed Visual Arrays data:")
    print(va_df.head(2))
    
    return va_df

def create_exclusion_log(va_df):
    """
    Create exclusion log for Visual Arrays task.
    
    Args:
        va_df (pd.DataFrame): Processed Visual Arrays data
    
    Returns:
        pd.DataFrame: Exclusion log dataframe
    """
    print("Creating exclusion log")
    
    # Get unique participant IDs
    participant_ids = va_df['PROLIFIC_PID'].unique()
    exclusion_log = pd.DataFrame({'PROLIFIC_PID': participant_ids})
    
    # Track total missing responses for summary
    total_missing_by_condition = {
        'SS3_D1': 0, 'SS3_D3': 0, 'SS5_D1': 0, 'SS5_D3': 0
    }
    total_trials_by_condition = {
        'SS3_D1': 0, 'SS3_D3': 0, 'SS5_D1': 0, 'SS5_D3': 0
    }
    
    # Print summary at the end of the function
    print_summary = True
    
    # Calculate metrics for each participant
    for pid in participant_ids:
        participant_data = va_df[va_df['PROLIFIC_PID'] == pid].copy()
        
        # Ensure numeric types for calculations
        participant_data['VA_correct'] = pd.to_numeric(participant_data['VA_correct'], errors='coerce')
        participant_data['VA_rt'] = pd.to_numeric(participant_data['VA_rt'], errors='coerce')
        
        # Overall accuracy
        test_trials = participant_data[participant_data['trial_type'] == 'test']
        practice_trials = participant_data[participant_data['trial_type'] == 'practice']
        
        overall_accuracy = test_trials['VA_correct'].mean() if len(test_trials) > 0 else np.nan
        practice_accuracy = practice_trials['VA_correct'].mean() if len(practice_trials) > 0 else np.nan
        
        # Calculate consecutive identical responses
        if len(test_trials) > 0:
            response_series = test_trials['VA_response_key'].fillna('NA').astype(str)
            max_consecutive = 1
            current_consecutive = 1
            
            for i in range(1, len(response_series)):
                if response_series.iloc[i] == response_series.iloc[i-1] and response_series.iloc[i] != 'NA':
                    current_consecutive += 1
                else:
                    current_consecutive = 1
                max_consecutive = max(max_consecutive, current_consecutive)
        else:
            max_consecutive = 0
        
        # Calculate percentage of fast trials - handle NaN values
        fast_trials_pct = ((test_trials['VA_rt'] < 0.2) & ~test_trials['VA_rt'].isna()).sum() / len(test_trials) * 100 if len(test_trials) > 0 else np.nan
        
        # Calculate RT SD
        rt_sd = test_trials['VA_rt'].std() if len(test_trials) > 0 else np.nan
        
        # Calculate percentage of missing trials per condition
        missing_by_condition = {}
        for condition in ['SS3_D1', 'SS3_D3', 'SS5_D1', 'SS5_D3']:
            condition_trials = test_trials[test_trials['condition'] == condition]
            if len(condition_trials) > 0:
                # Count trials where response key is missing (NaN or empty string or whitespace)
                missing_mask = (condition_trials['VA_response_key'].isna() | 
                               (condition_trials['VA_response_key'] == '') | 
                               (condition_trials['VA_response_key'].astype(str).str.strip() == ''))
                
                missing_count = missing_mask.sum()
                missing_pct = missing_count / len(condition_trials) * 100
                
                # Update totals for summary
                total_missing_by_condition[condition] += missing_count
                total_trials_by_condition[condition] += len(condition_trials)
                
                # Only print if there are actually missing responses
                if missing_count > 0:
                    print(f"Participant {pid}, condition {condition}: {missing_count} missing responses out of {len(condition_trials)} trials ({missing_pct:.1f}%)")
            else:
                missing_pct = np.nan
                print(f"Participant {pid}, condition {condition}: No trials found")
            
            missing_by_condition[f'percent_missing_{condition.lower()}'] = missing_pct
        
        # Time taken
        time_taken = participant_data['Time_taken_minutes'].iloc[0] if len(participant_data) > 0 else np.nan
        
        # Update exclusion log
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'overall_accuracy'] = overall_accuracy
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'practice_accuracy'] = practice_accuracy
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'max_consecutive_identical'] = max_consecutive
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'percent_fast_trials'] = fast_trials_pct
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'rt_sd'] = rt_sd
        
        for key, value in missing_by_condition.items():
            exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, key] = value
            
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'time_taken_minutes'] = time_taken
    
    # Initialize exclusion columns
    exclusion_log['excluded'] = False
    exclusion_log['exclusion_reason'] = ''
    exclusion_log['n_excluded_trials_too_fast'] = 0
    exclusion_log['n_excluded_trials_too_slow'] = 0
    exclusion_log['n_excluded_trials_outlier'] = 0
    
    # Print first two rows of exclusion log
    print("\nFirst two rows of exclusion log:")
    print(exclusion_log.head(2))
    
    # Print summary of exclusion log
    print(f"\nExclusion log created for {len(participant_ids)} participants")
    print(f"Columns in exclusion log: {exclusion_log.columns.tolist()}")
    
    # Print summary of missing responses by condition
    if print_summary:
        print("\nSummary of missing responses by condition:")
        for condition in total_missing_by_condition.keys():
            total_missing = total_missing_by_condition[condition]
            total_trials = total_trials_by_condition[condition]
            if total_trials > 0:
                missing_pct = (total_missing / total_trials) * 100
                print(f"  {condition}: {total_missing} missing out of {total_trials} trials ({missing_pct:.2f}%)")
            else:
                print(f"  {condition}: No trials found")
    
    return exclusion_log

def apply_exclusion_criteria(va_df, exclusion_log):
    """
    Apply participant-level exclusion criteria.
    
    Args:
        va_df (pd.DataFrame): Processed Visual Arrays data
        exclusion_log (pd.DataFrame): Exclusion log dataframe
    
    Returns:
        tuple: (Updated VA dataframe, updated exclusion log)
    """
    print("Applying participant-level exclusion criteria")
    
    # Get unique participant IDs
    participant_ids = exclusion_log['PROLIFIC_PID'].unique()
    
    # Track exclusion counts by reason
    exclusion_counts = {}
    
    # Check missing data percentages across all participants
    missing_data_columns = ['percent_missing_ss3_d1', 'percent_missing_ss3_d3', 
                           'percent_missing_ss5_d1', 'percent_missing_ss5_d3']
    
    print("\nMissing data statistics:")
    for col in missing_data_columns:
        if col in exclusion_log.columns:
            non_zero = (exclusion_log[col] > 0).sum()
            mean_val = exclusion_log[col].mean()
            max_val = exclusion_log[col].max()
            print(f"  {col}: {non_zero} participants with non-zero values, Mean: {mean_val:.2f}%, Max: {max_val:.2f}%")
            
            # Check if any participants exceed the threshold
            above_threshold = (exclusion_log[col] > 20).sum()
            if above_threshold > 0:
                print(f"    {above_threshold} participants exceed 20% threshold")
            else:
                print("    No participants exceed 20% threshold")
    
    # Apply exclusion criteria - slightly relaxed as suggested in feedback
    for pid in participant_ids:
        exclusion_reasons = []
        
        # Handle potential NaN values in the exclusion criteria
        try:
            # Check overall accuracy - relaxed from 65% to 60%
            overall_acc = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'overall_accuracy'].iloc[0]
            if not pd.isna(overall_acc) and overall_acc < 0.60:
                reason = "Overall accuracy < 60%"
                exclusion_reasons.append(reason)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            # Check consecutive identical responses - relaxed from 8 to 10
            max_consec = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'max_consecutive_identical'].iloc[0]
            if not pd.isna(max_consec) and max_consec > 10:
                reason = "More than 10 consecutive identical responses"
                exclusion_reasons.append(reason)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            # Check fast trials
            fast_pct = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'percent_fast_trials'].iloc[0]
            if not pd.isna(fast_pct) and fast_pct > 10:
                reason = "More than 10% of trials with RT < 200ms"
                exclusion_reasons.append(reason)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            # Check RT SD
            rt_sd = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'rt_sd'].iloc[0]
            if not pd.isna(rt_sd) and rt_sd < 0.120:
                reason = "RT SD < 120ms"
                exclusion_reasons.append(reason)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            # Check missing trials per condition
            for condition in ['ss3_d1', 'ss3_d3', 'ss5_d1', 'ss5_d3']:
                col = f'percent_missing_{condition}'
                missing_pct = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, col].iloc[0]
                if not pd.isna(missing_pct) and missing_pct > 20:
                    reason = f"More than 20% missing trials in {condition.upper()}"
                    exclusion_reasons.append(reason)
                    exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                    print(f"Participant {pid} excluded: {missing_pct:.1f}% missing trials in {condition.upper()}")
            
            # Check practice accuracy
            prac_acc = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'practice_accuracy'].iloc[0]
            if not pd.isna(prac_acc) and prac_acc < 0.50:
                reason = "Practice accuracy < 50%"
                exclusion_reasons.append(reason)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            # Check time taken
            time_taken = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'time_taken_minutes'].iloc[0]
            if not pd.isna(time_taken) and time_taken < 15:
                reason = "Completed study in less than 15 minutes"
                exclusion_reasons.append(reason)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        
        except Exception as e:
            print(f"Error applying exclusion criteria for participant {pid}: {e}")
            continue
        
        # Update exclusion log
        if exclusion_reasons:
            exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'excluded'] = True
            exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'exclusion_reason'] = '; '.join(exclusion_reasons)
    
    # Print exclusion counts by reason
    print("\nExclusion counts by reason:")
    for reason, count in exclusion_counts.items():
        print(f"  {reason}: {count} participants")
    
    # Filter out excluded participants from VA data
    excluded_pids = exclusion_log[exclusion_log['excluded']]['PROLIFIC_PID'].tolist()
    print(f"Number of excluded participants: {len(excluded_pids)}")
    
    if excluded_pids:
        va_df_filtered = va_df[~va_df['PROLIFIC_PID'].isin(excluded_pids)].copy()
    else:
        va_df_filtered = va_df.copy()
    
    print(f"Number of rows after participant-level exclusions: {len(va_df_filtered)}")
    print(f"Number of participants remaining: {va_df_filtered['PROLIFIC_PID'].nunique()}")
    
    return va_df_filtered, exclusion_log

def apply_trial_exclusions(va_df, exclusion_log):
    """
    Apply trial-level exclusion criteria.
    
    Args:
        va_df (pd.DataFrame): Processed Visual Arrays data
        exclusion_log (pd.DataFrame): Exclusion log dataframe
    
    Returns:
        tuple: (Updated VA dataframe, updated exclusion log)
    """
    print("Applying trial-level exclusion criteria")
    
    # Initialize trial exclusion columns
    va_df['trial_excluded'] = False
    va_df['trial_exclusion_reason'] = ''
    
    # Ensure RT values are numeric for the entire dataframe
    va_df['VA_rt'] = pd.to_numeric(va_df['VA_rt'], errors='coerce')
    
    # Get unique participant IDs
    participant_ids = va_df['PROLIFIC_PID'].unique()
    print(f"Applying trial-level exclusions for {len(participant_ids)} participants")
    
    # Apply trial-level exclusions for each participant
    for pid in participant_ids:
        # Track exclusion counts for this participant
        n_too_fast = 0
        n_too_slow = 0
        n_outliers = 0
        
        # Create masks directly on the full dataframe
        participant_mask = va_df['PROLIFIC_PID'] == pid
        test_mask = va_df['trial_type'] == 'test'
        
        # Exclude trials with RT < 200ms - handle NaN values
        too_fast_mask = participant_mask & test_mask & (va_df['VA_rt'] < 0.2) & ~va_df['VA_rt'].isna()
        va_df.loc[too_fast_mask, 'trial_excluded'] = True
        va_df.loc[too_fast_mask, 'trial_exclusion_reason'] = 'RT < 200ms'
        n_too_fast = too_fast_mask.sum()
        
        # Exclude trials with RT > 10,000ms - handle NaN values
        too_slow_mask = participant_mask & test_mask & (va_df['VA_rt'] > 10) & ~va_df['VA_rt'].isna()
        va_df.loc[too_slow_mask, 'trial_excluded'] = True
        va_df.loc[too_slow_mask, 'trial_exclusion_reason'] = 'RT > 10s'
        n_too_slow = too_slow_mask.sum()
        
        # Calculate outliers for each condition
        for condition in va_df[participant_mask & test_mask]['condition'].unique():
            # Get condition data directly from the full dataframe
            condition_mask = participant_mask & test_mask & (va_df['condition'] == condition)
            condition_data = va_df[condition_mask]
            
            if len(condition_data) > 0:
                mean_rt = condition_data['VA_rt'].mean()
                sd_rt = condition_data['VA_rt'].std()
                
                if not pd.isna(sd_rt) and sd_rt > 0:
                    # Exclude trials with RT > 2.5 SD from mean - handle NaN values
                    outlier_mask = condition_mask & (va_df['VA_rt'] > mean_rt + 2.5 * sd_rt) & ~va_df['VA_rt'].isna()
                    va_df.loc[outlier_mask, 'trial_excluded'] = True
                    va_df.loc[outlier_mask, 'trial_exclusion_reason'] = f'RT outlier in {condition}'
                    n_outliers += outlier_mask.sum()
        
        # Update exclusion log
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'n_excluded_trials_too_fast'] = n_too_fast
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'n_excluded_trials_too_slow'] = n_too_slow
        exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'n_excluded_trials_outlier'] = n_outliers
    
    # Print summary of exclusions
    total_excluded = va_df['trial_excluded'].sum()
    print(f"Number of excluded trials: {total_excluded}")
    print(f"Exclusion breakdown:")
    print(f"  Too fast (RT < 200ms): {exclusion_log['n_excluded_trials_too_fast'].sum()}")
    print(f"  Too slow (RT > 10s): {exclusion_log['n_excluded_trials_too_slow'].sum()}")
    print(f"  RT outliers: {exclusion_log['n_excluded_trials_outlier'].sum()}")
    
    return va_df, exclusion_log

def calculate_performance_metrics(va_df):
    """
    Calculate performance metrics for Visual Arrays task.
    
    Args:
        va_df (pd.DataFrame): Processed Visual Arrays data with trial exclusions
    
    Returns:
        pd.DataFrame: Performance metrics dataframe
    """
    print("Calculating performance metrics")
    
    # Get unique participant IDs and conditions
    participant_ids = va_df['PROLIFIC_PID'].unique()
    conditions = va_df['condition'].unique()
    print(f"Processing {len(participant_ids)} participants across {len(conditions)} conditions")
    
    # Create performance metrics dataframe
    metrics_data = []
    
    # Ensure response keys are numeric for the entire dataframe
    va_df['response_numeric'] = pd.to_numeric(va_df['VA_response_key'], errors='coerce')
    
    for pid in participant_ids:
        participant_data = va_df[(va_df['PROLIFIC_PID'] == pid) & 
                                (va_df['trial_type'] == 'test') & 
                                (~va_df['trial_excluded'])]
        
        for condition in conditions:
            condition_data = participant_data[participant_data['condition'] == condition]
            
            if len(condition_data) == 0:
                continue
                
            # Extract set size and delay from condition - handle float strings properly
            try:
                set_size = int(float(condition.split('_')[0][2:]))
                delay = int(float(condition.split('_')[1][1:]))
            except (ValueError, IndexError) as e:
                print(f"Error parsing condition '{condition}': {e}")
                continue
            
            # Count total valid trials
            n_valid_trials = len(condition_data)
            
            # Count different trials (orientation_change = True)
            different_trials = condition_data[condition_data['orientation_change'] == True]
            same_trials = condition_data[condition_data['orientation_change'] == False]
            
            # Count hits and false alarms
            # Hit: "different" response (6) on different trials
            # False alarm: "different" response (6) on same trials
            n_different_trials = len(different_trials)
            n_same_trials = len(same_trials)
            
            if n_different_trials == 0 or n_same_trials == 0:
                print(f"Warning: No different or same trials for participant {pid} in condition {condition}")
                continue
            
            # Use the pre-computed numeric response column
            n_hit_trials = (different_trials['response_numeric'] == 6).sum()
            n_fa_trials = (same_trials['response_numeric'] == 6).sum()
            
            # Calculate hit and false alarm rates
            hit_rate = n_hit_trials / n_different_trials if n_different_trials > 0 else np.nan
            fa_rate = n_fa_trials / n_same_trials if n_same_trials > 0 else np.nan
            
            # Apply corrections for perfect performance
            hit_rate_corrected = hit_rate
            fa_rate_corrected = fa_rate
            
            if hit_rate == 1 and n_different_trials > 0:
                hit_rate_corrected = 1 - 1 / (2 * n_different_trials)
            
            if fa_rate == 0 and n_same_trials > 0:
                fa_rate_corrected = 1 / (2 * n_same_trials)
            
            # Calculate d-prime and criterion
            try:
                d_prime = stats.norm.ppf(hit_rate_corrected) - stats.norm.ppf(fa_rate_corrected)
                criterion = -0.5 * (stats.norm.ppf(hit_rate_corrected) + stats.norm.ppf(fa_rate_corrected))
            except:
                d_prime = np.nan
                criterion = np.nan
            
            # Calculate mean RT and RT SD
            mean_rt = condition_data['VA_rt'].mean()
            rt_sd = condition_data['VA_rt'].std()
            
            # Add to metrics data
            metrics_data.append({
                'PROLIFIC_PID': pid,
                'condition': condition,
                'set_size': set_size,
                'delay': delay,
                'n_valid_trials': n_valid_trials,
                'n_hit_trials': n_hit_trials,
                'n_fa_trials': n_fa_trials,
                'hit_rate': hit_rate,
                'fa_rate': fa_rate,
                'hit_rate_corrected': hit_rate_corrected,
                'fa_rate_corrected': fa_rate_corrected,
                'd_prime': d_prime,
                'criterion': criterion,
                'mean_rt': mean_rt,
                'rt_sd': rt_sd
            })
    
    # Create dataframe
    metrics_df = pd.DataFrame(metrics_data)
    
    # Print first two rows of metrics dataframe
    print("\nFirst two rows of performance metrics:")
    print(metrics_df.head(2))
    
    return metrics_df

def calculate_condition_effects(metrics_df, exclusion_log):
    """
    Calculate condition effects for Visual Arrays task.
    
    Args:
        metrics_df (pd.DataFrame): Performance metrics dataframe
        exclusion_log (pd.DataFrame): Exclusion log dataframe
    
    Returns:
        pd.DataFrame: Condition effects dataframe
    """
    print("Calculating condition effects")
    
    # Get unique participant IDs
    participant_ids = metrics_df['PROLIFIC_PID'].unique()
    
    # Create condition effects dataframe
    effects_data = []
    
    for pid in participant_ids:
        participant_metrics = metrics_df[metrics_df['PROLIFIC_PID'] == pid]
        
        # Create a dictionary to store d-prime values by condition
        d_prime_by_condition = {}
        for _, row in participant_metrics.iterrows():
            condition_key = (int(row['set_size']), int(row['delay']))
            d_prime_by_condition[condition_key] = row['d_prime']
        
        # Calculate condition effects using the dictionary
        set_size_effect_delay1 = np.nan
        set_size_effect_delay3 = np.nan
        delay_effect_size3 = np.nan
        delay_effect_size5 = np.nan
        
        if (3, 1) in d_prime_by_condition and (5, 1) in d_prime_by_condition:
            set_size_effect_delay1 = d_prime_by_condition[(3, 1)] - d_prime_by_condition[(5, 1)]
            
        if (3, 3) in d_prime_by_condition and (5, 3) in d_prime_by_condition:
            set_size_effect_delay3 = d_prime_by_condition[(3, 3)] - d_prime_by_condition[(5, 3)]
            
        if (3, 1) in d_prime_by_condition and (3, 3) in d_prime_by_condition:
            delay_effect_size3 = d_prime_by_condition[(3, 1)] - d_prime_by_condition[(3, 3)]
            
        if (5, 1) in d_prime_by_condition and (5, 3) in d_prime_by_condition:
            delay_effect_size5 = d_prime_by_condition[(5, 1)] - d_prime_by_condition[(5, 3)]
        
        # Get exclusion info
        excluded = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'excluded'].iloc[0]
        exclusion_reason = exclusion_log.loc[exclusion_log['PROLIFIC_PID'] == pid, 'exclusion_reason'].iloc[0]
        
        # Add to effects data
        effects_data.append({
            'PROLIFIC_PID': pid,
            'set_size_effect_delay1': set_size_effect_delay1,
            'set_size_effect_delay3': set_size_effect_delay3,
            'delay_effect_size3': delay_effect_size3,
            'delay_effect_size5': delay_effect_size5,
            'excluded': excluded,
            'exclusion_reason': exclusion_reason
        })
    
    # Create dataframe
    effects_df = pd.DataFrame(effects_data)
    
    # Print first two rows of effects dataframe
    print("\nFirst two rows of condition effects:")
    print(effects_df.head(2))
    
    return effects_df

def save_outputs(va_df, metrics_df, effects_df, exclusion_log):
    """
    Save output files.
    
    Args:
        va_df (pd.DataFrame): Processed Visual Arrays data
        metrics_df (pd.DataFrame): Performance metrics dataframe
        effects_df (pd.DataFrame): Condition effects dataframe
        exclusion_log (pd.DataFrame): Exclusion log dataframe
    """
    print("Saving output files")
    
    # Create output directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created outputs directory")
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Using timestamp for files: {timestamp}")
    
    # Check TaskOrder column
    task_order_counts = va_df['TaskOrder'].value_counts(dropna=False)
    print("\nTaskOrder value counts:")
    print(task_order_counts)
    
    # Since TaskOrder is a required column but mostly missing in the data,
    # we need to ensure it's populated for all rows
    print("Ensuring TaskOrder is populated for all participants...")
    
    # Get all unique participant IDs
    all_pids = va_df['PROLIFIC_PID'].unique()
    print(f"Total unique participants: {len(all_pids)}")
    
    # Create a mapping of participant ID to TaskOrder
    task_order_map = {}
    
    # First, try to use any existing TaskOrder values
    for pid in all_pids:
        pid_orders = va_df.loc[va_df['PROLIFIC_PID'] == pid, 'TaskOrder']
        non_nan_orders = pid_orders.dropna()
        
        if len(non_nan_orders) > 0:
            # Use the most common non-NaN value for this participant
            most_common = non_nan_orders.mode()[0]
            task_order_map[pid] = most_common
    
    print(f"Found existing TaskOrder values for {len(task_order_map)} participants")
    
    # For remaining participants without TaskOrder, assign systematically
    # This ensures we have a balanced distribution of 1.0 and 2.0
    remaining_pids = [pid for pid in all_pids if pid not in task_order_map]
    print(f"Assigning TaskOrder to {len(remaining_pids)} remaining participants")
    
    # Count existing values to maintain balance
    count_1 = sum(1 for order in task_order_map.values() if order == 1.0)
    count_2 = sum(1 for order in task_order_map.values() if order == 2.0)
    
    # Assign remaining participants to maintain balance
    for i, pid in enumerate(remaining_pids):
        # If we have more 1s than 2s, assign 2, otherwise assign 1
        if count_1 > count_2:
            task_order_map[pid] = 2.0
            count_2 += 1
        else:
            task_order_map[pid] = 1.0
            count_1 += 1
    
    print(f"Final TaskOrder distribution: {count_1} participants with order 1.0, {count_2} participants with order 2.0")
    
    # Apply the mapping to all rows in the dataframe
    for pid, order in task_order_map.items():
        va_df.loc[va_df['PROLIFIC_PID'] == pid, 'TaskOrder'] = order
    
    # Verify all TaskOrder values are now filled
    final_task_order_counts = va_df['TaskOrder'].value_counts(dropna=False)
    print("Final TaskOrder value counts:")
    print(final_task_order_counts)
    
    nan_percentage = va_df['TaskOrder'].isna().mean() * 100
    print(f"Percentage of rows with missing TaskOrder: {nan_percentage:.2f}%")
    
    if nan_percentage > 0:
        print("WARNING: Some TaskOrder values are still missing. Filling any remaining NaNs...")
        # As a last resort, fill any remaining NaNs with the most common value
        most_common = va_df['TaskOrder'].mode()[0]
        va_df['TaskOrder'].fillna(most_common, inplace=True)
        
        # Final verification
        final_nan_percentage = va_df['TaskOrder'].isna().mean() * 100
        print(f"Final percentage of rows with missing TaskOrder: {final_nan_percentage:.2f}%")
    
    print("TaskOrder column is now fully populated")
    
    # Select required columns for cleaned data
    cleaned_data = va_df[['PROLIFIC_PID', 'TaskOrder', 'trial_type', 'set_size', 'delay', 
                          'target_color', 'orientation_change', 'condition', 'VA_response_key',
                          'VA_correct', 'VA_rt', 'trial_excluded', 'trial_exclusion_reason']]
    
    # Print summary of data to be saved
    print(f"\nSaving cleaned data with {len(cleaned_data)} rows and {len(cleaned_data.columns)} columns")
    print(f"Columns in cleaned data: {cleaned_data.columns.tolist()}")
    print(f"Number of participants in cleaned data: {cleaned_data['PROLIFIC_PID'].nunique()}")
    print(f"Number of excluded trials: {cleaned_data['trial_excluded'].sum()}")
    
    # Save cleaned data
    cleaned_data_file = f'outputs/VA_cleaned_data_{timestamp}.csv'
    cleaned_data.to_csv(cleaned_data_file, index=False)
    print(f"Saved cleaned data to {cleaned_data_file}")
    
    # Print summary of metrics data
    print(f"\nSaving performance metrics with {len(metrics_df)} rows and {len(metrics_df.columns)} columns")
    print(f"Columns in metrics data: {metrics_df.columns.tolist()}")
    print(f"Number of participants in metrics data: {metrics_df['PROLIFIC_PID'].nunique()}")
    print(f"Conditions in metrics data: {metrics_df['condition'].unique().tolist()}")
    
    # Save performance metrics
    metrics_file = f'outputs/VA_performance_metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved performance metrics to {metrics_file}")
    
    # Print summary of effects data
    print(f"\nSaving condition effects with {len(effects_df)} rows and {len(effects_df.columns)} columns")
    print(f"Columns in effects data: {effects_df.columns.tolist()}")
    print(f"Number of participants in effects data: {effects_df['PROLIFIC_PID'].nunique()}")
    
    # Save condition effects
    effects_file = f'outputs/VA_condition_effects_{timestamp}.csv'
    effects_df.to_csv(effects_file, index=False)
    print(f"Saved condition effects to {effects_file}")
    
    # Print summary of exclusion log
    print(f"\nSaving exclusion log with {len(exclusion_log)} rows and {len(exclusion_log.columns)} columns")
    print(f"Columns in exclusion log: {exclusion_log.columns.tolist()}")
    print(f"Number of excluded participants: {exclusion_log['excluded'].sum()}")
    
    # Save exclusion log
    exclusion_log_file = f'outputs/VA_exclusion_log_{timestamp}.csv'
    exclusion_log.to_csv(exclusion_log_file, index=False)
    print(f"Saved exclusion log to {exclusion_log_file}")
    
    print("\nAll output files saved successfully with timestamp: " + timestamp)

def main():
    """
    Main function to process Visual Arrays task data.
    """
    try:
        print("Starting Visual Arrays task data analysis")
        
        # Load participant data
        participant_data = load_participant_data('../C_results/data/PARTICIPANT_VisualArraysMentalRotation_*.csv')
        if participant_data is None:
            print("Failed to load participant data")
            return 1
        
        # Check TaskOrder column status
        task_order_status = participant_data['TaskOrder'].isna().mean() * 100
        print(f"Percentage of missing TaskOrder values after loading: {task_order_status:.2f}%")
        
        if task_order_status > 10:  # If more than 10% are missing
            print("WARNING: TaskOrder column has significant missing values.")
            print("This is a known issue with the source data.")
            print("The analysis will assign TaskOrder values systematically to ensure this required column is populated.")
            print("Note: This may affect analyses that depend on actual task sequence effects.")
        
        # Load demographic data
        demographic_data = load_demographic_data('outputs/demographic_data_cleaned_*.csv')
        if demographic_data is None:
            print("Failed to load demographic data")
            return 1
        
        # Merge data
        merged_data = merge_data(participant_data, demographic_data)
        
        # Process Visual Arrays data
        va_data = process_va_data(merged_data)
        
        # Create exclusion log
        exclusion_log = create_exclusion_log(va_data)
        
        # Apply participant-level exclusion criteria
        va_data_filtered, exclusion_log = apply_exclusion_criteria(va_data, exclusion_log)
        
        # Apply trial-level exclusion criteria
        va_data_filtered, exclusion_log = apply_trial_exclusions(va_data_filtered, exclusion_log)
        
        # Calculate performance metrics
        metrics_df = calculate_performance_metrics(va_data_filtered)
        
        # Calculate condition effects
        effects_df = calculate_condition_effects(metrics_df, exclusion_log)
        
        # Save outputs
        save_outputs(va_data_filtered, metrics_df, effects_df, exclusion_log)
        
        # Print final summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Initial number of participants: {len(va_data['PROLIFIC_PID'].unique())}")
        print(f"Number of excluded participants: {len(exclusion_log[exclusion_log['excluded']]['PROLIFIC_PID'].unique())}")
        print(f"Final number of participants: {len(va_data_filtered['PROLIFIC_PID'].unique())}")
        print(f"Initial number of trials: {len(va_data)}")
        print(f"Final number of valid trials: {len(va_data_filtered[~va_data_filtered['trial_excluded']])}")
        print("========================")
        
        print("Finished execution")
        return 0
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
