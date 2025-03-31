import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import re
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_latest_file(pattern):
    """
    Gets the most recently created file matching the specified pattern.
    
    Parameters:
    -----------
    pattern : str
        File pattern to search for, with wildcards
        
    Returns:
    --------
    str
        Path to the most recent file matching the pattern
    """
    matching_files = glob.glob(pattern)
    if not matching_files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    # Get the most recent file
    latest_file = max(matching_files, key=os.path.getctime)
    print(f"Latest file matching {pattern}: {latest_file}")
    return latest_file

def load_mental_rotation_data(file_pattern):
    """
    Load all Mental Rotation Task data files matching the specified pattern.
    
    Parameters:
    -----------
    file_pattern : str
        File pattern to search for, with wildcards
        
    Returns:
    --------
    pandas.DataFrame
        Combined dataframe with all MRT data
    """
    print(f"Looking for files matching pattern: {file_pattern}")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    print(f"Found {len(all_files)} files matching pattern")
    
    # List to store dataframes
    dfs = []
    
    # Required columns
    required_columns = [
        'PROLIFIC_PID', 'TaskOrder', 'MRTStimulus', 'MRTType', 'MRTDegRotation',
        'MRTCorrectAns', 'MRTPracResponse.keys', 'MRTPracResponse.corr', 'MRTPracResponse.rt',
        'MRTPracticeTrials.thisRepN', 'MRTPracticeTrials.thisTrialN', 'MRTPracticeTrials.thisN',
        'MRTPracticeTrials.thisIndex', 'MRTTestResponse.keys', 'MRTTestResponse.corr', 'MRTTestResponse.rt',
        'MRTTestTrials.thisRepN', 'MRTTestTrials.thisTrialN', 'MRTTestTrials.thisN', 'MRTTestTrials.thisIndex'
    ]
    
    # Read each file
    for file in all_files:
        print(f"Reading file: {file}")
        try:
            # Extract participant ID from filename
            participant_id_match = re.search(r'PARTICIPANT_VisualArraysMentalRotation_(.+?)\.csv', file)
            filename_pid = participant_id_match.group(1) if participant_id_match else None
            
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Check if all required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Skipping file {file} due to missing columns: {missing_cols}")
                continue
                
            # If PROLIFIC_PID column is empty but we have ID from filename, use that
            if 'PROLIFIC_PID' in df.columns and df['PROLIFIC_PID'].isna().all() and filename_pid:
                print(f"Using participant ID from filename: {filename_pid}")
                df['PROLIFIC_PID'] = filename_pid
                
            # Append to list of dataframes
            dfs.append(df)
            
            # Print dataframe info for verification
            print(f"File {file} has {df.shape[0]} rows and {df.shape[1]} columns")
            print("Columns:", df.columns.tolist())
            print("First 3 rows:")
            print(df.head(3))
            
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    if not dfs:
        print("No valid data files found")
        return None
        
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")
    
    # Print unique values for key experimental columns
    print("\nUnique values for key experimental parameters:")
    print(f"TaskOrder: {combined_df['TaskOrder'].unique()}")
    print(f"MRTType: {combined_df['MRTType'].unique()}")
    print(f"MRTDegRotation: {combined_df['MRTDegRotation'].unique()}")
    print(f"MRTCorrectAns: {combined_df['MRTCorrectAns'].unique()}")
    print("MRTPracResponse.keys type:", combined_df['MRTPracResponse.keys'].dtype)
    print("MRTTestResponse.keys type:", combined_df['MRTTestResponse.keys'].dtype)
    print("MRTPracResponse.rt type:", combined_df['MRTPracResponse.rt'].dtype)
    print("MRTTestResponse.rt type:", combined_df['MRTTestResponse.rt'].dtype)
    
    return combined_df

def load_demographic_data():
    """
    Load the most recent demographic data file.
    
    Returns:
    --------
    pandas.DataFrame
        Demographic data for included participants
    """
    # Get the latest demographic data file
    demographic_pattern = "outputs/demographic_data_cleaned_*.csv"
    latest_demographic_file = get_latest_file(demographic_pattern)
    
    if not latest_demographic_file:
        print("No demographic data file found")
        return None
        
    # Required columns
    required_columns = ['PROLIFIC_PID', 'Included', 'Time_taken_minutes']
    
    # Read the CSV file
    try:
        demographic_df = pd.read_csv(latest_demographic_file)
        
        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in demographic_df.columns]
        if missing_cols:
            print(f"Demographic data file missing required columns: {missing_cols}")
            return None
            
        print(f"Demographic data has {demographic_df.shape[0]} rows and {demographic_df.shape[1]} columns")
        print("Columns:", demographic_df.columns.tolist())
        print("First 3 rows:")
        print(demographic_df.head(3))
        
        # Filter to only included participants
        included_df = demographic_df[demographic_df['Included'] == True].copy()
        print(f"After filtering for Included=TRUE, {included_df.shape[0]} participants remain")
        
        return included_df
        
    except Exception as e:
        print(f"Error reading demographic data file: {e}")
        return None

def prepare_mrt_data(mrt_df):
    """
    Prepare Mental Rotation Task data by identifying trial types and creating unified response columns.
    
    Parameters:
    -----------
    mrt_df : pandas.DataFrame
        Combined MRT data
        
    Returns:
    --------
    pandas.DataFrame
        Prepared MRT data with trial types and unified response columns
    """
    # Make a copy to avoid modifying the original
    df = mrt_df.copy()
    
    # Identify trial types
    print("Identifying trial types...")
    
    # Create trial_type column
    df['trial_type'] = None
    
    # Check for practice trials (where MRTPracResponse columns have values)
    practice_mask = ~df['MRTPracResponse.keys'].isna()
    df.loc[practice_mask, 'trial_type'] = "practice"
    
    # Check for test trials (where MRTTestResponse columns have values)
    test_mask = ~df['MRTTestResponse.keys'].isna()
    df.loc[test_mask, 'trial_type'] = "test"
    
    print(f"Identified {practice_mask.sum()} practice trials and {test_mask.sum()} test trials")
    
    # Create unified response columns
    print("Creating unified response columns...")
    
    # Initialize new columns
    df['MRT_response_key'] = None
    df['MRT_correct'] = None
    df['MRT_rt'] = None
    
    # Combine practice and test responses
    df.loc[practice_mask, 'MRT_response_key'] = df.loc[practice_mask, 'MRTPracResponse.keys']
    df.loc[practice_mask, 'MRT_correct'] = df.loc[practice_mask, 'MRTPracResponse.corr']
    df.loc[practice_mask, 'MRT_rt'] = df.loc[practice_mask, 'MRTPracResponse.rt']
    
    df.loc[test_mask, 'MRT_response_key'] = df.loc[test_mask, 'MRTTestResponse.keys']
    df.loc[test_mask, 'MRT_correct'] = df.loc[test_mask, 'MRTTestResponse.corr']
    df.loc[test_mask, 'MRT_rt'] = df.loc[test_mask, 'MRTTestResponse.rt']
    
    # Print response key diagnostics
    print("\nResponse key value counts:")
    print(df['MRT_response_key'].value_counts(dropna=False).head(10))
    print("Response key unique values:", df['MRT_response_key'].unique())
    
    # Check for missing responses by trial type - include empty strings and 'None' values
    test_mask = df['trial_type'] == 'test'
    missing_mask = df['MRT_response_key'].isna() | (df['MRT_response_key'] == '') | (df['MRT_response_key'] == 'None')
    missing_test = df[test_mask & missing_mask]
    print(f"Found {len(missing_test)} missing responses in test trials")
    print("Sample of test trials with missing responses:")
    if len(missing_test) > 0:
        print(missing_test[['PROLIFIC_PID', 'angular_disparity', 'MRTTestResponse.keys']].head(5))
    
    # Check for NaNs in the original response columns
    test_nan_count = df['MRTTestResponse.keys'].isna().sum()
    test_total = len(df[~df['MRTTestResponse.keys'].isna()])
    print(f"MRTTestResponse.keys: {test_nan_count} NaNs out of {test_total + test_nan_count} total values ({test_nan_count/(test_total + test_nan_count)*100:.2f}%)")
    
    # Check for missing responses by participant - include empty strings and 'None' values
    missing_by_participant = df[test_mask].groupby('PROLIFIC_PID')['MRT_response_key'].apply(
        lambda x: (x.isna() | (x == '') | (x == 'None')).sum())
    participants_with_missing = missing_by_participant[missing_by_participant > 0]
    print(f"\nParticipants with missing responses: {len(participants_with_missing)} out of {len(missing_by_participant)}")
    if len(participants_with_missing) > 0:
        print("Top 5 participants with most missing responses:")
        for pid, count in participants_with_missing.sort_values(ascending=False).head(5).items():
            total_trials = len(df[(df['trial_type'] == 'test') & (df['PROLIFIC_PID'] == pid)])
            print(f"  {pid}: {count} missing out of {total_trials} trials ({count/total_trials*100:.2f}%)")
    
    # Create angular_disparity, stimulus_type, and condition columns
    df['angular_disparity'] = pd.to_numeric(df['MRTDegRotation'], errors='coerce')
    df['stimulus_type'] = df['MRTType']
    
    # Handle NaN values in angular_disparity properly - set to None instead of "ADnan"
    df['condition'] = df['angular_disparity'].apply(lambda x: f"AD{int(x)}" if pd.notnull(x) else None)
    
    # Validate the condition column
    print(f"Unique conditions after processing: {df['condition'].unique().tolist()}")
    
    # Print the first few rows of the prepared data
    print("\nPrepared MRT data (first 2 rows):")
    print(df[['PROLIFIC_PID', 'trial_type', 'angular_disparity', 'stimulus_type', 
              'condition', 'MRT_response_key', 'MRT_correct', 'MRT_rt']].head(2))
    
    # Print counts of trial types
    print(f"\nTrial type counts: {df['trial_type'].value_counts().to_dict()}")
    
    # Print counts of conditions
    print(f"Condition counts: {df['condition'].value_counts().to_dict()}")
    
    return df

def apply_participant_exclusions(mrt_df, demographic_df):
    """
    Apply participant-level exclusion criteria and create exclusion log.
    
    Parameters:
    -----------
    mrt_df : pandas.DataFrame
        Prepared MRT data
    demographic_df : pandas.DataFrame
        Demographic data for included participants
        
    Returns:
    --------
    tuple
        (filtered_df, exclusion_log_df, excluded_participants)
        
    Notes:
    ------
    This function applies several exclusion criteria:
    - Overall accuracy < 60% (relaxed from 65% based on expert feedback)
    - Practice accuracy < 50%
    - Identical responses for > 8 consecutive trials
    - Fast trials (< 200ms) > 10%
    - RT SD < 120ms
    - Missing trials > 20% in any condition
    - Time taken < 15 minutes
    """
    print("Applying participant-level exclusion criteria...")
    
    # Create a copy of the data
    df = mrt_df.copy()
    
    # Merge with demographic data to get time taken
    merged_df = pd.merge(df, demographic_df[['PROLIFIC_PID', 'Time_taken_minutes']], 
                         on='PROLIFIC_PID', how='left')
    
    # Get unique participant IDs
    participant_ids = df['PROLIFIC_PID'].unique()
    print(f"Evaluating {len(participant_ids)} participants for exclusion criteria")
    
    # Print trial counts by condition for a sample participant
    sample_pid = participant_ids[0]
    sample_data = df[df['PROLIFIC_PID'] == sample_pid]
    sample_test_trials = sample_data[sample_data['trial_type'] == 'test']
    print(f"\nTrial counts for sample participant {sample_pid}:")
    for angle in [0, 50, 100, 150]:
        angle_trials = sample_test_trials[sample_test_trials['angular_disparity'] == angle]
        print(f"AD{angle}: {len(angle_trials)} trials, {angle_trials['MRT_response_key'].isna().sum()} missing")
    
    # Check for participants with uneven trial counts across conditions
    print("\nAnalyzing trial count distribution across conditions:")
    trial_counts = df[df['trial_type'] == 'test'].groupby(['PROLIFIC_PID', 'angular_disparity']).size().unstack(fill_value=0)
    if not trial_counts.empty:
        print(f"Trial count statistics:")
        for angle in [0, 50, 100, 150]:
            if angle in trial_counts.columns:
                stats = trial_counts[angle].describe()
                print(f"  AD{angle}: Min={stats['min']:.0f}, Max={stats['max']:.0f}, Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
        
        # Check for participants with highly uneven trial counts
        trial_count_std = trial_counts.std(axis=1)
        uneven_participants = trial_count_std[trial_count_std > 1]
        if len(uneven_participants) > 0:
            print(f"\nParticipants with uneven trial counts across conditions: {len(uneven_participants)}")
            print("Examples (first 3):")
            for pid in uneven_participants.head(3).index:
                print(f"  {pid}: {trial_counts.loc[pid].to_dict()}")
    
    # Initialize exclusion log dataframe
    exclusion_log = []
    
    # Initialize list of excluded participants
    excluded_participants = []
    
    # Loop through each participant
    for pid in participant_ids:
        # Get participant data
        p_data = df[df['PROLIFIC_PID'] == pid].copy()
        
        # Get test trials
        test_trials = p_data[p_data['trial_type'] == 'test'].copy()
        
        # Get practice trials
        practice_trials = p_data[p_data['trial_type'] == 'practice'].copy()
        
        # Get time taken
        time_taken = demographic_df.loc[demographic_df['PROLIFIC_PID'] == pid, 'Time_taken_minutes'].values
        time_taken = time_taken[0] if len(time_taken) > 0 else np.nan
        
        # Initialize exclusion info with explicit boolean values for consistency
        exclusion_info = {
            'PROLIFIC_PID': pid,
            'overall_accuracy': np.nan,
            'practice_accuracy': np.nan,
            'max_consecutive_identical': 0,
            'percent_fast_trials': 0,
            'rt_sd': np.nan,
            'percent_missing_AD0': 0,
            'percent_missing_AD50': 0,
            'percent_missing_AD100': 0,
            'percent_missing_AD150': 0,
            'time_taken_minutes': time_taken,
            'negative_slope': False,  # Explicitly boolean
            'large_angle_accuracy': np.nan,
            'excluded': False,  # Explicitly boolean
            'exclusion_reason': "",
            'n_excluded_trials_too_fast': 0,
            'n_excluded_trials_too_slow': 0,
            'n_excluded_trials_outlier': 0
        }
        
        # Calculate overall accuracy for test trials
        if len(test_trials) > 0:
            test_accuracy = test_trials['MRT_correct'].mean()
            exclusion_info['overall_accuracy'] = test_accuracy
        
        # Calculate practice accuracy
        if len(practice_trials) > 0:
            practice_accuracy = practice_trials['MRT_correct'].mean()
            exclusion_info['practice_accuracy'] = practice_accuracy
        
        # Check for consecutive identical responses
        if len(test_trials) > 0:
            # Get response keys as a list
            responses = test_trials['MRT_response_key'].fillna('missing').tolist()
            
            # Count consecutive identical responses
            max_consecutive = 1
            current_consecutive = 1
            
            for i in range(1, len(responses)):
                if responses[i] == responses[i-1] and responses[i] != 'missing':
                    current_consecutive += 1
                else:
                    current_consecutive = 1
                    
                max_consecutive = max(max_consecutive, current_consecutive)
            
            exclusion_info['max_consecutive_identical'] = max_consecutive
        
        # Check for fast trials
        if len(test_trials) > 0:
            fast_trials = test_trials[test_trials['MRT_rt'] < 0.2]
            percent_fast = len(fast_trials) / len(test_trials) * 100
            exclusion_info['percent_fast_trials'] = percent_fast
        
        # Calculate RT SD
        if len(test_trials) > 0:
            rt_sd = test_trials['MRT_rt'].std()
            exclusion_info['rt_sd'] = rt_sd
        
        # Calculate missing trials by condition
        for angle in [0, 50, 100, 150]:
            condition_trials = test_trials[test_trials['angular_disparity'] == angle]
            total_trials = len(condition_trials)
            
            if total_trials > 0:
                # Check for NaN values, empty strings, and 'None' values
                missing_mask = condition_trials['MRT_response_key'].isna() | (condition_trials['MRT_response_key'] == '') | (condition_trials['MRT_response_key'] == 'None')
                missing_trials = condition_trials[missing_mask]
                percent_missing = len(missing_trials) / total_trials * 100
                exclusion_info[f'percent_missing_AD{angle}'] = percent_missing
                
                # Print detailed missing trial info for debugging
                if len(missing_trials) > 0:
                    print(f"  Found {len(missing_trials)} missing trials out of {total_trials} for participant {pid} in AD{angle} condition ({percent_missing:.2f}%)")
                    # Check if these trials actually exist in the original data
                    original_trials = p_data[(p_data['trial_type'] == 'test') & (p_data['angular_disparity'] == angle)]
                    original_missing = (original_trials['MRT_response_key'].isna() | 
                                       (original_trials['MRT_response_key'] == '') | 
                                       (original_trials['MRT_response_key'] == 'None')).sum()
                    if original_missing != len(missing_trials):
                        print(f"  WARNING: Mismatch in missing trial count for participant {pid} in AD{angle}: {original_missing} vs {len(missing_trials)}")
                
                # If missing percentage is high, add to exclusion reasons
                if percent_missing > 20:
                    if not exclusion_info['excluded']:
                        exclusion_info['excluded'] = True
                        exclusion_info['exclusion_reason'] = f"Missing > 20% of trials in AD{angle} condition"
                    elif f"Missing > 20% of trials in AD{angle} condition" not in exclusion_info['exclusion_reason']:
                        exclusion_info['exclusion_reason'] += f"; Missing > 20% of trials in AD{angle} condition"
            else:
                exclusion_info[f'percent_missing_AD{angle}'] = 0
            
            # Print for debugging
            print(f"Participant {pid}: AD{angle} - {len(missing_trials) if total_trials > 0 else 0}/{total_trials} missing trials ({exclusion_info[f'percent_missing_AD{angle}']:.2f}%)")
        
        # Calculate RT by angle slope
        if len(test_trials) > 0:
            # Get mean RTs for each angle
            angle_rts = {}
            for angle in [0, 50, 100, 150]:
                angle_data = test_trials[(test_trials['angular_disparity'] == angle) & 
                                        (test_trials['MRT_correct'] == 1)]
                if len(angle_data) > 0:
                    angle_rts[angle] = angle_data['MRT_rt'].mean()
            
            # If we have at least 2 angles with data, calculate slope
            if len(angle_rts) >= 2:
                angles = np.array(list(angle_rts.keys())).reshape(-1, 1)
                rts = np.array(list(angle_rts.values()))
                
                model = LinearRegression()
                model.fit(angles, rts)
                
                slope = model.coef_[0]
                exclusion_info['negative_slope'] = slope < 0
                
                # Calculate accuracy at larger angles
                large_angle_data = test_trials[(test_trials['angular_disparity'].isin([100, 150]))]
                if len(large_angle_data) > 0:
                    large_angle_accuracy = large_angle_data['MRT_correct'].mean()
                    exclusion_info['large_angle_accuracy'] = large_angle_accuracy
        
        # Apply exclusion criteria
        exclusion_reasons = []
        
        # 1. Overall accuracy < 60% (relaxed from 65% as suggested by expert feedback)
        if not np.isnan(exclusion_info['overall_accuracy']) and exclusion_info['overall_accuracy'] < 0.60:
            exclusion_reasons.append("Overall accuracy < 60%")
        
        # 2. Identical responses for > 8 consecutive trials
        if exclusion_info['max_consecutive_identical'] > 8:
            exclusion_reasons.append("Identical responses for > 8 consecutive trials")
        
        # 3. Fast trials > 10%
        if exclusion_info['percent_fast_trials'] > 10:
            exclusion_reasons.append("Fast trials (< 200ms) > 10%")
        
        # 4. RT SD < 120ms
        if not np.isnan(exclusion_info['rt_sd']) and exclusion_info['rt_sd'] < 0.120:
            exclusion_reasons.append("RT SD < 120ms")
        
        # 5. Missing trials > 20% in any condition
        for angle in [0, 50, 100, 150]:
            if exclusion_info[f'percent_missing_AD{angle}'] > 20:
                exclusion_reasons.append(f"Missing > 20% of trials in AD{angle} condition")
        
        # 6. Practice accuracy < 50%
        if not np.isnan(exclusion_info['practice_accuracy']) and exclusion_info['practice_accuracy'] < 0.5:
            exclusion_reasons.append("Practice accuracy < 50%")
        
        # 7. Time taken < 15 minutes
        if not np.isnan(exclusion_info['time_taken_minutes']) and exclusion_info['time_taken_minutes'] < 15:
            exclusion_reasons.append("Completed study in < 15 minutes")
        
        # Note: Removed negative slope + low accuracy exclusion criterion as suggested by expert feedback
        # This allows for participants who may use different mental rotation strategies
        
        # Set exclusion status and reason
        if exclusion_reasons:
            exclusion_info['excluded'] = True
            exclusion_info['exclusion_reason'] = "; ".join(exclusion_reasons)
            excluded_participants.append(pid)
        
        # Add to exclusion log
        exclusion_log.append(exclusion_info)
    
    # Create exclusion log dataframe
    exclusion_log_df = pd.DataFrame(exclusion_log)
    
    # Filter out excluded participants
    filtered_df = df[~df['PROLIFIC_PID'].isin(excluded_participants)].copy()
    
    # Print exclusion summary
    print(f"\nExclusion summary:")
    print(f"Total participants: {len(participant_ids)}")
    print(f"Excluded participants: {len(excluded_participants)}")
    print(f"Remaining participants: {len(participant_ids) - len(excluded_participants)}")
    
    # Print exclusion reasons
    if excluded_participants:
        reason_counts = exclusion_log_df[exclusion_log_df['excluded']]['exclusion_reason'].value_counts()
        print("\nExclusion reasons:")
        for reason, count in reason_counts.items():
            print(f"- {reason}: {count}")
        
        # Check for missing trial exclusions
        missing_exclusions = sum([1 for reason in exclusion_log_df['exclusion_reason'] if isinstance(reason, str) and 'Missing > 20%' in reason])
        print(f"Participants excluded due to missing trials: {missing_exclusions}")
    
    return filtered_df, exclusion_log_df, excluded_participants

def apply_trial_exclusions(mrt_df, exclusion_log_df=None):
    """
    Apply trial-level exclusion criteria.
    
    Parameters:
    -----------
    mrt_df : pandas.DataFrame
        MRT data after participant-level exclusions
    exclusion_log_df : pandas.DataFrame, optional
        Exclusion log dataframe to update with trial exclusion counts
        
    Returns:
    --------
    pandas.DataFrame
        MRT data with trial exclusions applied
    """
    print("Applying trial-level exclusion criteria...")
    
    # Create a copy of the data
    df = mrt_df.copy()
    
    # Initialize trial exclusion columns with explicit boolean type for consistency
    df['trial_excluded'] = False  # Use boolean instead of 0
    df['trial_exclusion_reason'] = ""
    df['n_excluded_trials_too_fast'] = 0
    df['n_excluded_trials_too_slow'] = 0
    df['n_excluded_trials_outlier'] = 0
    
    # Get unique participant IDs
    participant_ids = df['PROLIFIC_PID'].unique()
    
    # Initialize dictionary to track exclusion counts per participant
    participant_exclusion_counts = {pid: {'too_fast': 0, 'too_slow': 0, 'outlier': 0} for pid in participant_ids}
    
    # Initialize counters for summary
    total_test_trials = 0
    total_excluded_too_fast = 0
    total_excluded_too_slow = 0
    total_excluded_outliers = 0
    
    # Loop through each participant
    for pid in participant_ids:
        # Get participant data
        p_data = df[df['PROLIFIC_PID'] == pid]
        
        # Get test trials
        test_trials = p_data[p_data['trial_type'] == 'test']
        total_test_trials += len(test_trials)
        
        # 1. Exclude trials with RTs < 200ms or > 7,500ms
        too_fast_mask = (test_trials['MRT_rt'] < 0.2) & (~test_trials['MRT_rt'].isna())
        too_slow_mask = (test_trials['MRT_rt'] > 7.5) & (~test_trials['MRT_rt'].isna())
        
        # Update counters
        total_excluded_too_fast += too_fast_mask.sum()
        total_excluded_too_slow += too_slow_mask.sum()
        
        # Update exclusion columns
        df.loc[df['PROLIFIC_PID'] == pid, 'n_excluded_trials_too_fast'] = too_fast_mask.sum()
        df.loc[df['PROLIFIC_PID'] == pid, 'n_excluded_trials_too_slow'] = too_slow_mask.sum()
        
        df.loc[df.index[df['PROLIFIC_PID'] == pid].intersection(test_trials.index[too_fast_mask]), 'trial_excluded'] = True
        df.loc[df.index[df['PROLIFIC_PID'] == pid].intersection(test_trials.index[too_fast_mask]), 'trial_exclusion_reason'] = "RT < 200ms"
        
        df.loc[df.index[df['PROLIFIC_PID'] == pid].intersection(test_trials.index[too_slow_mask]), 'trial_excluded'] = True
        df.loc[df.index[df['PROLIFIC_PID'] == pid].intersection(test_trials.index[too_slow_mask]), 'trial_exclusion_reason'] = "RT > 7,500ms"
        
        # 2. Exclude trials with RTs > 2.5SD from participant's mean within each angular disparity condition
        participant_outliers = 0
        for angle in [0, 50, 100, 150]:
            angle_trials = test_trials[test_trials['angular_disparity'] == angle]
            
            if len(angle_trials) > 0:
                # Calculate mean and SD of RT
                rt_mean = angle_trials['MRT_rt'].mean()
                rt_sd = angle_trials['MRT_rt'].std()
                
                # Identify outliers
                outlier_mask = (np.abs(angle_trials['MRT_rt'] - rt_mean) > 2.5 * rt_sd) & (~angle_trials['MRT_rt'].isna())
                participant_outliers += outlier_mask.sum()
                
                # Update exclusion columns
                df.loc[df.index[df['PROLIFIC_PID'] == pid].intersection(angle_trials.index[outlier_mask]), 'trial_excluded'] = True
                df.loc[df.index[df['PROLIFIC_PID'] == pid].intersection(angle_trials.index[outlier_mask]), 'trial_exclusion_reason'] = f"RT outlier (> 2.5SD) in AD{angle}"
        
        # Update exclusion count for outliers
        df.loc[df['PROLIFIC_PID'] == pid, 'n_excluded_trials_outlier'] = participant_outliers
        total_excluded_outliers += participant_outliers
        
        # Update participant exclusion counts dictionary
        participant_exclusion_counts[pid]['too_fast'] = too_fast_mask.sum()
        participant_exclusion_counts[pid]['too_slow'] = too_slow_mask.sum()
        participant_exclusion_counts[pid]['outlier'] = participant_outliers
    
    # Print detailed exclusion summary
    print(f"\nDetailed trial exclusion summary:")
    print(f"Total test trials: {total_test_trials}")
    print(f"Trials excluded due to RT < 200ms: {total_excluded_too_fast} ({total_excluded_too_fast/total_test_trials*100:.2f}%)")
    print(f"Trials excluded due to RT > 7,500ms: {total_excluded_too_slow} ({total_excluded_too_slow/total_test_trials*100:.2f}%)")
    print(f"Trials excluded as RT outliers: {total_excluded_outliers} ({total_excluded_outliers/total_test_trials*100:.2f}%)")
    
    # Print exclusion reasons
    reason_counts = df[(df['trial_type'] == 'test') & (df['trial_excluded'])]['trial_exclusion_reason'].value_counts()
    print("\nTrial exclusion reasons:")
    for reason, count in reason_counts.items():
        print(f"- {reason}: {count}")
    
    # Update exclusion_log_df with trial exclusion counts if provided
    if exclusion_log_df is not None:
        for pid, counts in participant_exclusion_counts.items():
            exclusion_log_df.loc[exclusion_log_df['PROLIFIC_PID'] == pid, 'n_excluded_trials_too_fast'] = counts['too_fast']
            exclusion_log_df.loc[exclusion_log_df['PROLIFIC_PID'] == pid, 'n_excluded_trials_too_slow'] = counts['too_slow']
            exclusion_log_df.loc[exclusion_log_df['PROLIFIC_PID'] == pid, 'n_excluded_trials_outlier'] = counts['outlier']
    
    return df

def calculate_performance_metrics(mrt_df):
    """
    Calculate performance metrics for each participant and angular disparity.
    
    Parameters:
    -----------
    mrt_df : pandas.DataFrame
        MRT data with trial exclusions applied
        
    Returns:
    --------
    pandas.DataFrame
        Performance metrics
    """
    print("Calculating performance metrics...")
    
    # Verify input is a valid DataFrame
    if not isinstance(mrt_df, pd.DataFrame) or mrt_df.empty:
        print("WARNING: Input DataFrame is empty or invalid")
        return pd.DataFrame()
    
    # Verify required columns exist
    required_cols = ['PROLIFIC_PID', 'trial_type', 'trial_excluded', 'angular_disparity', 'MRT_correct', 'MRT_rt']
    missing_cols = [col for col in required_cols if col not in mrt_df.columns]
    if missing_cols:
        print(f"WARNING: Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Get test trials that are not excluded
    test_df = mrt_df[(mrt_df['trial_type'] == 'test') & (~mrt_df['trial_excluded'])].copy()
    
    # Check if we have any valid test trials
    if test_df.empty:
        print("WARNING: No valid test trials found after filtering")
        return pd.DataFrame()
    
    # Get unique participant IDs
    participant_ids = test_df['PROLIFIC_PID'].unique()
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Initialize counters for summary
    total_valid_trials = 0
    total_correct_trials = 0
    angle_valid_trials = {0: 0, 50: 0, 100: 0, 150: 0}
    angle_correct_trials = {0: 0, 50: 0, 100: 0, 150: 0}
    
    print(f"Calculating metrics for {len(participant_ids)} participants...")
    
    # Loop through each participant
    for pid in participant_ids:
        # Get participant data
        p_data = test_df[test_df['PROLIFIC_PID'] == pid]
        
        # Loop through each angular disparity
        for angle in [0, 50, 100, 150]:
            # Get trials for this angle
            angle_data = p_data[p_data['angular_disparity'] == angle]
            
            if len(angle_data) > 0:
                # Count total valid trials
                n_valid_trials = len(angle_data)
                total_valid_trials += n_valid_trials
                angle_valid_trials[angle] += n_valid_trials
                
                # Count correct responses
                n_correct_trials = angle_data['MRT_correct'].sum()
                total_correct_trials += n_correct_trials
                angle_correct_trials[angle] += n_correct_trials
                
                # Calculate accuracy
                accuracy = n_correct_trials / n_valid_trials
                
                # Get correct trials
                correct_trials = angle_data[angle_data['MRT_correct'] == 1]
                
                # Calculate mean RT for correct trials
                mean_rt_correct = correct_trials['MRT_rt'].mean() if len(correct_trials) > 0 else np.nan
                
                # Calculate SD of RT for correct trials
                rt_sd_correct = correct_trials['MRT_rt'].std() if len(correct_trials) > 0 else np.nan
                
                # Add to metrics list
                metrics_list.append({
                    'PROLIFIC_PID': pid,
                    'angular_disparity': angle,
                    'n_valid_trials': n_valid_trials,
                    'n_correct_trials': n_correct_trials,
                    'accuracy': accuracy,
                    'mean_rt_correct': mean_rt_correct,
                    'rt_sd_correct': rt_sd_correct
                })
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics_list)
    
    # Print detailed performance summary
    print(f"\nPerformance metrics summary:")
    print(f"Total valid trials: {total_valid_trials}")
    print(f"Total correct trials: {total_correct_trials}")
    if total_valid_trials > 0:  # Add check to prevent division by zero
        print(f"Overall accuracy: {total_correct_trials/total_valid_trials*100:.2f}%")
    else:
        print("Overall accuracy: N/A (no valid trials)")
    
    print("\nPerformance by angular disparity:")
    for angle in [0, 50, 100, 150]:
        if angle_valid_trials[angle] > 0:
            accuracy = angle_correct_trials[angle] / angle_valid_trials[angle] * 100
            print(f"  {angle}° rotation: {accuracy:.2f}% accuracy ({angle_correct_trials[angle]}/{angle_valid_trials[angle]} trials)")
    
    # Print sample of metrics
    print("\nSample of performance metrics (first 2 rows):")
    if len(metrics_df) > 0:  # Add check to prevent accessing empty DataFrame
        print(metrics_df.head(2))
    else:
        print("No metrics data available")
    
    return metrics_df

def calculate_regression_metrics(performance_metrics_df, exclusion_log_df):
    """
    Calculate RT-by-angle regression metrics for each participant.
    
    Parameters:
    -----------
    performance_metrics_df : pandas.DataFrame
        Performance metrics for participants who passed exclusion criteria
    exclusion_log_df : pandas.DataFrame
        Exclusion log containing all participants (both included and excluded)
        
    Returns:
    --------
    pandas.DataFrame
        Regression metrics with exclusion status from the original exclusion log
    """
    print("Calculating RT-by-angle regression metrics...")
    
    # Verify inputs are valid DataFrames
    if not isinstance(performance_metrics_df, pd.DataFrame) or performance_metrics_df.empty:
        print("WARNING: Performance metrics DataFrame is empty or invalid")
        return pd.DataFrame()
        
    if not isinstance(exclusion_log_df, pd.DataFrame) or exclusion_log_df.empty:
        print("WARNING: Exclusion log DataFrame is empty or invalid")
        return pd.DataFrame()
    
    # Verify required columns exist
    required_perf_cols = ['PROLIFIC_PID', 'angular_disparity', 'mean_rt_correct']
    missing_cols = [col for col in required_perf_cols if col not in performance_metrics_df.columns]
    if missing_cols:
        print(f"WARNING: Missing required columns in performance metrics: {missing_cols}")
        return pd.DataFrame()
        
    required_log_cols = ['PROLIFIC_PID', 'excluded', 'exclusion_reason']
    missing_cols = [col for col in required_log_cols if col not in exclusion_log_df.columns]
    if missing_cols:
        print(f"WARNING: Missing required columns in exclusion log: {missing_cols}")
        return pd.DataFrame()
    
    # Get unique participant IDs from both performance metrics and exclusion log
    perf_participant_ids = performance_metrics_df['PROLIFIC_PID'].unique()
    all_participant_ids = exclusion_log_df['PROLIFIC_PID'].unique()
    print(f"Performance metrics contains {len(perf_participant_ids)} participants")
    print(f"Exclusion log contains {len(all_participant_ids)} participants")
    
    # Initialize list to store regression metrics
    regression_metrics_list = []
    
    # Loop through all participants in the exclusion log
    for pid in all_participant_ids:
        # Get exclusion info
        exclusion_info = exclusion_log_df[exclusion_log_df['PROLIFIC_PID'] == pid]
        if not exclusion_info.empty:
            excluded = bool(exclusion_info['excluded'].iloc[0]) if not pd.isna(exclusion_info['excluded'].iloc[0]) else False
            exclusion_reason = str(exclusion_info['exclusion_reason'].iloc[0]) if not pd.isna(exclusion_info['exclusion_reason'].iloc[0]) else ""
            print(f"Participant {pid}: Exclusion status from log: {excluded}, Reason: {exclusion_reason[:30]}...")
        else:
            excluded = False
            exclusion_reason = ""
            print(f"WARNING: Participant {pid} not found in exclusion log")
        
        # Initialize regression metrics with exclusion status
        regression_metrics = {
            'PROLIFIC_PID': pid,
            'rt_by_angle_slope': np.nan,
            'rt_by_angle_intercept': np.nan,
            'rt_by_angle_r_squared': np.nan,
            'slope_p_value': np.nan,
            'slope_significant': False,
            'excluded': excluded,
            'exclusion_reason': exclusion_reason
        }
        
        # Calculate regression metrics for all participants in performance metrics
        if pid in perf_participant_ids:
            # Get participant data
            p_data = performance_metrics_df[performance_metrics_df['PROLIFIC_PID'] == pid]
        
            # Get data for regression
            angles = p_data['angular_disparity'].values
            rts = p_data['mean_rt_correct'].values
            
            # Remove any NaN values
            valid_indices = ~np.isnan(rts)
            angles = angles[valid_indices]
            rts = rts[valid_indices]
            
            # If we have at least 2 valid angles, perform regression
            if len(angles) >= 2:
                try:
                    # Reshape angles for sklearn
                    angles_reshaped = angles.reshape(-1, 1)
                    
                    # Fit linear regression model
                    model = LinearRegression()
                    model.fit(angles_reshaped, rts)
                
                    # Get slope and intercept
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    
                    # Calculate R-squared
                    r_squared = model.score(angles_reshaped, rts)
                    
                    # Calculate p-value for slope
                    n = len(angles)
                    if n > 2:
                        # Calculate predicted values
                        y_pred = model.predict(angles_reshaped)
                        
                        # Calculate residuals
                        residuals = rts - y_pred
                        
                        # Calculate residual sum of squares
                        rss = np.sum(residuals**2)
                        
                        # Calculate mean squared error
                        mse = rss / (n - 2)
                        
                        # Calculate standard error of slope
                        x_mean = np.mean(angles)
                        x_var = np.sum((angles - x_mean)**2)
                        
                        # Avoid division by zero
                        if x_var > 0:
                            se_slope = np.sqrt(mse / x_var)
                            
                            # Calculate t-statistic
                            t_stat = slope / se_slope
                            
                            # Calculate p-value
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                            
                            # Check if slope is significant
                            slope_significant = p_value < 0.05
                            
                            # Update regression metrics
                            regression_metrics['rt_by_angle_slope'] = slope
                            regression_metrics['rt_by_angle_intercept'] = intercept
                            regression_metrics['rt_by_angle_r_squared'] = r_squared
                            regression_metrics['slope_p_value'] = p_value
                            regression_metrics['slope_significant'] = slope_significant
                except Exception as e:
                    print(f"Error calculating regression for participant {pid}: {e}")
        
        # Add to regression metrics list
        regression_metrics_list.append(regression_metrics)
    
    # Create regression metrics dataframe
    regression_metrics_df = pd.DataFrame(regression_metrics_list)
    
    # Ensure excluded column is boolean and print summary of exclusion statuses
    regression_metrics_df['excluded'] = regression_metrics_df['excluded'].astype(bool)
    
    # Double-check that excluded status is correctly transferred from exclusion log
    for pid in all_participant_ids:
        if pid in exclusion_log_df['PROLIFIC_PID'].values and pid in regression_metrics_df['PROLIFIC_PID'].values:
            log_excluded = bool(exclusion_log_df.loc[exclusion_log_df['PROLIFIC_PID'] == pid, 'excluded'].iloc[0])
            metrics_excluded = bool(regression_metrics_df.loc[regression_metrics_df['PROLIFIC_PID'] == pid, 'excluded'].iloc[0])
            if log_excluded != metrics_excluded:
                print(f"WARNING: Exclusion status mismatch for {pid}: Log={log_excluded}, Metrics={metrics_excluded}")
                # Fix the mismatch
                regression_metrics_df.loc[regression_metrics_df['PROLIFIC_PID'] == pid, 'excluded'] = log_excluded
    
    # Print detailed exclusion status summary
    excluded_participants = regression_metrics_df[regression_metrics_df['excluded']]['PROLIFIC_PID'].tolist()
    print(f"\nDetailed exclusion status in regression metrics:")
    print(f"Total excluded participants: {len(excluded_participants)}")
    if len(excluded_participants) > 0:
        print(f"First 5 excluded participants: {excluded_participants[:5]}")
        
        # Verify these participants are actually marked as excluded in the original log
        for pid in excluded_participants[:5]:
            orig_status = exclusion_log_df.loc[exclusion_log_df['PROLIFIC_PID'] == pid, 'excluded'].iloc[0]
            print(f"  {pid}: Original exclusion status: {orig_status}")
    
    # Print summary
    print(f"\nRegression metrics summary:")
    print(f"Calculated metrics for {len(regression_metrics_df)} participants (including both excluded and non-excluded)")
    
    # Count excluded participants in regression metrics
    excluded_count = regression_metrics_df['excluded'].sum()
    print(f"Participants marked as excluded: {excluded_count}")
    print(f"Participants marked as included: {len(all_participant_ids) - excluded_count}")
    
    # Print sample of metrics
    print("\nSample of regression metrics (first 2 rows):")
    print(regression_metrics_df.head(2))
    
    return regression_metrics_df

def save_output_files(cleaned_data_df, performance_metrics_df, regression_metrics_df, exclusion_log_df):
    """
    Save output files with timestamps.
    
    Parameters:
    -----------
    cleaned_data_df : pandas.DataFrame
        Cleaned MRT data
    performance_metrics_df : pandas.DataFrame
        Performance metrics
    regression_metrics_df : pandas.DataFrame
        Regression metrics
    exclusion_log_df : pandas.DataFrame
        Exclusion log
        
    Returns:
    --------
    list
        List of paths to the saved output files
    """
    print("Saving output files...")
    output_files = []
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created outputs directory")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save cleaned data
    cleaned_data_file = f"outputs/MRT_cleaned_data_{timestamp}.csv"
    cleaned_data_columns = [
        'PROLIFIC_PID', 'TaskOrder', 'trial_type', 'angular_disparity', 'stimulus_type',
        'condition', 'MRT_response_key', 'MRT_correct', 'MRT_rt', 'trial_excluded',
        'trial_exclusion_reason'
    ]
    
    # Validate that all required columns exist in the dataframe
    missing_cols = [col for col in cleaned_data_columns if col not in cleaned_data_df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns in cleaned data: {missing_cols}")
        # Use only available columns
        available_cols = [col for col in cleaned_data_columns if col in cleaned_data_df.columns]
        cleaned_data_df[available_cols].to_csv(cleaned_data_file, index=False)
    else:
        cleaned_data_df[cleaned_data_columns].to_csv(cleaned_data_file, index=False)
    
    print(f"Saved cleaned data to {cleaned_data_file}")
    output_files.append(cleaned_data_file)
    
    # Save performance metrics
    performance_metrics_file = f"outputs/MRT_performance_metrics_{timestamp}.csv"
    performance_metrics_columns = [
        'PROLIFIC_PID', 'angular_disparity', 'n_valid_trials', 'n_correct_trials',
        'accuracy', 'mean_rt_correct', 'rt_sd_correct'
    ]
    
    # Validate that all required columns exist in the dataframe
    missing_cols = [col for col in performance_metrics_columns if col not in performance_metrics_df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns in performance metrics: {missing_cols}")
        # Use only available columns
        available_cols = [col for col in performance_metrics_columns if col in performance_metrics_df.columns]
        performance_metrics_df[available_cols].to_csv(performance_metrics_file, index=False)
    else:
        performance_metrics_df[performance_metrics_columns].to_csv(performance_metrics_file, index=False)
    
    print(f"Saved performance metrics to {performance_metrics_file}")
    output_files.append(performance_metrics_file)
    
    # Save regression metrics
    regression_metrics_file = f"outputs/MRT_regression_metrics_{timestamp}.csv"
    regression_metrics_columns = [
        'PROLIFIC_PID', 'rt_by_angle_slope', 'rt_by_angle_intercept', 'rt_by_angle_r_squared',
        'slope_p_value', 'slope_significant', 'excluded', 'exclusion_reason'
    ]
    
    # Validate that all required columns exist in the dataframe
    missing_cols = [col for col in regression_metrics_columns if col not in regression_metrics_df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns in regression metrics: {missing_cols}")
        # Use only available columns
        available_cols = [col for col in regression_metrics_columns if col in regression_metrics_df.columns]
        regression_metrics_df[available_cols].to_csv(regression_metrics_file, index=False)
    else:
        regression_metrics_df[regression_metrics_columns].to_csv(regression_metrics_file, index=False)
    
    print(f"Saved regression metrics to {regression_metrics_file}")
    output_files.append(regression_metrics_file)
    
    # Save exclusion log
    exclusion_log_file = f"outputs/MRT_exclusion_log_{timestamp}.csv"
    exclusion_log_columns = [
        'PROLIFIC_PID', 'overall_accuracy', 'practice_accuracy', 'max_consecutive_identical',
        'percent_fast_trials', 'rt_sd', 'percent_missing_AD0', 'percent_missing_AD50',
        'percent_missing_AD100', 'percent_missing_AD150', 'time_taken_minutes',
        'negative_slope', 'large_angle_accuracy', 'excluded', 'exclusion_reason',
        'n_excluded_trials_too_fast', 'n_excluded_trials_too_slow', 'n_excluded_trials_outlier'
    ]
    
    # Validate that all required columns exist in the dataframe
    missing_cols = [col for col in exclusion_log_columns if col not in exclusion_log_df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns in exclusion log: {missing_cols}")
        # Use only available columns
        available_cols = [col for col in exclusion_log_columns if col in exclusion_log_df.columns]
        exclusion_log_df[available_cols].to_csv(exclusion_log_file, index=False)
    else:
        exclusion_log_df[exclusion_log_columns].to_csv(exclusion_log_file, index=False)
    
    print(f"Saved exclusion log to {exclusion_log_file}")
    output_files.append(exclusion_log_file)
    
    return output_files

def main():
    """
    Main function to execute the MRT data processing pipeline.
    
    The pipeline follows these steps:
    1. Load and prepare MRT data
    2. Apply participant-level exclusions
    3. Apply trial-level exclusions
    4. Calculate performance metrics
    5. Calculate regression metrics using the full exclusion log
    6. Save output files
    
    Note: The regression metrics file includes exclusion status from participant-level
    exclusions, even though the metrics are only calculated for non-excluded participants.
    The exclusion status is taken directly from the original exclusion log to ensure accuracy.
    
    Exclusion criteria:
    - Overall accuracy < 60% (relaxed from 65% based on expert feedback)
    - Practice accuracy < 50%
    - Identical responses for > 8 consecutive trials
    - Fast trials (< 200ms) > 10%
    - RT SD < 120ms
    - Missing trials > 20% in any condition
    - Time taken < 15 minutes
    
    Trial exclusion criteria:
    - RT < 200ms
    - RT > 7,500ms
    - RT > 2.5 SD from participant's mean within each angular disparity condition
    """
    try:
        print("Starting Mental Rotation Task data processing...")
        
        # Load MRT data
        mrt_file_pattern = "../C_results/data/PARTICIPANT_VisualArraysMentalRotation_*.csv"
        mrt_df = load_mental_rotation_data(mrt_file_pattern)
        
        if mrt_df is None:
            print("Error: Failed to load Mental Rotation Task data")
            return 1
        
        # Load demographic data
        demographic_df = load_demographic_data()
        
        if demographic_df is None:
            print("Error: Failed to load demographic data")
            return 1
        
        # Merge data
        print("\nMerging MRT data with demographic data...")
        
        # Get participant IDs from demographic data
        included_pids = demographic_df['PROLIFIC_PID'].unique()
        print(f"Found {len(included_pids)} included participants in demographic data")
        
        # Filter MRT data to only include participants in demographic data
        merged_df = mrt_df[mrt_df['PROLIFIC_PID'].isin(included_pids)].copy()
        print(f"After merging, MRT data has {merged_df.shape[0]} rows for {merged_df['PROLIFIC_PID'].nunique()} participants")
        
        # Prepare MRT data
        prepared_df = prepare_mrt_data(merged_df)
        
        # Apply participant-level exclusions
        filtered_df, exclusion_log_df, excluded_participants = apply_participant_exclusions(prepared_df, demographic_df)
        
        # Apply trial-level exclusions and update exclusion log
        cleaned_df = apply_trial_exclusions(filtered_df, exclusion_log_df)
        
        # Verify exclusion counts were properly updated in exclusion log
        total_fast = exclusion_log_df['n_excluded_trials_too_fast'].sum()
        total_slow = exclusion_log_df['n_excluded_trials_too_slow'].sum()
        total_outliers = exclusion_log_df['n_excluded_trials_outlier'].sum()
        print(f"\nVerifying exclusion counts in exclusion log:")
        print(f"Total trials excluded as too fast: {total_fast}")
        print(f"Total trials excluded as too slow: {total_slow}")
        print(f"Total trials excluded as outliers: {total_outliers}")
        
        # Verify exclusion status in exclusion log
        total_excluded = exclusion_log_df['excluded'].sum()
        print(f"\nVerifying participant exclusion status in exclusion log:")
        print(f"Total participants marked as excluded: {total_excluded}")
        print(f"Total participants marked as included: {len(exclusion_log_df) - total_excluded}")
        
        # Print sample of exclusion log
        print("\nSample of exclusion log (first 2 rows):")
        print(exclusion_log_df[['PROLIFIC_PID', 'excluded', 'exclusion_reason']].head(2))
        
        # Calculate performance metrics
        performance_metrics_df = calculate_performance_metrics(cleaned_df)
        
        # Calculate regression metrics for all participants
        regression_metrics_df = calculate_regression_metrics(performance_metrics_df, exclusion_log_df)
        
        # Verify regression metrics exclusion status
        excluded_in_log = exclusion_log_df['excluded'].sum()
        excluded_in_metrics = regression_metrics_df['excluded'].sum()
        print(f"\nVerifying regression metrics exclusion status:")
        print(f"Participants marked as excluded in exclusion log: {excluded_in_log}")
        print(f"Participants marked as excluded in regression metrics: {excluded_in_metrics}")
        
        # If there's a mismatch, print details and fix it
        if excluded_in_log != excluded_in_metrics:
            print("WARNING: Mismatch between exclusion log and regression metrics!")
            
            # Find participants that are excluded in log but not in metrics
            for pid in exclusion_log_df[exclusion_log_df['excluded']]['PROLIFIC_PID']:
                if pid in regression_metrics_df['PROLIFIC_PID'].values:
                    metrics_excluded = regression_metrics_df.loc[regression_metrics_df['PROLIFIC_PID'] == pid, 'excluded'].iloc[0]
                    if not metrics_excluded:
                        print(f"  Participant {pid} is excluded in log but not in metrics - fixing...")
                        regression_metrics_df.loc[regression_metrics_df['PROLIFIC_PID'] == pid, 'excluded'] = True
                        # Also copy the exclusion reason
                        exclusion_reason = exclusion_log_df.loc[exclusion_log_df['PROLIFIC_PID'] == pid, 'exclusion_reason'].iloc[0]
                        regression_metrics_df.loc[regression_metrics_df['PROLIFIC_PID'] == pid, 'exclusion_reason'] = exclusion_reason
            
            # Recount after fixes
            excluded_in_metrics = regression_metrics_df['excluded'].sum()
            print(f"After fixes: {excluded_in_metrics} participants marked as excluded in regression metrics")
            
            # Print some examples of excluded participants for verification
            print("\nSample of excluded participants in regression metrics:")
            excluded_sample = regression_metrics_df[regression_metrics_df['excluded']].head(3)
            for _, row in excluded_sample.iterrows():
                print(f"  {row['PROLIFIC_PID']}: {row['exclusion_reason'][:50]}...")
        
        # Save output files
        output_files = save_output_files(cleaned_df, performance_metrics_df, regression_metrics_df, exclusion_log_df)
        
        # Verify output files exist
        for file_path in output_files:
            if not os.path.exists(file_path):
                print(f"ERROR: Output file {file_path} was not created!")
                return 1
            else:
                print(f"Successfully created: {file_path}")
        
        # Print missing trial stats for verification
        print(f"\nMissing trial percentages:")
        for angle in [0, 50, 100, 150]:
            # Check if column exists before accessing
            if f'percent_missing_AD{angle}' in exclusion_log_df.columns:
                missing_pcts = exclusion_log_df[f'percent_missing_AD{angle}']
                if not missing_pcts.empty:
                    print(f"AD{angle}: Min={missing_pcts.min():.2f}%, Max={missing_pcts.max():.2f}%, Mean={missing_pcts.mean():.2f}%")
                    print(f"Participants with >20% missing in AD{angle}: {(missing_pcts > 20).sum()}")
                
                    # Print examples of participants with high missing rates for verification
                    high_missing = exclusion_log_df[exclusion_log_df[f'percent_missing_AD{angle}'] > 10]
                    if len(high_missing) > 0:
                        print(f"  Example participants with >10% missing in AD{angle}:")
                        for _, row in high_missing.head(2).iterrows():
                            print(f"    {row['PROLIFIC_PID']}: {row[f'percent_missing_AD{angle}']:.2f}%")
                else:
                    print(f"AD{angle}: No data available")
            else:
                print(f"AD{angle}: Missing column 'percent_missing_AD{angle}' in exclusion log")
        
            # Print participants with any missing trials in this condition
            if f'percent_missing_AD{angle}' in exclusion_log_df.columns:
                any_missing = exclusion_log_df[exclusion_log_df[f'percent_missing_AD{angle}'] > 0]
                print(f"  Total participants with any missing trials in AD{angle}: {len(any_missing)}")
                if len(any_missing) > 0:
                    print(f"  First 3 participants with missing trials in AD{angle}:")
                    for _, row in any_missing.head(3).iterrows():
                        print(f"    {row['PROLIFIC_PID']}: {row[f'percent_missing_AD{angle}']:.2f}%")
            else:
                print(f"  Cannot check for missing trials in AD{angle}: column not found")
        
            # Check for systematic differences in trial counts across conditions
            if 'trial_type' in cleaned_df.columns and 'angular_disparity' in cleaned_df.columns:
                condition_counts = cleaned_df[cleaned_df['trial_type'] == 'test'].groupby(['PROLIFIC_PID', 'angular_disparity']).size()
                if len(condition_counts) > 0:
                    try:
                        condition_counts_df = condition_counts.unstack(fill_value=0)
                        if angle in condition_counts_df.columns:
                            print(f"\n  Trial count distribution for AD{angle}:")
                            count_stats = condition_counts_df[angle].describe()
                            print(f"    Min: {count_stats['min']:.0f}, Max: {count_stats['max']:.0f}, Mean: {count_stats['mean']:.2f}, Std: {count_stats['std']:.2f}")
                        
                            # Check for participants with unusually low trial counts
                            low_count_participants = condition_counts_df[condition_counts_df[angle] < count_stats['mean'] - count_stats['std']]
                            if len(low_count_participants) > 0:
                                print(f"    Participants with unusually low trial counts in AD{angle}: {len(low_count_participants)}")
                                print(f"    First 3 examples:")
                                for pid, row in low_count_participants.head(3).iterrows():
                                    print(f"      {pid}: {row[angle]:.0f} trials")
                        else:
                            print(f"\n  Trial count distribution for AD{angle}: No data available for this angle")
                    except Exception as e:
                        print(f"\n  Error analyzing trial counts for AD{angle}: {e}")
                else:
                    print(f"\n  Trial count distribution for AD{angle}: No trials found")
            else:
                print(f"\n  Cannot analyze trial counts: required columns missing")
        
        # Print final summary statistics
        participant_ids = cleaned_df['PROLIFIC_PID'].unique()
        print("\nFinal summary statistics:")
        print(f"Total participants: {len(included_pids)}")
        print(f"Participants after exclusions: {len(participant_ids)}")
        exclusion_rate = (len(included_pids) - len(participant_ids)) / len(included_pids) * 100
        print(f"Exclusion rate: {exclusion_rate:.2f}%")
        print(f"Total test trials: {len(cleaned_df[cleaned_df['trial_type'] == 'test'])}")
        valid_test_trials = len(cleaned_df[(cleaned_df['trial_type'] == 'test') & (~cleaned_df['trial_excluded'])])
        print(f"Valid test trials after exclusions: {valid_test_trials}")
        
        # Add more detailed summary statistics
        print("\nDetailed summary statistics:")
        
        # Accuracy by angular disparity
        print("Accuracy by angular disparity:")
        for angle in [0, 50, 100, 150]:
            angle_data = performance_metrics_df[performance_metrics_df['angular_disparity'] == angle]
            if len(angle_data) > 0:
                mean_accuracy = angle_data['accuracy'].mean() * 100
                std_accuracy = angle_data['accuracy'].std() * 100
                print(f"  AD{angle}: Mean={mean_accuracy:.2f}%, SD={std_accuracy:.2f}%")
        
        # RT by angular disparity
        print("\nRT by angular disparity (correct trials only):")
        for angle in [0, 50, 100, 150]:
            angle_data = performance_metrics_df[performance_metrics_df['angular_disparity'] == angle]
            if len(angle_data) > 0:
                mean_rt = angle_data['mean_rt_correct'].mean() * 1000  # Convert to ms
                std_rt = angle_data['mean_rt_correct'].std() * 1000  # Convert to ms
                print(f"  AD{angle}: Mean={mean_rt:.2f}ms, SD={std_rt:.2f}ms")
        
        # Regression metrics summary
        print("\nRT-by-angle regression metrics summary:")
        if 'excluded' in regression_metrics_df.columns:
            included_metrics = regression_metrics_df[~regression_metrics_df['excluded']]
            if len(included_metrics) > 0:
                # Check if all required columns exist
                required_cols = ['rt_by_angle_slope', 'rt_by_angle_intercept', 'rt_by_angle_r_squared', 'slope_significant']
                missing_cols = [col for col in required_cols if col not in included_metrics.columns]
                
                if not missing_cols:
                    # Handle potential NaN values in calculations
                    valid_slopes = included_metrics['rt_by_angle_slope'].dropna()
                    valid_intercepts = included_metrics['rt_by_angle_intercept'].dropna()
                    valid_r_squared = included_metrics['rt_by_angle_r_squared'].dropna()
                    
                    if len(valid_slopes) > 0 and len(valid_intercepts) > 0 and len(valid_r_squared) > 0:
                        mean_slope = valid_slopes.mean() * 1000  # Convert to ms/degree
                        std_slope = valid_slopes.std() * 1000  # Convert to ms/degree
                        mean_intercept = valid_intercepts.mean() * 1000  # Convert to ms
                        std_intercept = valid_intercepts.std() * 1000  # Convert to ms
                        mean_r_squared = valid_r_squared.mean()
                        std_r_squared = valid_r_squared.std()
                        
                        # Handle potential non-boolean values in slope_significant
                        if included_metrics['slope_significant'].dtype == bool:
                            significant_count = included_metrics['slope_significant'].sum()
                        else:
                            significant_count = included_metrics['slope_significant'].astype(bool).sum()
                            
                        significant_percent = significant_count / len(included_metrics) * 100
                        
                        print(f"  Slope: Mean={mean_slope:.2f}ms/degree, SD={std_slope:.2f}ms/degree")
                        print(f"  Intercept: Mean={mean_intercept:.2f}ms, SD={std_intercept:.2f}ms")
                        print(f"  R-squared: Mean={mean_r_squared:.4f}, SD={std_r_squared:.4f}")
                        print(f"  Significant slopes: {significant_count}/{len(included_metrics)} ({significant_percent:.2f}%)")
                    else:
                        print("  Cannot calculate regression metrics summary: too many NaN values")
                else:
                    print(f"  Cannot calculate regression metrics summary: missing columns {missing_cols}")
            else:
                print("  No included participants with regression metrics")
        else:
            print("  Cannot calculate regression metrics summary: 'excluded' column not found")
        
        print("Finished execution")
        return 0
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
