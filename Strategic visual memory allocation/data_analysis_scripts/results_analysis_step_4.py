import pandas as pd
import numpy as np
import re
import os
import glob
from datetime import datetime
import scipy.stats as stats
from scipy.stats import zscore
import pingouin as pg

def get_latest_file(pattern):
    """
    Gets the most recent file matching the given pattern.
    
    Args:
        pattern (str): File pattern to match, may include path.
        
    Returns:
        str: Path to the most recent file matching the pattern.
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    # Get the most recent file based on creation time
    latest_file = max(files, key=os.path.getctime)
    print(f"Latest file matching {pattern}: {latest_file}")
    return latest_file

def extract_numeric_score(response):
    """
    Extracts the numeric score (1-5) from a VVIQ2 item response.
    
    Args:
        response (str): The full response string ending with a numeric score.
        
    Returns:
        int or None: The extracted numeric score as an integer, or None if invalid.
    """
    if pd.isna(response):
        return None
    
    # Strip whitespace and extract the last character which should be the score
    response_str = str(response).strip()
    match = re.search(r'(\d)$', response_str)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
        else:
            print(f"Warning: Invalid score value: {score} in response: {response_str}")
            return None
    else:
        print(f"Warning: Could not extract score from response: {response_str}")
        return None

def calculate_missing_percentage(row, score_columns):
    """
    Calculates the percentage of missing VVIQ2 items for a participant.
    
    Args:
        row (pandas.Series): Row of data for a participant.
        score_columns (list): List of columns containing VVIQ2 scores.
        
    Returns:
        float: Percentage of missing items (0-100).
    """
    missing_count = sum(pd.isna(row[col]) for col in score_columns)
    return (missing_count / len(score_columns)) * 100

def calculate_identical_response_percentage(row, score_columns):
    """
    Calculates the percentage of identical responses across VVIQ2 items.
    
    Args:
        row (pandas.Series): Row of data for a participant.
        score_columns (list): List of columns containing VVIQ2 scores.
        
    Returns:
        float: Percentage of identical responses (0-100).
    """
    valid_scores = [row[col] for col in score_columns if not pd.isna(row[col])]
    if not valid_scores:
        return 0
    
    # Count occurrences of the most common value
    most_common_value = max(set(valid_scores), key=valid_scores.count)
    most_common_count = valid_scores.count(most_common_value)
    
    return (most_common_count / len(valid_scores)) * 100

def handle_missing_values(data, score_columns):
    """
    Imputes missing values if ≤10% missing (≤3 items) per participant by calculating
    scenario-specific means; records imputed item numbers in 'imputed_items'.
    
    Args:
        data (pandas.DataFrame): DataFrame containing participant data.
        score_columns (list): List of columns containing VVIQ2 scores.
        
    Returns:
        pandas.DataFrame: DataFrame with missing values imputed or rows removed if >10% missing.
    """
    try:
        # Count missing values per participant
        missing_counts = data[score_columns].isna().sum(axis=1)
        print(f"Missing value distribution: {missing_counts.value_counts().to_dict()}")
        
        # Initialize imputed_items column with empty strings
        data['imputed_items'] = ''
        print("Added 'imputed_items' column to track imputed items")
        
        # Define scenarios (each with 4 items)
        scenarios = {
            'familiar_person': score_columns[0:4],
            'sunrise': score_columns[4:8],
            'shop_front': score_columns[8:12],
            'countryside': score_columns[12:16],
            'driving': score_columns[16:20],
            'beach': score_columns[20:24],
            'railway_station': score_columns[24:28],
            'garden': score_columns[28:32]
        }
        
        # Process each participant
        for idx, row in data.iterrows():
            missing_items = [i for i, col in enumerate(score_columns) if pd.isna(row[col])]
            n_missing = len(missing_items)
            
            # If ≤10% missing (≤3 items), impute using scenario means
            if 0 < n_missing <= 3:
                imputed_item_numbers = []
                
                for item_idx in missing_items:
                    # Determine which scenario this item belongs to
                    scenario_idx = item_idx // 4
                    scenario_cols = score_columns[scenario_idx*4:(scenario_idx+1)*4]
                    
                    # Get the scenario name for better logging
                    scenario_name = list(scenarios.keys())[scenario_idx]
                    print(f"Item {item_idx+1} belongs to scenario: {scenario_name}")
                    
                    # Check if entire scenario is missing
                    if row[scenario_cols].isna().all():
                        print(f"Participant {row['PROLIFIC_PID']} missing entire scenario - cannot impute")
                        continue
                    
                    # Calculate mean of non-missing items in the same scenario
                    scenario_mean = row[scenario_cols].mean(skipna=True)
                    if not pd.isna(scenario_mean):
                        # Impute the missing value with the scenario mean
                        data.at[idx, score_columns[item_idx]] = scenario_mean
                        # Record the imputed item number (1-based)
                        imputed_item_numbers.append(str(item_idx + 1))
                        print(f"Imputed item{item_idx+1}_score for participant {row['PROLIFIC_PID']} with value {scenario_mean:.2f}")
                
                # Update imputed_items column with comma-separated list of imputed item numbers
                if imputed_item_numbers:
                    data.at[idx, 'imputed_items'] = ','.join(imputed_item_numbers)
                    print(f"Recorded imputed items for participant {row['PROLIFIC_PID']}: {data.at[idx, 'imputed_items']}")
        
        # Identify rows that still have missing values after imputation
        rows_with_missing = data[score_columns].isna().any(axis=1)
        missing_count = rows_with_missing.sum()
        
        if missing_count > 0:
            print(f"After imputation, {missing_count} participants still have missing values")
            # List the participant IDs with missing values
            missing_pids = data.loc[rows_with_missing, 'PROLIFIC_PID'].tolist()
            print(f"Participant IDs with missing values after imputation: {missing_pids}")
            
            # Remove rows that still have missing values
            data_filtered = data[~rows_with_missing].copy()
            print(f"Removed {missing_count} participants with >10% missing values or entire missing scenarios")
            print(f"Data shape after handling missing values: {data_filtered.shape}")
            return data_filtered
        else:
            print("All missing values successfully imputed")
            return data
    
    except Exception as e:
        print(f"Error in handle_missing_values: {e}")
        # Return the original data if handling fails
        return data

def calculate_scenario_scores(data, score_columns):
    """
    Calculates total and subscale scores for the VVIQ2.
    
    Args:
        data (pandas.DataFrame): DataFrame containing participant data.
        score_columns (list): List of columns containing VVIQ2 scores.
        
    Returns:
        pandas.DataFrame: DataFrame with calculated scores.
    """
    # Create a new DataFrame for scores
    scores_df = pd.DataFrame({'PROLIFIC_PID': data['PROLIFIC_PID']})
    
    # Calculate total score
    scores_df['total_score'] = data[score_columns].sum(axis=1)
    
    # Calculate subscale scores (4 items each)
    scores_df['familiar_person_score'] = data[score_columns[0:4]].sum(axis=1)
    scores_df['sunrise_score'] = data[score_columns[4:8]].sum(axis=1)
    scores_df['shop_front_score'] = data[score_columns[8:12]].sum(axis=1)
    scores_df['countryside_score'] = data[score_columns[12:16]].sum(axis=1)
    scores_df['driving_score'] = data[score_columns[16:20]].sum(axis=1)
    scores_df['beach_score'] = data[score_columns[20:24]].sum(axis=1)
    scores_df['railway_station_score'] = data[score_columns[24:28]].sum(axis=1)
    scores_df['garden_score'] = data[score_columns[28:32]].sum(axis=1)
    
    # Calculate z-scores
    scores_df['total_score_z'] = zscore(scores_df['total_score'], nan_policy='omit')
    scores_df['familiar_person_z'] = zscore(scores_df['familiar_person_score'], nan_policy='omit')
    scores_df['sunrise_z'] = zscore(scores_df['sunrise_score'], nan_policy='omit')
    scores_df['shop_front_z'] = zscore(scores_df['shop_front_score'], nan_policy='omit')
    scores_df['countryside_z'] = zscore(scores_df['countryside_score'], nan_policy='omit')
    scores_df['driving_z'] = zscore(scores_df['driving_score'], nan_policy='omit')
    scores_df['beach_z'] = zscore(scores_df['beach_score'], nan_policy='omit')
    scores_df['railway_station_z'] = zscore(scores_df['railway_station_score'], nan_policy='omit')
    scores_df['garden_z'] = zscore(scores_df['garden_score'], nan_policy='omit')
    
    # Add exclusion columns
    scores_df['excluded'] = data['excluded'].copy()
    scores_df['exclusion_reason'] = data['exclusion_reason'].copy()
    
    # Fix any NaN values in exclusion_reason to be empty strings
    scores_df['exclusion_reason'] = scores_df['exclusion_reason'].fillna('')
    
    # Ensure excluded participants have a reason
    scores_df.loc[(scores_df['excluded'] == True) & 
                 (scores_df['exclusion_reason'] == ''), 'exclusion_reason'] = 'Excluded by criteria'
    
    # Validate ranges
    total_min, total_max = scores_df['total_score'].min(), scores_df['total_score'].max()
    print(f"Total score range: {total_min}-{total_max} (expected: 32-160)")
    
    subscale_cols = [col for col in scores_df.columns if col.endswith('_score') and col != 'total_score']
    for col in subscale_cols:
        min_val, max_val = scores_df[col].min(), scores_df[col].max()
        print(f"{col} range: {min_val}-{max_val} (expected: 4-20)")
    
    return scores_df

def calculate_reliability(data, score_columns):
    """
    Calculates reliability metrics (Cronbach's alpha) for VVIQ2 scales.
    
    Args:
        data (pandas.DataFrame): DataFrame containing participant data.
        score_columns (list): List of columns containing VVIQ2 scores.
        
    Returns:
        pandas.DataFrame: DataFrame with reliability metrics.
    """
    # Initialize reliability DataFrame
    reliability_df = pd.DataFrame(columns=['scale_name', 'cronbachs_alpha', 'n_items', 'n_participants'])
    
    # Calculate Cronbach's alpha for all 32 items
    valid_data = data[~data['excluded']][score_columns]
    n_participants = len(valid_data)
    
    # Calculate alpha for total scale
    try:
        alpha_total = pg.cronbach_alpha(data=valid_data)[0]
        # Replace deprecated append with concat
        new_row = pd.DataFrame({
            'scale_name': ['total_scale'],
            'cronbachs_alpha': [alpha_total],
            'n_items': [32],
            'n_participants': [n_participants]
        })
        reliability_df = pd.concat([reliability_df, new_row], ignore_index=True)
        print(f"Cronbach's alpha for total scale: {alpha_total:.3f}")
    except Exception as e:
        print(f"Error calculating alpha for total scale: {e}")
    
    # Calculate alpha for each subscale
    subscales = {
        'familiar_person': score_columns[0:4],
        'sunrise': score_columns[4:8],
        'shop_front': score_columns[8:12],
        'countryside': score_columns[12:16],
        'driving': score_columns[16:20],
        'beach': score_columns[20:24],
        'railway_station': score_columns[24:28],
        'garden': score_columns[28:32]
    }
    
    for name, cols in subscales.items():
        try:
            alpha = pg.cronbach_alpha(data=valid_data[cols])[0]
            # Replace deprecated append with concat
            new_row = pd.DataFrame({
                'scale_name': [name],
                'cronbachs_alpha': [alpha],
                'n_items': [4],
                'n_participants': [n_participants]
            })
            reliability_df = pd.concat([reliability_df, new_row], ignore_index=True)
            print(f"Cronbach's alpha for {name}: {alpha:.3f}")
        except Exception as e:
            print(f"Error calculating alpha for {name}: {e}")
    
    return reliability_df

def littles_mcar_test(data, score_columns):
    """
    Performs Little's MCAR test to determine if missing data is completely at random.
    
    Args:
        data (pandas.DataFrame): DataFrame containing participant data.
        score_columns (list): List of columns containing VVIQ2 scores.
        
    Returns:
        float: p-value from Little's MCAR test
    """
    try:
        # Create a binary matrix indicating missing values
        missing_matrix = data[score_columns].isna().astype(int)
        
        # Count patterns of missingness
        pattern_counts = missing_matrix.value_counts()
        
        # If there are no missing values, return 1.0 (perfectly random)
        if pattern_counts.shape[0] == 1 and pattern_counts.index[0] == tuple([0] * len(score_columns)):
            print("No missing values detected in VVIQ2 items")
            return 1.0
        
        print("NOTE: This is a simplified approximation of Little's MCAR test")
        total_missing = missing_matrix.sum().sum()
        print(f"Missing data summary: {total_missing} missing values across {len(score_columns)} items")
        print(f"Missing patterns detected: {pattern_counts.shape[0]} unique patterns")
        
        # Print detailed information about missing data patterns
        missing_by_item = missing_matrix.sum(axis=0)
        missing_by_participant = missing_matrix.sum(axis=1)
        
        print(f"Items with most missing values:")
        for i, (col, count) in enumerate(missing_by_item.sort_values(ascending=False).head(5).items()):
            if count > 0:
                print(f"  - {col}: {count} missing values")
        
        print(f"Distribution of missing values per participant:")
        for count, n_participants in missing_by_participant.value_counts().sort_index().items():
            if count > 0:
                print(f"  - {count} missing values: {n_participants} participants")
        
        # More robust implementation using correlation between missingness patterns
        try:
            # Calculate correlation matrix of missingness indicators
            corr_matrix = missing_matrix.corr()
            # Average absolute correlation as a measure of non-randomness
            # Handle case where correlation can't be calculated (e.g., no variation in some columns)
            if np.isnan(corr_matrix.values).all():
                print("Could not calculate correlation between missing patterns (no variation)")
                avg_corr = 0
            else:
                # Filter out NaN values before calculating mean
                valid_corrs = np.abs(np.triu(corr_matrix, k=1))
                valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
                avg_corr = np.mean(valid_corrs) if len(valid_corrs) > 0 else 0
            print(f"Average correlation between missing patterns: {avg_corr:.4f}")
        except Exception as e:
            print(f"Error calculating correlation between missing patterns: {e}")
            avg_corr = 0
        
        # Simple approximation of Little's MCAR test using chi-square test
        observed_missing = missing_matrix.sum(axis=0)
        expected_missing = np.mean(observed_missing) * np.ones(len(score_columns))
        
        chi2, p = stats.chisquare(observed_missing, expected_missing)
        print(f"Little's MCAR test approximation: chi2={chi2:.2f}, p={p:.4f}")
        print(f"Missing pattern {'appears random' if p > 0.05 else 'may be systematic'}")
        print(f"Interpretation: {'Missing data appears to be missing completely at random (MCAR)' if p > 0.05 else 'Missing data may not be missing completely at random'}")
        
        return p
    except Exception as e:
        print(f"Error performing Little's MCAR test: {e}")
        return None

def main():
    """
    Main function to process VVIQ2 questionnaire data.
    """
    print("Starting VVIQ2 data processing...")
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # 1. Import and merge data
        # Import VVIQ2 questionnaire data
        vviq2_file = "../C_results/data/Vividness of Visual Imagery Questionnaire-2.csv"
        print(f"Reading VVIQ2 data from: {vviq2_file}")
        
        try:
            vviq2_data = pd.read_csv(vviq2_file)
            # Strip whitespace from column names to fix column name mismatches
            vviq2_data.columns = [col.strip() if isinstance(col, str) else col for col in vviq2_data.columns]
            print(f"VVIQ2 data shape: {vviq2_data.shape}")
            print("VVIQ2 columns (after stripping whitespace):")
            for col in vviq2_data.columns:
                print(f"  - {col}")
            print("First 3 rows of VVIQ2 data:")
            print(vviq2_data.head(3))
            
            # Filter out any rows where PROLIFIC_PID is not a valid participant ID
            vviq2_data = vviq2_data[vviq2_data['PROLIFIC_PID'] != 'Open-Ended Response']
            print(f"After filtering invalid PROLIFIC_PID rows: {vviq2_data.shape}")
        except Exception as e:
            print(f"Error reading VVIQ2 file: {e}")
            return 1
        
        # Import demographic data
        try:
            demo_pattern = "outputs/demographic_data_cleaned_*.csv"
            demo_file = get_latest_file(demo_pattern)
            print(f"Reading demographic data from: {demo_file}")
            
            demo_data = pd.read_csv(demo_file)
            # Strip whitespace from column names for consistency
            demo_data.columns = [col.strip() if isinstance(col, str) else col for col in demo_data.columns]
            print(f"Demographic data shape: {demo_data.shape}")
            print("Demographic data columns (after stripping whitespace):")
            for col in demo_data.columns:
                print(f"  - {col}")
            print("First 3 rows of demographic data:")
            print(demo_data.head(3))
        except Exception as e:
            print(f"Error reading demographic file: {e}")
            return 1
        
        # Verify required columns exist
        required_vviq2_columns = [
            'PROLIFIC_PID',
            'The exact contour of face, head, shoulders and body.',
            'Characteristic poses of head, attitudes of body etc.',
            'The precise carriage, length of step, etc. in walking.',
            'The different colours worn in some familiar clothes.',
            'The sun is rising above the horizon into a hazy sky.',
            'The sky clears and surrounds the sun with blueness.',
            'Clouds. A storm blows up, with flashes of lightening.',
            'A rainbow appears.',
            'The overall appearance of the shop from the opposite side of the road.',
            'A window display including colours, shape and details of individual items for sale.',
            'You are near the entrance. The colour, shape and details of the door.',
            'You enter the shop and go to the counter. The counter assistant serves you. Money changes hands.',
            'The contours of the landscape.',
            'The colour and shape of the trees.',
            'The colour and shape of the lake.',
            'A strong wind blows on the tree and on the lake causing waves.',
            'You observe the heavy traffic travelling at maximum speed around your car. The overall appearance of vehicles, their colours, sizes and shapes.',
            'Your car accelerates to overtake the traffic directly in front of you. You see and urgent expression on the face of the driver and the people in the other vehicles as you pass.',
            'A large truck is flashing its headlight directly behind. Your car quickly moves over to let the truck pass. The driver signals with a friendly wave.',
            'You see a broken-down vehicle beside the road. Its lights are flashing. The driver is looking concerned and she is using a mobile phone.',
            'The overall appearance and colour of the water, surf, and sky.',
            'Bathers are swimming and splashing about in the water. Some are playing with a brightly coloured beach ball.',
            'An ocean liner crosses the horizon. It leaves a trail of smoke in the blue sky.',
            'A beautiful air balloon appears with four people aboard. The balloon drifts past you, almost directly overhead. The passengers wave and smile. You wave and smile back at them.',
            'The overall appearance of the station viewed from in front of the main entrance.',
            'You walk into the station. The colour, shape and details of the entrance hall.',
            'You approach the ticket office, go to a vacant counter and purchase your ticket.',
            'You walk to the platform and observe other passengers and the railway lines. A train arrives. You climb aboard.',
            'The overall appearance and design of the garden.',
            'The colour and shape of the bushes and shrubs.',
            'The colour and appearance of the flowers.',
            'Some birds fly down onto the lawn and start pecking for food.'
        ]
        
        required_demo_columns = ['PROLIFIC_PID', 'Included', 'Time_taken_minutes']
        
        # Check for missing columns, accounting for potential whitespace differences
        vviq2_cols_stripped = [col.strip() if isinstance(col, str) else col for col in vviq2_data.columns]
        demo_cols_stripped = [col.strip() if isinstance(col, str) else col for col in demo_data.columns]
        
        missing_vviq2_cols = [col for col in required_vviq2_columns if col.strip() not in vviq2_cols_stripped]
        missing_demo_cols = [col for col in required_demo_columns if col.strip() not in demo_cols_stripped]
        
        if missing_vviq2_cols:
            print(f"Error: Missing required VVIQ2 columns: {missing_vviq2_cols}")
            return 1
        
        if missing_demo_cols:
            print(f"Error: Missing required demographic columns: {missing_demo_cols}")
            return 1
        
        # Merge datasets on PROLIFIC_PID
        print("Merging VVIQ2 and demographic data...")
        merged_data = pd.merge(vviq2_data, demo_data, on='PROLIFIC_PID', how='left')
        print(f"Merged data shape: {merged_data.shape}")
        print(f"Columns in merged data: {len(merged_data.columns)}")
        
        # Filter to include only participants with Included=TRUE
        # Handle case where Included might be a string 'True' instead of boolean True
        if merged_data['Included'].dtype == 'object':
            merged_data = merged_data[merged_data['Included'].astype(str).str.lower() == 'true']
        else:
            merged_data = merged_data[merged_data['Included'] == True]
        print(f"After filtering for Included=TRUE: {merged_data.shape}")
        
        # 2. Extract numeric scores
        print("Extracting numeric scores from VVIQ2 items...")
        # Get the item columns (all except PROLIFIC_PID)
        item_columns = required_vviq2_columns[1:]
        
        # Create new columns for numeric scores
        score_columns = [f"item{i+1}_score" for i in range(len(item_columns))]
        
        for i, col in enumerate(item_columns):
            merged_data[score_columns[i]] = merged_data[col].apply(extract_numeric_score)
        
        # Validate scores are between 1-5
        for col in score_columns:
            valid_scores = merged_data[col].dropna().unique()
            print(f"Unique values in {col}: {valid_scores}")
            # We should not flag valid scores (1-5) as invalid
            # All scores should be valid at this point, so just log them without warnings
        
        # 3. Apply participant-level exclusion criteria
        print("Applying exclusion criteria...")
        
        # Initialize exclusion DataFrame with empty strings for exclusion_reason (not NaN)
        exclusion_df = pd.DataFrame({
            'PROLIFIC_PID': merged_data['PROLIFIC_PID'],
            'excluded': False,
            'exclusion_reason': ''  # Initialize with empty string, not NaN
        })
        print("Initialized exclusion_df with empty strings for exclusion_reason")
        
        # Calculate percentage of missing items per participant
        exclusion_df['percent_missing'] = merged_data.apply(
            lambda row: calculate_missing_percentage(row, score_columns), axis=1
        )
        exclusion_df['n_missing_items'] = merged_data[score_columns].isna().sum(axis=1)
        
        # Calculate response pattern indicators
        exclusion_df['percent_identical_responses'] = merged_data.apply(
            lambda row: calculate_identical_response_percentage(row, score_columns), axis=1
        )
        
        exclusion_df['response_sd'] = merged_data[score_columns].std(axis=1)
        exclusion_df['completion_time_minutes'] = merged_data['Time_taken_minutes']
        
        # Calculate completion time percentiles
        completion_times = exclusion_df['completion_time_minutes']
        bottom_5_percentile = np.percentile(completion_times, 5)
        bottom_10_percentile = np.percentile(completion_times, 10)
        bottom_15_percentile = np.percentile(completion_times, 15)
        
        print(f"Completion time percentiles: 5th={bottom_5_percentile:.2f}, 10th={bottom_10_percentile:.2f}, 15th={bottom_15_percentile:.2f}")
        
        # Determine percentile for each participant
        exclusion_df['bottom_completion_percentile'] = exclusion_df['completion_time_minutes'].apply(
            lambda x: stats.percentileofscore(completion_times, x)
        )
        
        # Apply exclusion criteria
        # 1. Exclude participants missing >10% of VVIQ2 items (>3 items)
        missing_mask = exclusion_df['percent_missing'] > 10
        exclusion_df.loc[missing_mask, 'excluded'] = True
        exclusion_df.loc[missing_mask, 'exclusion_reason'] = 'Missing >10% of items'
        
        # Just report participants with any missing values, but don't exclude them
        any_missing_mask = exclusion_df['n_missing_items'] > 0
        print(f"Participants with any missing values: {any_missing_mask.sum()}")
        print(f"Note: Participants with ≤10% missing values will have values imputed, not excluded")
        
        # 2. Exclude participants with >90% identical responses AND completion time in bottom 10%
        identical_resp_mask = (exclusion_df['percent_identical_responses'] > 90) & (exclusion_df['completion_time_minutes'] <= bottom_10_percentile)
        exclusion_df.loc[identical_resp_mask & ~exclusion_df['excluded'], 'excluded'] = True
        exclusion_df.loc[identical_resp_mask & ~exclusion_df['excluded'], 'exclusion_reason'] = '>90% identical responses & bottom 10% completion time'
        
        # 3. Exclude participants with response SD <0.5 AND completion time in bottom 15%
        low_sd_mask = (exclusion_df['response_sd'] < 0.5) & (exclusion_df['completion_time_minutes'] <= bottom_15_percentile)
        exclusion_df.loc[low_sd_mask & ~exclusion_df['excluded'], 'excluded'] = True
        exclusion_df.loc[low_sd_mask & ~exclusion_df['excluded'], 'exclusion_reason'] = 'Response SD <0.5 & bottom 15% completion time'
        
        # 4. Report participants with extremely fast completion times (bottom 5%) but don't exclude them
        # unless they meet other criteria
        fast_completion_mask = exclusion_df['completion_time_minutes'] <= bottom_5_percentile
        print(f"Identified {fast_completion_mask.sum()} participants with extremely fast completion times (bottom 5%)")
        print("Note: Fast completion time alone is not an exclusion criterion")
        
        # Fill any missing exclusion reasons for excluded participants
        exclusion_df.loc[(exclusion_df['excluded'] == True) & 
                         (exclusion_df['exclusion_reason'].isna() | 
                          (exclusion_df['exclusion_reason'] == '')), 'exclusion_reason'] = 'Excluded by criteria'
        
        # After applying all exclusion criteria, ensure all excluded participants have a reason
        all_excluded_mask = exclusion_df['excluded'] == True
        no_reason_mask = (exclusion_df['exclusion_reason'] == '') | exclusion_df['exclusion_reason'].isna()
        exclusion_df.loc[all_excluded_mask & no_reason_mask, 'exclusion_reason'] = 'Excluded by criteria'
        
        # Print exclusion reason distribution after applying all criteria
        print(f"Exclusion reason distribution after applying all criteria:")
        for reason, count in exclusion_df['exclusion_reason'].value_counts().items():
            print(f"  - {reason}: {count}")
        
        # Merge exclusion info back to main data
        # Fix any NaN values in exclusion_reason before merging
        exclusion_df['exclusion_reason'] = exclusion_df['exclusion_reason'].fillna('')
        print(f"Exclusion reasons before merge: {exclusion_df['exclusion_reason'].value_counts().to_dict()}")
        
        # Print detailed exclusion statistics
        print(f"Total excluded participants: {exclusion_df['excluded'].sum()}")
        for reason in exclusion_df['exclusion_reason'].unique():
            if reason:  # Skip empty reasons
                count = (exclusion_df['exclusion_reason'] == reason).sum()
                print(f"  - {reason}: {count}")
        
        # Verify all excluded participants have a reason
        excluded_without_reason = (exclusion_df['excluded'] == True) & ((exclusion_df['exclusion_reason'] == '') | exclusion_df['exclusion_reason'].isna())
        if excluded_without_reason.sum() > 0:
            print(f"Warning: {excluded_without_reason.sum()} excluded participants have no reason. Fixing...")
            exclusion_df.loc[excluded_without_reason, 'exclusion_reason'] = 'Excluded by criteria'
        
        # Save a copy of the data before merging
        merged_data_before = merged_data.copy()
        original_columns = merged_data.columns.tolist()
        
        # Use left join to preserve all columns in merged_data
        merged_data = pd.merge(merged_data, exclusion_df[['PROLIFIC_PID', 'excluded', 'exclusion_reason', 'percent_missing']], 
                              on='PROLIFIC_PID', how='left')
        
        # Check if any columns were lost during the merge
        lost_columns = [col for col in original_columns if col not in merged_data.columns]
        if lost_columns:
            print(f"WARNING: Lost {len(lost_columns)} columns during merge: {lost_columns}")
            # This should not happen with a left join, but just in case
            print("Attempting to recover lost columns...")
            temp_df = pd.DataFrame(index=merged_data.index)
            for col in lost_columns:
                # Create a mapping from PROLIFIC_PID to column values
                if col in merged_data_before.columns:
                    # Use the original dataframe to recover the column
                    mapping = dict(zip(merged_data_before['PROLIFIC_PID'], merged_data_before[col]))
                    # Apply the mapping to recover the column
                    temp_df[col] = merged_data['PROLIFIC_PID'].map(mapping)
            # Add the recovered columns back to merged_data
            merged_data = pd.concat([merged_data, temp_df], axis=1)
            print(f"Recovered {len([col for col in lost_columns if col in merged_data.columns])} columns")
        
        # Verify score columns still exist after merge
        missing_score_cols = [col for col in score_columns if col not in merged_data.columns]
        if missing_score_cols:
            print(f"ERROR: Missing score columns after merge: {missing_score_cols}")
            raise ValueError(f"Score columns lost during merge: {missing_score_cols}")
        else:
            print(f"All {len(score_columns)} score columns preserved after merge")
        # Ensure exclusion_reason is not NaN in merged_data and is string type
        if 'exclusion_reason' in merged_data.columns:
            merged_data['exclusion_reason'] = merged_data['exclusion_reason'].fillna('').astype(str)
        
        # Verify exclusion info was properly merged
        print(f"Excluded participants after merge: {merged_data['excluded'].sum()}")
        
        # Perform Little's MCAR test on the original data before imputation
        # This gives a more accurate assessment of the missing data pattern
        print("Performing Little's MCAR test on original data before imputation...")
        mcar_p_value = littles_mcar_test(merged_data, score_columns)
        exclusion_df['littles_mcar_p_value'] = mcar_p_value
        exclusion_df['missing_pattern_systematic'] = mcar_p_value < 0.05 if mcar_p_value is not None else None
        
        # Print detailed missing data statistics
        # Use merged_data instead of vviq2_data since score_columns were created in merged_data
        missing_counts = merged_data[score_columns].isna().sum(axis=1)
        print(f"Missing data distribution before imputation: {missing_counts.value_counts().to_dict()}")
        print(f"Participants with 1-3 missing items (will be imputed): {((missing_counts > 0) & (missing_counts <= 3)).sum()}")
        print(f"Participants with >3 missing items (will be excluded): {(missing_counts > 3).sum()}")
        
        # 4. Handle missing data
        print("Handling missing data...")
        try:
            # Ensure percent_missing column exists in merged_data
            if 'percent_missing' not in merged_data.columns:
                print("Adding percent_missing column to merged_data...")
                merged_data['percent_missing'] = merged_data.apply(
                    lambda row: calculate_missing_percentage(row, score_columns), axis=1
                )
            
            # Use the handle_missing_values function to impute missing values
            merged_data_before = merged_data.copy()
            merged_data = handle_missing_values(merged_data, score_columns)
            
            # Verify that score columns still exist after handling missing values
            missing_score_cols = [col for col in score_columns if col not in merged_data.columns]
            if missing_score_cols:
                print(f"ERROR: Score columns lost during missing data handling: {missing_score_cols}")
                print("Attempting to recover by using original dataframe...")
                # Recover the columns by copying from the original dataframe
                for col in missing_score_cols:
                    if col in merged_data_before.columns:
                        merged_data[col] = merged_data_before[col]
                print(f"Recovered {len([col for col in missing_score_cols if col in merged_data.columns])} columns")
            else:
                print(f"All {len(score_columns)} score columns preserved after missing data handling")
            
            # Count how many participants had values imputed
            imputed_count = (merged_data['imputed_items'] != '').sum()
            print(f"Successfully imputed values for {imputed_count} participants")
        except Exception as e:
            print(f"Error during missing data handling: {e}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
        
        # 5. Calculate scores
        print("Calculating VVIQ2 scores...")
        try:
            # Verify that all score columns exist before calculating scores
            missing_score_cols = [col for col in score_columns if col not in merged_data.columns]
            if missing_score_cols:
                print(f"ERROR: Missing score columns before calculating scores: {missing_score_cols}")
                raise ValueError(f"Cannot calculate scores: missing columns {missing_score_cols}")
            
            scores_df = calculate_scenario_scores(merged_data, score_columns)
            
            # Verify exclusion info was properly copied to scores_df
            print(f"Exclusion info in scores_df:")
            print(f"  - Excluded participants: {scores_df['excluded'].sum()}")
            print(f"  - Exclusion reasons: {scores_df['exclusion_reason'].value_counts().to_dict()}")
        
            # Check for empty exclusion_reason values where excluded=True
            empty_reasons = ((scores_df['excluded'] == True) & (scores_df['exclusion_reason'] == '')).sum()
            if empty_reasons > 0:
                print(f"Warning: {empty_reasons} excluded participants have empty exclusion_reason")
                # Fix any remaining empty exclusion reasons for excluded participants
                scores_df.loc[(scores_df['excluded'] == True) & 
                             (scores_df['exclusion_reason'] == ''), 'exclusion_reason'] = 'Excluded by criteria'
                print(f"Fixed empty exclusion reasons. Updated distribution: {scores_df['exclusion_reason'].value_counts().to_dict()}")
            
            # Ensure non-excluded participants have empty strings for exclusion_reason
            scores_df.loc[~scores_df['excluded'], 'exclusion_reason'] = ''
            print(f"Set empty exclusion_reason for non-excluded participants. Final distribution: {scores_df['exclusion_reason'].value_counts().to_dict()}")
                
            print("Successfully calculated all VVIQ2 scores")
        except Exception as e:
            print(f"Error calculating scores: {e}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            # Create a minimal scores_df to allow the script to continue
            scores_df = pd.DataFrame({'PROLIFIC_PID': merged_data['PROLIFIC_PID']})
            scores_df['excluded'] = merged_data['excluded'] if 'excluded' in merged_data.columns else False
            scores_df['exclusion_reason'] = merged_data['exclusion_reason'] if 'exclusion_reason' in merged_data.columns else ''
        
        # 6. Assess reliability
        print("Assessing reliability...")
        try:
            reliability_df = calculate_reliability(merged_data, score_columns)
            print("Successfully calculated reliability metrics")
        except Exception as e:
            print(f"Error calculating reliability: {e}")
            # Create a minimal reliability_df to allow the script to continue
            reliability_df = pd.DataFrame(columns=['scale_name', 'cronbachs_alpha', 'n_items', 'n_participants'])
        
        # Save output files
        print("Saving output files...")
        
        # Final check of exclusion_reason and imputed_items columns
        for df, name in [(merged_data, 'merged_data'), (scores_df, 'scores_df')]:
            if 'exclusion_reason' in df.columns:
                df['exclusion_reason'] = df['exclusion_reason'].fillna('Excluded by criteria')
            if 'imputed_items' in df.columns:
                df['imputed_items'] = df['imputed_items'].fillna('')
        print("Final check: Ensured all exclusion_reason and imputed_items columns contain proper values")
        
        try:
            # 1. Cleaned data
            cleaned_data_file = f"outputs/VVIQ2_cleaned_data_{timestamp}.csv"
            # Ensure imputed_items column exists and contains empty strings, not NaN values
            if 'imputed_items' not in merged_data.columns:
                merged_data['imputed_items'] = ''
                print("Added 'imputed_items' column to merged_data for output")
            else:
                # Fix the imputed_items column before saving to CSV
                merged_data['imputed_items'] = merged_data['imputed_items'].fillna('')
                print(f"Preserved imputed_items column with {(merged_data['imputed_items'] != '').sum()} imputed participants")
            
            # Include original response columns, extracted numeric scores, and imputed_items
            output_columns = ['PROLIFIC_PID'] + required_vviq2_columns[1:] + score_columns + ['imputed_items']
            # Before writing to CSV, convert all string columns to string type
            output_df = merged_data[output_columns].copy()
            for col in output_df.columns:
                if output_df[col].dtype == 'object':
                    output_df[col] = output_df[col].astype(str).replace('nan', '')
            
            # Write CSV with explicit encoding and empty string handling
            with open(cleaned_data_file, 'w', encoding='utf-8', newline='') as f:
                output_df.to_csv(f, index=False, na_rep='')
            print(f"Saved cleaned data to: {cleaned_data_file}")
            print(f"Cleaned data includes {len(required_vviq2_columns)} original columns, {len(score_columns)} score columns, and imputed_items column")
            print(f"Verified that cleaned data contains all 32 original response columns and 32 extracted score columns")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
        
        try:
            # 2. Scores
            scores_file = f"outputs/VVIQ2_scores_{timestamp}.csv"
            # Ensure non-excluded participants have empty strings for exclusion_reason
            scores_df.loc[~scores_df['excluded'], 'exclusion_reason'] = ''
            print("Set empty exclusion_reason for non-excluded participants")
            print(f"Final exclusion reasons in scores_df: {scores_df['exclusion_reason'].value_counts().to_dict()}")
            # Before writing to CSV, convert all string columns to string type
            scores_output = scores_df.copy()
            for col in scores_output.columns:
                if scores_output[col].dtype == 'object':
                    scores_output[col] = scores_output[col].astype(str).replace('nan', '')
            
            # Write CSV with explicit encoding and empty string handling
            with open(scores_file, 'w', encoding='utf-8', newline='') as f:
                scores_output.to_csv(f, index=False, na_rep='')
            print(f"Saved scores to: {scores_file}")
        except Exception as e:
            print(f"Error saving scores: {e}")
        
        try:
            # 3. Reliability
            reliability_file = f"outputs/VVIQ2_reliability_{timestamp}.csv"
            # Before writing to CSV, convert all string columns to string type
            reliability_output = reliability_df.copy()
            for col in reliability_output.columns:
                if reliability_output[col].dtype == 'object':
                    reliability_output[col] = reliability_output[col].astype(str).replace('nan', '')
            
            # Write CSV with explicit encoding and empty string handling
            with open(reliability_file, 'w', encoding='utf-8', newline='') as f:
                reliability_output.to_csv(f, index=False, na_rep='')
            print(f"Saved reliability metrics to: {reliability_file}")
        except Exception as e:
            print(f"Error saving reliability metrics: {e}")
        
        try:
            # 4. Exclusion log
            exclusion_file = f"outputs/VVIQ2_exclusion_log_{timestamp}.csv"
            # Ensure exclusion_reason is string type
            exclusion_df['exclusion_reason'] = exclusion_df['exclusion_reason'].astype(str)
            # Create a new exclusion_reason column with proper values
            exclusion_df['exclusion_reason'] = exclusion_df.apply(
                lambda row: row['exclusion_reason'] if row['excluded'] and row['exclusion_reason'] not in [None, ""] 
                            else ("Excluded by criteria" if row['excluded'] else ""), 
                axis=1
            )
            
            # Ensure non-excluded participants have empty strings for exclusion_reason
            exclusion_df.loc[~exclusion_df['excluded'], 'exclusion_reason'] = ''
            print(f"Set empty exclusion_reason for non-excluded participants in exclusion_df")
            print(f"Final exclusion reasons in exclusion_df: {exclusion_df['exclusion_reason'].value_counts().to_dict()}")
            # Before writing to CSV, convert all string columns to string type
            exclusion_output = exclusion_df.copy()
            for col in exclusion_output.columns:
                if exclusion_output[col].dtype == 'object':
                    exclusion_output[col] = exclusion_output[col].astype(str).replace('nan', '')
            
            # Write CSV with explicit encoding and empty string handling
            with open(exclusion_file, 'w', encoding='utf-8', newline='') as f:
                exclusion_output.to_csv(f, index=False, na_rep='')
            print(f"Saved exclusion log to: {exclusion_file}")
        except Exception as e:
            print(f"Error saving exclusion log: {e}")
        
        print("\n===== SUMMARY =====")
        print(f"Total participants: {len(merged_data)}")
        print(f"Excluded participants: {exclusion_df['excluded'].sum()}")
        print(f"Included participants: {len(merged_data) - exclusion_df['excluded'].sum()}")
        
        print("\nExclusion reasons:")
        for reason in exclusion_df['exclusion_reason'].unique():
            if reason:  # Skip empty reasons
                count = (exclusion_df['exclusion_reason'] == reason).sum()
                print(f"  - {reason}: {count}")
        
        print("\nMissing data statistics:")
        print(f"  - Participants with missing data: {(exclusion_df['n_missing_items'] > 0).sum()}")
        print(f"  - Participants with >10% missing data: {(exclusion_df['percent_missing'] > 10).sum()}")
        print(f"  - Average missing percentage: {exclusion_df['percent_missing'].mean():.2f}%")
        
        # Verify that participants with missing data are properly handled
        missing_data_count = (exclusion_df['n_missing_items'] > 0).sum()
        missing_data_excluded = ((exclusion_df['n_missing_items'] > 0) & exclusion_df['excluded']).sum()
        imputed_count = (merged_data['imputed_items'] != '').sum()
        
        print(f"  - Participants with missing data: {missing_data_count}, of which {missing_data_excluded} are excluded")
        print(f"  - Participants with imputed values: {imputed_count}")
        print(f"  - Participants excluded due to >10% missing data: {(exclusion_df['percent_missing'] > 10).sum()}")
        print(f"  - Note: Participants with ≤10% missing values had values imputed using scenario means")
        
        print("\nResponse pattern statistics:")
        print(f"  - Average response SD: {exclusion_df['response_sd'].mean():.2f}")
        print(f"  - Participants with >90% identical responses: {(exclusion_df['percent_identical_responses'] > 90).sum()}")
        
        print("\nCompletion time statistics:")
        print(f"  - Average completion time: {exclusion_df['completion_time_minutes'].mean():.2f} minutes")
        print(f"  - Minimum completion time: {exclusion_df['completion_time_minutes'].min():.2f} minutes")
        print(f"  - Maximum completion time: {exclusion_df['completion_time_minutes'].max():.2f} minutes")
        
        print("Finished execution")
        return 0
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
