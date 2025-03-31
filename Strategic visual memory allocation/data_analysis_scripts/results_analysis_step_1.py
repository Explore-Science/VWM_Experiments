#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def create_output_dir():
    """
    Create the outputs directory if it doesn't exist.
    
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        if not os.path.exists('outputs'):
            print(f"Creating outputs directory...")
            os.makedirs('outputs')
            print(f"Successfully created outputs directory.")
        return True
    except Exception as e:
        print(f"Error creating outputs directory: {e}")
        return False

def get_latest_file(file_pattern):
    """
    Get the most recent file matching the given pattern.
    
    Args:
        file_pattern (str): File pattern to match
    
    Returns:
        str: Path to the most recent file, or None if no files found
    """
    matching_files = glob.glob(file_pattern)
    print(f"Found {len(matching_files)} files matching pattern '{file_pattern}'")
    
    if not matching_files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    # Get the most recent file based on creation time
    latest_file = max(matching_files, key=os.path.getctime)
    print(f"Latest file: {latest_file} (created: {datetime.fromtimestamp(os.path.getctime(latest_file))})")
    
    return latest_file

def load_prolific_data(file_path):
    """
    Load the Prolific export CSV file containing participant demographic data.
    
    This function reads the CSV file exported from Prolific, which contains
    demographic information about study participants. It prints basic information
    about the loaded data including row count, column count, column names, and
    a preview of the first few rows.
    
    Args:
        file_path (str): Full path to the Prolific export CSV file to be loaded
    
    Returns:
        pandas.DataFrame: DataFrame containing the loaded participant data with all
                         original columns from the Prolific export, or None if the
                         loading process failed due to file not found, permission
                         issues, or malformed data
    """
    try:
        print(f"Loading data from: {file_path}")
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Print columns
        print("Columns in the dataset:")
        print(data.columns.tolist())
        
        # Print first three rows
        print("\nFirst three rows of the dataset:")
        print(data.head(3))
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def validate_required_columns(data):
    """
    Validate that all required columns are present in the dataframe.
    
    This function checks if all columns required for analysis are present in the
    dataset. It also verifies that the data types are appropriate for analysis.
    
    Args:
        data (pandas.DataFrame): The dataframe to validate containing the raw
                                Prolific export data
    
    Returns:
        bool: True if all required columns exist with appropriate data types,
              False otherwise
    """
    required_columns = [
        'Participant id', 'Age', 'Sex', 'Colourblindness', 'Vision',
        'Country of residence', 'Student status', 'Employment status',
        'Time taken', 'Status'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    # Check data types for critical columns
    try:
        # Age should be numeric
        if not pd.api.types.is_numeric_dtype(data['Age']):
            print(f"Warning: 'Age' column is not numeric. Attempting to convert...")
            data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
            if data['Age'].isna().any():
                print(f"Warning: Some age values could not be converted to numbers")
        
        # Time taken should be numeric
        if not pd.api.types.is_numeric_dtype(data['Time taken']):
            print(f"Warning: 'Time taken' column is not numeric. Attempting to convert...")
            data['Time taken'] = pd.to_numeric(data['Time taken'], errors='coerce')
            if data['Time taken'].isna().any():
                print(f"Warning: Some time values could not be converted to numbers")
    
    except Exception as e:
        print(f"Error validating data types: {e}")
    
    print("All required columns are present in the dataset.")
    return True

def filter_and_validate_data(data):
    """
    Filter out participants with Status not equal to "APPROVED" and validate inclusion criteria.
    
    This function performs several key data processing steps:
    1. Selects only the required columns from the original dataset
    2. Filters to keep only participants with "APPROVED" status
    3. Adds derived columns (Time_taken_minutes, Included, Exclusion_reason)
    4. Validates participants against inclusion criteria:
       - Age between 18-35 years
       - Vision response is "Yes"
       - No colorblindness issues
    5. Handles missing data and DATA_EXPIRED values
    6. Renames columns to match output requirements
    
    Args:
        data (pandas.DataFrame): The original dataframe containing all Prolific export data
                                with at least the required demographic columns
    
    Returns:
        pandas.DataFrame: Processed dataframe with:
                         - Only required columns
                         - Additional derived columns
                         - Inclusion/exclusion flags
                         - Standardized column names
                         - Missing data properly handled
    """
    # Make a copy with only the required columns to avoid SettingWithCopyWarning
    columns_to_keep = [
        'Participant id', 'Age', 'Sex', 'Colourblindness', 'Vision',
        'Country of residence', 'Student status', 'Employment status',
        'Time taken', 'Status'
    ]
    df = data[columns_to_keep].copy()
    
    # Print unique values of key columns
    for col in ['Status', 'Sex', 'Vision', 'Colourblindness', 'Student status', 'Employment status']:
        if col in df.columns:
            print(f"\nUnique values in {col}:")
            print(df[col].unique())
    
    # Check for missing values in required columns
    print("\nChecking for missing values in required columns:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"Column '{col}' has {missing} missing values")
    
    # Filter out participants with Status not equal to "APPROVED"
    print(f"\nTotal participants before filtering: {df.shape[0]}")
    approved_df = df[df['Status'] == 'APPROVED'].copy()
    print(f"Approved participants: {approved_df.shape[0]}")
    print(f"Excluded participants (not approved): {df.shape[0] - approved_df.shape[0]}")
    
    # Rename 'Participant id' to 'PROLIFIC_PID'
    approved_df.rename(columns={'Participant id': 'PROLIFIC_PID'}, inplace=True)
    
    # Add 'Time_taken_minutes' column
    approved_df['Time_taken_seconds'] = approved_df['Time taken']
    approved_df['Time_taken_minutes'] = approved_df['Time taken'] / 60
    
    # Add 'Included' and 'Exclusion_reason' columns
    approved_df['Included'] = True
    approved_df['Exclusion_reason'] = ''
    
    # Verify Age is between 18-35 years (based on actual data range)
    age_exclusions = approved_df[(approved_df['Age'] < 18) | (approved_df['Age'] > 35)].copy()
    if not age_exclusions.empty:
        print(f"Excluding {age_exclusions.shape[0]} participants with age outside 18-35 range")
        approved_df.loc[age_exclusions.index, 'Included'] = False
        approved_df.loc[age_exclusions.index, 'Exclusion_reason'] = 'Age outside 18-35 range'
    else:
        print(f"All {approved_df.shape[0]} participants meet the age criteria (18-35 years)")
    
    # Verify Vision is "Yes"
    vision_exclusions = approved_df[approved_df['Vision'] != 'Yes'].copy()
    if not vision_exclusions.empty:
        print(f"Excluding {vision_exclusions.shape[0]} participants with Vision not equal to 'Yes'")
        approved_df.loc[vision_exclusions.index, 'Included'] = False
        approved_df.loc[vision_exclusions.index, 'Exclusion_reason'] = 'Vision not equal to Yes'
    
    # Verify Colourblindness is "No, I have no issues seeing colours"
    colorblind_exclusions = approved_df[approved_df['Colourblindness'] != 'No, I have no issues seeing colours'].copy()
    if not colorblind_exclusions.empty:
        print(f"Excluding {colorblind_exclusions.shape[0]} participants with Colourblindness issues")
        approved_df.loc[colorblind_exclusions.index, 'Included'] = False
        approved_df.loc[colorblind_exclusions.index, 'Exclusion_reason'] = 'Has colorblindness issues'
    
    # Rename column to match output requirements
    approved_df.rename(columns={'Country of residence': 'Country_of_residence', 
                               'Student status': 'Student_status',
                               'Employment status': 'Employment_status'}, inplace=True)
    
    # Handle DATA_EXPIRED values as missing data
    for col in ['Student_status', 'Employment_status', 'Country_of_residence']:
        if col in approved_df.columns:
            expired_count = (approved_df[col] == 'DATA_EXPIRED').sum()
            if expired_count > 0:
                approved_df.loc[approved_df[col] == 'DATA_EXPIRED', col] = np.nan
                print(f"Replaced 'DATA_EXPIRED' with NaN in {col} for {expired_count} participants")
    
    # Handle any other missing values in important columns
    for col in ['Country_of_residence', 'Student_status', 'Employment_status']:
        if col in approved_df.columns:
            na_count = approved_df[col].isna().sum()
            if na_count > 0:
                print(f"Found {na_count} missing values in {col}")
    
    # Report on inclusion criteria counts
    print("\nInclusion criteria summary:")
    total_participants = approved_df.shape[0]
    print(f"Total participants assessed: {total_participants}")
    
    # Age criteria
    age_criteria_met = approved_df[(approved_df['Age'] >= 18) & (approved_df['Age'] <= 35)].shape[0]
    print(f"Age criteria (18-35): {age_criteria_met} participants ({age_criteria_met/total_participants*100:.2f}%)")
    
    # Vision criteria
    vision_criteria_met = approved_df[approved_df['Vision'] == 'Yes'].shape[0]
    print(f"Vision criteria ('Yes'): {vision_criteria_met} participants ({vision_criteria_met/total_participants*100:.2f}%)")
    
    # Colorblindness criteria
    colorblind_criteria_met = approved_df[approved_df['Colourblindness'] == 'No, I have no issues seeing colours'].shape[0]
    print(f"Colorblindness criteria (no issues): {colorblind_criteria_met} participants ({colorblind_criteria_met/total_participants*100:.2f}%)")
    
    # Final included participants
    included_df = approved_df[approved_df['Included'] == True].copy()
    print(f"\nFinal included participants: {included_df.shape[0]} ({included_df.shape[0]/total_participants*100:.2f}%)")
    
    # Print first two rows of the processed dataframe
    print("\nFirst two rows of the processed dataframe:")
    print(approved_df.head(2))
    
    return approved_df

def calculate_descriptive_statistics(data):
    """
    Calculate descriptive statistics for the filtered and validated data.
    
    This function computes comprehensive descriptive statistics for the demographic
    data, including:
    1. Age statistics (mean, SD, range, median)
    2. Sex distribution (counts and percentages)
    3. Country of residence distribution
    4. Student status distribution
    5. Employment status distribution
    6. Time taken statistics (mean, SD, range)
    7. Sample size statistics relative to target
    
    Args:
        data (pandas.DataFrame): The filtered and validated dataframe containing
                                participant demographic data with inclusion flags
    
    Returns:
        pandas.DataFrame: Structured dataframe containing all calculated statistics
                         with columns for statistic name, value, category, and percentage
    """
    # Only include participants that meet all inclusion criteria
    included_data = data[data['Included'] == True].copy()
    print(f"Calculating statistics for {included_data.shape[0]} included participants")
    
    # Initialize list to store statistics
    stats = []
    
    # Age statistics
    age_mean = included_data['Age'].mean()
    age_sd = included_data['Age'].std()
    age_min = included_data['Age'].min()
    age_max = included_data['Age'].max()
    age_median = included_data['Age'].median()
    
    stats.append({'Statistic_name': 'Age_Mean', 'Value': age_mean, 'Category': 'Age', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Age_SD', 'Value': age_sd, 'Category': 'Age', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Age_Min', 'Value': age_min, 'Category': 'Age', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Age_Max', 'Value': age_max, 'Category': 'Age', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Age_Median', 'Value': age_median, 'Category': 'Age', 'Percentage': np.nan})
    
    print(f"Age statistics: Mean={age_mean:.2f}, SD={age_sd:.2f}, Range={age_min}-{age_max}, Median={age_median}")
    
    # Sex statistics
    sex_counts = included_data['Sex'].value_counts()
    total_participants = included_data.shape[0]
    
    for sex, count in sex_counts.items():
        percentage = (count / total_participants) * 100
        stats.append({'Statistic_name': 'Sex_Count', 'Value': count, 'Category': sex, 'Percentage': percentage})
        print(f"Sex '{sex}': Count={count}, Percentage={percentage:.2f}%")
    
    # Country of residence statistics
    country_counts = included_data['Country_of_residence'].value_counts()
    
    for country, count in country_counts.items():
        percentage = (count / total_participants) * 100
        stats.append({'Statistic_name': 'Country_Count', 'Value': count, 'Category': country, 'Percentage': percentage})
        print(f"Country '{country}': Count={count}, Percentage={percentage:.2f}%")
    
    # Student status statistics
    student_counts = included_data['Student_status'].value_counts()
    
    for status, count in student_counts.items():
        percentage = (count / total_participants) * 100
        stats.append({'Statistic_name': 'Student_Status_Count', 'Value': count, 'Category': status, 'Percentage': percentage})
        print(f"Student status '{status}': Count={count}, Percentage={percentage:.2f}%")
    
    # Employment status statistics
    employment_counts = included_data['Employment_status'].value_counts()
    
    for status, count in employment_counts.items():
        percentage = (count / total_participants) * 100
        stats.append({'Statistic_name': 'Employment_Status_Count', 'Value': count, 'Category': status, 'Percentage': percentage})
        print(f"Employment status '{status}': Count={count}, Percentage={percentage:.2f}%")
    
    # Time taken statistics (in minutes)
    time_mean = included_data['Time_taken_minutes'].mean()
    time_sd = included_data['Time_taken_minutes'].std()
    time_min = included_data['Time_taken_minutes'].min()
    time_max = included_data['Time_taken_minutes'].max()
    
    stats.append({'Statistic_name': 'Time_Taken_Mean_Minutes', 'Value': time_mean, 'Category': 'Time', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Time_Taken_SD_Minutes', 'Value': time_sd, 'Category': 'Time', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Time_Taken_Min_Minutes', 'Value': time_min, 'Category': 'Time', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Time_Taken_Max_Minutes', 'Value': time_max, 'Category': 'Time', 'Percentage': np.nan})
    
    print(f"Time taken statistics (minutes): Mean={time_mean:.2f}, SD={time_sd:.2f}, Range={time_min:.2f}-{time_max:.2f}")
    
    # Sample size statistics
    target_sample_size = 91
    actual_sample_size = included_data.shape[0]
    percentage_achieved = (actual_sample_size / target_sample_size) * 100
    
    stats.append({'Statistic_name': 'Total_Approved_Participants', 'Value': data[data['Status'] == 'APPROVED'].shape[0], 'Category': 'Sample', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Total_Included_Participants', 'Value': actual_sample_size, 'Category': 'Sample', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Target_Sample_Size', 'Value': target_sample_size, 'Category': 'Sample', 'Percentage': np.nan})
    stats.append({'Statistic_name': 'Percentage_Target_Achieved', 'Value': percentage_achieved, 'Category': 'Sample', 'Percentage': percentage_achieved})
    
    print(f"Sample size: {actual_sample_size} out of target {target_sample_size} ({percentage_achieved:.2f}%)")
    
    # Convert statistics to dataframe
    stats_df = pd.DataFrame(stats)
    
    # Print first two rows of the statistics dataframe
    print("\nFirst two rows of the statistics dataframe:")
    print(stats_df.head(2))
    
    return stats_df

def save_output_files(cleaned_data, statistics):
    """
    Save the cleaned data and statistics to output CSV files.
    
    Args:
        cleaned_data (pandas.DataFrame): The cleaned demographic data
        statistics (pandas.DataFrame): The calculated statistics
    
    Returns:
        tuple: (bool, str, str) indicating success and the paths to the saved files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output filenames
    cleaned_data_filename = f"outputs/demographic_data_cleaned_{timestamp}.csv"
    statistics_filename = f"outputs/sample_statistics_{timestamp}.csv"
    
    try:
        # Subset the dataframe to include only the required columns
        required_columns = [
            'PROLIFIC_PID', 'Age', 'Sex', 'Colourblindness', 'Vision',
            'Country_of_residence', 'Student_status', 'Employment_status',
            'Time_taken_seconds', 'Time_taken_minutes', 'Included', 'Exclusion_reason'
        ]
        # Verify all required columns exist
        missing_cols = [col for col in required_columns if col not in cleaned_data.columns]
        if missing_cols:
            print(f"Warning: Missing columns in output: {missing_cols}")
            # Only keep columns that exist
            available_cols = [col for col in required_columns if col in cleaned_data.columns]
            cleaned_data_to_save = cleaned_data[available_cols].copy()
        else:
            cleaned_data_to_save = cleaned_data[required_columns].copy()
        
        # Save cleaned data
        print(f"Saving cleaned demographic data to {cleaned_data_filename}")
        print(f"Saving {len(required_columns)} columns: {', '.join(required_columns)}")
        cleaned_data_to_save.to_csv(cleaned_data_filename, index=False, na_rep='NA')
        
        # Save statistics
        print(f"Saving sample statistics to {statistics_filename}")
        statistics.to_csv(statistics_filename, index=False, na_rep='NA')
        
        print(f"Successfully saved output files")
        return True, cleaned_data_filename, statistics_filename
    except Exception as e:
        print(f"Error saving output files: {e}")
        return False, None, None

def main():
    """
    Main function to execute the demographic data analysis.
    
    Returns:
        int: 0 for success, 1 for failure
    """
    # Create output directory
    if not create_output_dir():
        return 1
    
    # Get the most recent prolific export file
    file_pattern = "../C_results/data/prolific_export_*.csv"
    input_file = get_latest_file(file_pattern)
    
    if not input_file:
        print("No input file found. Exiting.")
        return 1
    
    # Load the data
    data = load_prolific_data(input_file)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Validate required columns
    if not validate_required_columns(data):
        print("Missing required columns. Exiting.")
        return 1
    
    # Filter and validate data
    cleaned_data = filter_and_validate_data(data)
    
    # Calculate descriptive statistics
    statistics = calculate_descriptive_statistics(cleaned_data)
    
    # Save output files
    success, cleaned_data_file, statistics_file = save_output_files(cleaned_data, statistics)
    
    if not success:
        print("Failed to save output files. Exiting.")
        return 1
    
    # Print summary statistics
    print(f"\nSummary of Data Processing:")
    print(f"Initial participants: {data.shape[0]}")
    print(f"Approved participants: {cleaned_data[cleaned_data['Status'] == 'APPROVED'].shape[0]}")
    print(f"Included participants: {cleaned_data[cleaned_data['Included'] == True].shape[0]}")
    print(f"Target sample size: 91")
    
    # Print exclusion counts
    exclusion_counts = cleaned_data[cleaned_data['Included'] == False]['Exclusion_reason'].value_counts()
    if not exclusion_counts.empty:
        print("\nExclusion counts by reason:")
        for reason, count in exclusion_counts.items():
            print(f"- {reason}: {count}")
    else:
        print("\nNo participants were excluded based on criteria.")
    
    print("Finished execution")
    return 0

if __name__ == "__main__":
    sys.exit(main())
