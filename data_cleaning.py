import pandas as pd
import numpy as np

def clean_response(x):
    """Convert responses to numeric values (1-7)."""
    if pd.isna(x):
        return x
    if isinstance(x, str):
        # Get number if exists
        if any(char.isdigit() for char in x):
            return int(''.join(filter(str.isdigit, x)))
    return x

def load_and_clean_data():
    """Load and clean survey data."""
    # Load data
    versions = {
        'female_female': 'Version 1 - Version 1 (2).csv',
        'male_male': 'Version 2 - Version 2 (1).csv',
        'female_male': 'Version 3 - Version 3 (1).csv',
        'male_female': 'Version 4 - Version 4 (1).csv'
    }
    
    dfs = []
    for condition, file in versions.items():
        df = pd.read_csv(file)
        df['condition'] = condition
        df['Duration (in seconds)'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')
        df['Duration (in seconds)'] = df['Duration (in seconds)'].fillna(df['Duration (in seconds)'].mean())
        dfs.append(df)
    
    # Combine data
    raw_df = pd.concat(dfs, ignore_index=True)
    print(f"Total participants: {raw_df.shape[0]}")
    
    # Clean data
    raw_df.columns = raw_df.columns.str.strip()
    raw_df['participant_id'] = raw_df['Response ID']
    
    # Define necessary columns
    base_cols = ['participant_id', 'condition', 'What is your age?', 'What is your gender?',
                 'Duration (in seconds)', 'Do you want to change your ranking based on the video you just watched?']
    
    # Define Likert scale questions for each dimension
    dimension_questions = {
        'credibility': ['intelligent', 'expert', 'informed', 'competent', 'bright'],
        'trust': ['honest', 'trustworthy', 'honorable', 'moral', 'ethical', 'genuine'],
        'eerie': ['self-centered', 'insensitive', 'unconcerned']
    }
    
    # Get all Likert scale columns
    likert_cols = []
    for dimension, terms in dimension_questions.items():
        for term in terms:
            matching_cols = [col for col in raw_df.columns if term in col.lower() and col.startswith('Did the virtual agent')]
            likert_cols.extend(matching_cols)
    
    # Combine all necessary columns
    all_cols = base_cols + likert_cols
    raw_df = raw_df[all_cols]
    
    # Rename columns
    raw_df = raw_df.rename(columns={
        'What is your age?': 'age',
        'What is your gender?': 'gender',
        'Duration (in seconds)': 'duration_seconds',
        'Do you want to change your ranking based on the video you just watched?': 'change_ranking'
    })
    
    # Clean responses
    raw_df['change_ranking'] = raw_df['change_ranking'].map({'Yes': 1, 'No': 0})
    for col in likert_cols:
        raw_df[col] = raw_df[col].apply(clean_response)
    
    # Normalize scores
    for col in likert_cols:
        raw_df[col] = (raw_df[col] - 1) / 6
    
    return raw_df

def check_data_quality(df):
    """Remove exactly 2 fastest responders to ensure 10 participants per condition."""
    df_clean = df.copy()
    
    # Find conditions that have 11 participants
    condition_counts = df_clean['condition'].value_counts()
    conditions_with_11 = condition_counts[condition_counts == 11].index.tolist()
    
    if len(conditions_with_11) == 2:
        # Remove 1 fastest from each condition that has 11
        final_df = df_clean.copy()
        for condition in conditions_with_11:
            condition_data = final_df[final_df['condition'] == condition]
            # Keep 'other' gender if they exist
            other_participant = condition_data[condition_data['gender'] == 'Other']
            if not other_participant.empty:
                # Remove the fastest non-other participant
                non_other = condition_data[condition_data['gender'] != 'Other']
                fastest = non_other.sort_values('duration_seconds').head(1)
                final_df = final_df.drop(fastest.index)
            else:
                # Remove the fastest participant
                fastest = condition_data.sort_values('duration_seconds').head(1)
                final_df = final_df.drop(fastest.index)
    
    print(f"\nParticipants removed (fastest responders):")
    for condition in df_clean['condition'].unique():
        removed = df_clean[df_clean['condition'] == condition].shape[0] - final_df[final_df['condition'] == condition].shape[0]
        if removed > 0:
            print(f"- {removed} from {condition}")
    
    print("\nGender Distribution:")
    print(final_df['gender'].value_counts())
    
    return final_df

def prepare_for_modeling(df):
    """Prepare data for analysis."""
    # Create categories
    categories = {
        'credibility': ['intelligent', 'expert', 'informed', 'competent', 'bright'],
        'trust': ['honest', 'trustworthy', 'honorable', 'moral', 'ethical', 'genuine'],
        'eerie': ['self-centered', 'insensitive', 'unconcerned']
    }
    
    # Calculate scores
    for dimension, terms in categories.items():
        cols = [col for col in df.columns if any(term in col.lower() for term in terms)]
        df[f'{dimension}_score'] = df[cols].mean(axis=1)
        df[f'{dimension}_category'] = pd.qcut(df[f'{dimension}_score'], q=3, labels=['low', 'medium', 'high'])
    
    # Set categories
    df['condition'] = pd.Categorical(df['condition'], categories=['female_female', 'male_male', 'female_male', 'male_female'])
    df['gender'] = pd.Categorical(df['gender'])
    df['participant_id'] = pd.Categorical(df['participant_id'])
    df['change_ranking'] = pd.Categorical(df['change_ranking'], categories=[0, 1], ordered=True)
    
    return df

if __name__ == "__main__":
    # Load and clean data
    df = load_and_clean_data()
    df_clean = check_data_quality(df)
    df_final = prepare_for_modeling(df_clean)
    
    # Save cleaned data
    df_final.to_csv('cleaned_data.csv', index=False)
    
    # Print summary
    print("\nData Summary:")
    print(f"Total participants: {len(df_final['participant_id'].unique())}")
    print("\nCondition counts:")
    print(df_final['condition'].value_counts()) 