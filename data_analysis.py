import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from data_cleaning import load_and_clean_data, check_data_quality, prepare_for_modeling

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_composite_scores(df):
    """Plot composite scores by condition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dimension in enumerate(['credibility', 'trust', 'eerie']):
        sns.boxplot(x='condition', y=f'{dimension}_score', data=df, ax=axes[i])
        axes[i].set_title(f'{dimension.capitalize()} Score by Condition')
        axes[i].set_xlabel('Condition')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('composite_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_demographics(df):
    """Plot age and gender distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Age distribution
    sns.histplot(data=df, x='age', bins=20, ax=axes[0])
    axes[0].set_title('Age Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    
    # Gender distribution
    gender_counts = df['gender'].value_counts()
    axes[1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Gender Distribution')
    
    plt.tight_layout()
    plt.savefig('demographics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_logistic_regression_table(results):
    """Create a table visualization for logistic regression results."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    data = [
        ['Variable', 'Coef.', 'Std. Err.', 'z', 'P>|z|', '[0.025', '0.975]', 'Odds Ratio'],
        ['const', f'{results.params[0]:.3f}', f'{results.bse[0]:.3f}', 
         f'{results.tvalues[0]:.3f}', f'{results.pvalues[0]:.3f}',
         f'{results.conf_int()[0][0]:.3f}', f'{results.conf_int()[0][1]:.3f}',
         f'{np.exp(results.params[0]):.2f}'],
        ['is_consistent', f'{results.params[1]:.3f}', f'{results.bse[1]:.3f}',
         f'{results.tvalues[1]:.3f}', f'{results.pvalues[1]:.3f}',
         f'{results.conf_int()[1][0]:.3f}', f'{results.conf_int()[1][1]:.3f}',
         f'{np.exp(results.params[1]):.2f}']
    ]
    
    # Create table
    table = ax.table(cellText=data,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(8):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # Add title
    plt.title('Table 2: Logistic Regression Results for RQ1\n(Predicting Opinion Change from Consistency)',
              pad=20, fontsize=14)
    
    # Save figure
    plt.savefig('logistic_regression_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ordinal_regression_table(results):
    """Create a table visualization for ordinal regression results."""
    # Convert results to DataFrame for easier access
    params = results.params
    bse = results.bse
    tvalues = results.tvalues
    pvalues = results.pvalues
    conf_int = results.conf_int()

    # Prepare table data
    data = [
        ['Variable', 'Coef.', 'Std. Err.', 'z', 'P>|z|', '[0.025', '0.975]'],
        ['const (y=1)', f'{params.iloc[0,0]:.3f}', f'{bse.iloc[0,0]:.3f}', f'{tvalues.iloc[0,0]:.3f}', f'{pvalues.iloc[0,0]:.3f}', f'{conf_int.iloc[0,0]:.3f}', f'{conf_int.iloc[0,1]:.3f}'],
        ['is_consistent (y=1)', f'{params.iloc[0,1]:.3f}', f'{bse.iloc[0,1]:.3f}', f'{tvalues.iloc[0,1]:.3f}', f'{pvalues.iloc[0,1]:.3f}', f'{conf_int.iloc[0,1]:.3f}', f'{conf_int.iloc[1,1]:.3f}'],
        ['const (y=2)', f'{params.iloc[1,0]:.3f}', f'{bse.iloc[1,0]:.3f}', f'{tvalues.iloc[1,0]:.3f}', f'{pvalues.iloc[1,0]:.3f}', f'{conf_int.iloc[1,0]:.3f}', f'{conf_int.iloc[1,1]:.3f}'],
        ['is_consistent (y=2)', f'{params.iloc[1,1]:.3f}', f'{bse.iloc[1,1]:.3f}', f'{tvalues.iloc[1,1]:.3f}', f'{pvalues.iloc[1,1]:.3f}', f'{conf_int.iloc[1,1]:.3f}', f'{conf_int.iloc[1,1]:.3f}']
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(color='white', weight='bold')
    plt.title('Table 3: Ordinal Regression Results for RQ2\n(Credibility Categories by Consistency)',
              pad=20, fontsize=14)
    plt.savefig('ordinal_regression_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_rq1(df):
    """
    RQ1: Consistency increases persuasion
    Hypothesis: Consistency ↑ persuasion
    Dependent Variable: Opinion change
    Test: Logistic regression
    """
    print("\nResearch Question 1: Consistency and Persuasion")
    print("----------------------------------------------")
    
    # Create consistency variable (1 for consistent, 0 for inconsistent)
    df['is_consistent'] = df['condition'].isin(['male_male', 'female_female']).astype(int)
    
    # Ensure change_ranking is binary
    df['change_ranking_binary'] = df['change_ranking'].astype(int)
    
    # Prepare data for logistic regression
    X = sm.add_constant(df['is_consistent'])
    y = df['change_ranking_binary']
    
    # Fit logistic regression
    model = sm.Logit(y, X)
    results = model.fit()
    
    print("\nLogistic Regression Results:")
    print(results.summary().tables[1])
    
    # Calculate and print odds ratios
    print("\nOdds Ratios:")
    odds_ratios = np.exp(results.params)
    print(odds_ratios)
    
    # Create table visualization
    plot_logistic_regression_table(results)
    
    # Get predicted probabilities
    pred_probs = results.predict()
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    prop_change = df.groupby('is_consistent')['change_ranking_binary'].mean().reset_index()
    # Ensure correct order: 0 (Inconsistent), 1 (Consistent)
    prop_change = prop_change.sort_values('is_consistent')
    colors = ['#4dd0e1', '#e57373']  # 0: Inconsistent, 1: Consistent
    
    # Plot observed proportions
    bar = sns.barplot(x='is_consistent', y='change_ranking_binary', data=prop_change, 
                     palette=colors, errorbar=('ci', 95))
    
    # Overlay individual condition points
    for i, group in enumerate([0, 1]):
        y_vals = df[df['is_consistent'] == group].groupby('condition')['change_ranking_binary'].mean()
        plt.scatter([i]*len(y_vals), y_vals, color='k', label=None)
    
    # Overlay predicted probabilities
    mean_probs = [np.mean(pred_probs[df['is_consistent'] == i]) for i in [0, 1]]
    plt.plot([-0.2, 0.2], [mean_probs[0], mean_probs[0]], 'r--', alpha=0.5)
    plt.plot([0.8, 1.2], [mean_probs[1], mean_probs[1]], 'r--', alpha=0.5)
    
    # Annotate means
    for i, row in prop_change.iterrows():
        plt.text(i, row['change_ranking_binary'] + 0.03, f"{row['change_ranking_binary']:.2f}", ha='center', fontsize=12)
    
    plt.title('Opinion Change by Consistency (Logistic Regression)')
    plt.xlabel('Consistency')
    plt.ylabel('Proportion Changed Opinion')
    plt.xticks([0, 1], ['Inconsistent', 'Consistent'])
    plt.legend(['Individual Conditions', 'Model Predictions'], loc='upper right')
    plt.ylim(0, 1)
    plt.savefig('rq1_consistency_persuasion.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_rq2(df):
    """
    RQ2: Credibility and trust mediation
    Hypothesis: Credibility & trust mediate
    Dependent Variable: Credibility tertile
    Test: Ordinal regression
    """
    print("\nResearch Question 2: Credibility and Trust Mediation")
    print("------------------------------------------------")
    
    # Create consistency variable
    df['is_consistent'] = df['condition'].isin(['male_male', 'female_female']).astype(int)
    
    # Step 1: Ordinal regression for credibility tertiles
    credibility_model = sm.MNLogit(df['credibility_category'].cat.codes, 
                                  sm.add_constant(df['is_consistent'])).fit()
    print("\nOrdinal Regression Results (Credibility):")
    print(credibility_model.summary().tables[1])
    
    # Create table visualization
    plot_ordinal_regression_table(credibility_model)

def main():
    # Load and clean data
    df = load_and_clean_data()
    
    # Check data quality
    df_final = check_data_quality(df)
    
    # Prepare data for modeling
    df_final = prepare_for_modeling(df_final)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for dimension in ['credibility', 'trust', 'eerie']:
        print(f"\n{dimension.capitalize()} Scores:")
        print(df_final.groupby('condition')[f'{dimension}_score'].describe())
    
    # Create visualizations
    plot_composite_scores(df_final)
    plot_demographics(df_final)
    
    # Run analyses
    analyze_rq1(df_final)
    analyze_rq2(df_final)

if __name__ == "__main__":
    main() 