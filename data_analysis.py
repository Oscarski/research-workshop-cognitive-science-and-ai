import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
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
    
    # Style header row
    for i in range(8):
        table[(1, i)].set_facecolor('#f2f2cb')
    
    # Add title
    plt.title('Table 2: Logistic Regression Results for RQ1\n(Predicting Opinion Change from Consistency)',
              pad=20, fontsize=14)
    
    # Save figure
    plt.savefig('logistic_regression_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ordinal_regression_table(results):
    """Create a simplified table visualization for ordinal regression results."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Get parameter names and values
    params = results.params
    bse = results.bse
    
    # Calculate z-values and p-values
    zvalues = params / bse
    pvalues = 2 * (1 - stats.norm.cdf(abs(zvalues)))
    
    # Create simplified table data
    data = [
        ['Parameter', 'Coefficient', 'Std. Error', 'z-value', 'p-value'],
        ['Consistency', f'{params.iloc[0]:.3f}', f'{bse.iloc[0]:.3f}', f'{zvalues.iloc[0]:.3f}', f'{pvalues[0]:.3f}'],
        ['Cut: Low-Medium', f'{params.iloc[1]:.3f}', f'{bse.iloc[1]:.3f}', f'{zvalues.iloc[1]:.3f}', f'{pvalues[1]:.3f}'],
        ['Cut: Medium-High', f'{params.iloc[2]:.3f}', f'{bse.iloc[2]:.3f}', f'{zvalues.iloc[2]:.3f}', f'{pvalues[2]:.3f}']
    ]
    
    # Create table
    table = ax.table(
        cellText=data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.2, 0.2, 0.2, 0.2]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#3a75a8')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # Highlight important value (consistency p-value)
    table[(1, 0)].set_facecolor('#e6f2ff')
    table[(1, 4)].set_facecolor('#e6f2ff')
    
    # Title
    plt.title('Ordinal Regression: Effect of Consistency on Credibility Categories', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
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
    
    # Create simplified visualization for opinion change by consistency
    plt.figure(figsize=(8, 6))
    
    # Prepare data for plotting
    condition_labels = ['Inconsistent', 'Consistent']
    conditions = [0, 1]
    
    # Calculate proportion who changed opinion in each condition
    change_props = [
        df[df['is_consistent'] == 0]['change_ranking_binary'].mean(),
        df[df['is_consistent'] == 1]['change_ranking_binary'].mean()
    ]
    
    # Bar colors
    colors = ['#5DA5DA', '#FAA43A']
    
    # Create simplified bar chart
    plt.bar(conditions, change_props, color=colors, width=0.6)
    
    # Add data labels on top of bars
    for i, prop in enumerate(change_props):
        plt.text(i, prop + 0.02, f'{prop:.2f}', ha='center', fontsize=14, fontweight='bold')
    
    # Add labels and title
    plt.title('Opinion Change by Consistency', fontsize=16)
    plt.ylabel('Proportion Changed Opinion', fontsize=14)
    plt.xticks(conditions, condition_labels, fontsize=14)
    plt.ylim(0, 0.7)  # Set y-axis limit with some padding
    
    # Add a horizontal line for the overall average
    overall_mean = df['change_ranking_binary'].mean()
    plt.axhline(y=overall_mean, color='gray', linestyle='--', alpha=0.7)
    plt.text(0.5, overall_mean + 0.02, f'Overall: {overall_mean:.2f}', 
             ha='center', color='gray', fontsize=12)
    
    # Add odds ratio information
    plt.figtext(0.5, 0.01, 
                f'Odds Ratio: {odds_ratios[1]:.2f} (p = {results.pvalues[1]:.3f})', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
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
    
    # Convert categorical codes to numeric values (0=low, 1=medium, 2=high)
    credibility_numeric = df['credibility_category'].cat.codes
    
    # Prepare exogenous variables - OrderedModel adds constant internally
    exog = df[['is_consistent']]
    
    # Fit ordinal regression model
    model = OrderedModel(credibility_numeric, exog, distr='logit')
    results = model.fit(method='bfgs', disp=False)
    
    print("\nOrdinal Regression Results (Credibility):")
    print(results.summary())
    
    # Create table visualization
    plot_ordinal_regression_table(results)

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