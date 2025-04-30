import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.mediation import Mediation
import statsmodels.api as sm
from data_cleaning import load_and_clean_data, check_data_quality, prepare_for_modeling
from scipy.stats import pearsonr

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

def plot_tertiles(df):
    """Plot distribution of tertiles by condition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dimension in enumerate(['credibility', 'trust', 'eerie']):
        sns.countplot(x='condition', hue=f'{dimension}_category', data=df, ax=axes[i])
        axes[i].set_title(f'{dimension.capitalize()} Tertiles by Condition')
        axes[i].set_xlabel('Condition')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Tertile')
    
    plt.tight_layout()
    plt.savefig('tertiles.png', dpi=300, bbox_inches='tight')
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

def analyze_rq1(df):
    """
    RQ1: Consistency increases persuasion
    Analysis: Mixed logistic regression for opinion change
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
    print(np.exp(results.params))
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    prop_change = df.groupby('is_consistent')['change_ranking_binary'].mean().reset_index()
    sns.barplot(x='is_consistent', y='change_ranking_binary', data=prop_change, 
                palette=['#e57373', '#4dd0e1'], ci=95, capsize=0.1)
    
    # Overlay individual condition points
    for i, group in enumerate([1, 0]):
        y = df[df['is_consistent'] == group].groupby('condition')['change_ranking_binary'].mean()
        plt.scatter([i]*len(y), y, color='k', label=None)
    
    plt.title('Opinion Change by Consistency')
    plt.xlabel('Consistent (1) vs Inconsistent (0)')
    plt.ylabel('Proportion Changed Opinion')
    plt.legend(['Individual Conditions'], loc='upper right')
    plt.savefig('rq1_consistency_persuasion.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_rq2(df):
    """
    RQ2: Credibility and trust mediation
    Analysis: Ordinal regression + mediation analysis
    """
    print("\nResearch Question 2: Credibility and Trust Mediation")
    print("------------------------------------------------")
    
    # Create consistency variable
    df['is_consistent'] = df['condition'].isin(['male_male', 'female_female']).astype(int)
    
    # Step 1: Consistency -> Credibility (a path)
    model_a = sm.OLS(df['credibility_score'], sm.add_constant(df['is_consistent'])).fit()
    a_coef = model_a.params['is_consistent']
    a_pval = model_a.pvalues['is_consistent']
    
    # Step 2: Credibility -> Change_ranking (b path)
    model_b = sm.Logit(df['change_ranking'], sm.add_constant(df['credibility_score'])).fit()
    b_coef = model_b.params['credibility_score']
    b_pval = model_b.pvalues['credibility_score']
    
    # Step 3: Consistency -> Change_ranking (c path - total effect)
    model_c = sm.Logit(df['change_ranking'], sm.add_constant(df['is_consistent'])).fit()
    c_coef = model_c.params['is_consistent']
    c_pval = model_c.pvalues['is_consistent']
    
    # Step 4: Consistency + Credibility -> Change_ranking (c' path - direct effect)
    model_cprime = sm.Logit(df['change_ranking'], 
                           sm.add_constant(pd.concat([df['is_consistent'], 
                                                    df['credibility_score']], axis=1))).fit()
    cprime_coef = model_cprime.params['is_consistent']
    cprime_pval = model_cprime.pvalues['is_consistent']
    
    # Calculate indirect effect (a*b)
    indirect_effect = a_coef * b_coef
    
    print("\nMediation Analysis Results:")
    print("---------------------------")
    print(f"Path a (Consistency -> Credibility): β = {a_coef:.3f}, p = {a_pval:.3f}")
    print(f"Path b (Credibility -> Change): β = {b_coef:.3f}, p = {b_pval:.3f}")
    print(f"Path c (Total Effect): β = {c_coef:.3f}, p = {c_pval:.3f}")
    print(f"Path c' (Direct Effect): β = {cprime_coef:.3f}, p = {cprime_pval:.3f}")
    print(f"Indirect Effect (a*b): {indirect_effect:.3f}")
    
    # Additional analysis with trust
    print("\nTrust as Secondary Mediator:")
    print("----------------------------")
    
    # Credibility -> Trust
    model_d = sm.OLS(df['trust_score'], sm.add_constant(df['credibility_score'])).fit()
    d_coef = model_d.params['credibility_score']
    d_pval = model_d.pvalues['credibility_score']
    
    # Trust -> Change_ranking
    model_e = sm.Logit(df['change_ranking'], sm.add_constant(df['trust_score'])).fit()
    e_coef = model_e.params['trust_score']
    e_pval = model_e.pvalues['trust_score']
    
    print(f"Credibility -> Trust: β = {d_coef:.3f}, p = {d_pval:.3f}")
    print(f"Trust -> Change: β = {e_coef:.3f}, p = {e_pval:.3f}")
    
    # Visualize relationships
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Credibility by Consistency
    sns.boxplot(x='is_consistent', y='credibility_score', data=df, ax=ax1)
    ax1.set_title('Credibility by Consistency')
    ax1.set_xlabel('Consistent (1) vs Inconsistent (0)')
    ax1.set_ylabel('Credibility Score')
    
    # Plot 2: Trust by Credibility Category
    sns.boxplot(x='credibility_category', y='trust_score', data=df, ax=ax2)
    ax2.set_title('Trust by Credibility Level')
    ax2.set_xlabel('Credibility Category')
    ax2.set_ylabel('Trust Score')
    
    plt.tight_layout()
    plt.savefig('rq2_mediation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_rq3(df):
    """
    RQ3: Inconsistency increases eeriness, decreases trust
    Analysis: Interaction effects
    """
    print("\nResearch Question 3: Inconsistency, Eeriness, and Trust")
    print("-------------------------------------------------")
    
    # Create consistency variable
    df['is_consistent'] = df['condition'].isin(['male_male', 'female_female']).astype(int)
    
    # Analyze relationship between inconsistency and eeriness
    eeriness_model = sm.OLS(df['eerie_score'], 
                           sm.add_constant(df['is_consistent'])).fit()
    
    print("\nEffect of Consistency on Eeriness:")
    print(eeriness_model.summary().tables[1])
    
    # Analyze relationship between inconsistency and trust
    trust_model = sm.OLS(df['trust_score'],
                        sm.add_constant(df['is_consistent'])).fit()
    
    print("\nEffect of Consistency on Trust:")
    print(trust_model.summary().tables[1])
    
    # Interaction analysis
    df['eeriness_x_consistency'] = df['eerie_score'] * df['is_consistent']
    interaction_model = sm.OLS(df['trust_score'],
                             sm.add_constant(pd.concat([
                                 df['is_consistent'],
                                 df['eerie_score'],
                                 df['eeriness_x_consistency']
                             ], axis=1))).fit()
    
    print("\nInteraction Analysis Results:")
    print(interaction_model.summary().tables[1])
    
    # Visualize relationships
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Eeriness by Consistency
    sns.boxplot(x='is_consistent', y='eerie_score', data=df, ax=ax1)
    ax1.set_title('Eeriness by Consistency')
    ax1.set_xlabel('Consistent (1) vs Inconsistent (0)')
    ax1.set_ylabel('Eeriness Score')
    
    # Plot 2: Trust vs Eeriness with Consistency
    sns.scatterplot(data=df, x='eerie_score', y='trust_score', 
                    hue='is_consistent', ax=ax2)
    ax2.set_title('Trust vs Eeriness by Consistency')
    ax2.set_xlabel('Eeriness Score')
    ax2.set_ylabel('Trust Score')
    
    plt.tight_layout()
    plt.savefig('rq3_eeriness_trust.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_research_questions_summary(df):
    # Prepare Consistency variable
    df['is_consistent'] = df['condition'].isin(['male_male', 'female_female']).astype(int)
    # Ensure change_ranking is binary (0/1)
    df['change_ranking_binary'] = df['change_ranking'].astype(int)

    # --- Panel 1: Opinion Change by Consistency Type ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), gridspec_kw={'height_ratios': [1, 1, 1.2, 1]})

    # Bar plot for proportion of opinion change
    prop_change = df.groupby('is_consistent')['change_ranking_binary'].mean().reset_index()
    sns.barplot(x='is_consistent', y='change_ranking_binary', data=prop_change, ax=axes[0], palette=['#e57373', '#4dd0e1'])
    # Overlay individual condition points
    for i, group in enumerate([1, 0]):
        y = df[df['is_consistent'] == group].groupby('condition')['change_ranking_binary'].mean()
        axes[0].scatter([i]*len(y), y, color='k', label=None)
    axes[0].set_xticklabels(['Consistent', 'Inconsistent'])
    axes[0].set_ylabel('Proportion of Opinion Change')
    axes[0].set_title('Opinion Change by Consistency Type')
    axes[0].legend(['Individual Conditions'], loc='upper right')

    # --- Panel 2: Logistic Regression Probability Plot ---
    # Fit logistic regression
    X = sm.add_constant(df['is_consistent'].astype(int))
    y = df['change_ranking_binary'].astype(int)
    model = sm.Logit(y, X).fit(disp=0)
    pred_df = pd.DataFrame({'is_consistent': [0, 1]})
    pred_X = sm.add_constant(pred_df)
    pred_probs = model.predict(pred_X)
    se = np.sqrt(np.diag(np.dot(pred_X, np.dot(model.cov_params(), pred_X.T))))
    logit = model.predict(pred_X, linear=True)
    lower = sm.families.links.logit().inverse(logit - 1.96 * se)
    upper = sm.families.links.logit().inverse(logit + 1.96 * se)
    axes[1].bar(['Inconsistent', 'Consistent'], pred_probs, color=['#4dd0e1', '#e57373'], alpha=0.7)
    axes[1].errorbar(['Inconsistent', 'Consistent'], pred_probs, yerr=[pred_probs-lower, upper-pred_probs], fmt='none', color='k', capsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Predicted Probability')
    axes[1].set_title('Logistic Regression: Probability of Opinion Change')

    # --- Panel 3: Trust by Credibility Category ---
    trust_means = df.groupby('credibility_category')['trust_score'].mean()
    trust_se = df.groupby('credibility_category')['trust_score'].sem()
    categories = ['low', 'medium', 'high']
    axes[2].errorbar(categories, trust_means[categories], yerr=trust_se[categories], fmt='o-', color='#e57373')
    cred_num = df['credibility_category'].cat.codes
    corr, _ = pearsonr(cred_num, df['trust_score'])
    axes[2].annotate(f'Correlation: {corr:.2f}', xy=(0.02, 0.92), xycoords='axes fraction')
    axes[2].set_ylabel('Mean Trust Score')
    axes[2].set_xlabel('Credibility Category')
    axes[2].set_title('Trust Scores by Credibility Categories (Mediation Path)')

    # --- Panel 4: Trust vs Eeriness by Consistency Type ---
    colors = {1: '#e57373', 0: '#4dd0e1'}
    for is_consistent, group_df in df.groupby('is_consistent'):
        axes[3].scatter(group_df['eerie_score'], group_df['trust_score'], color=colors[is_consistent], alpha=0.7, label='Consistent' if is_consistent else 'Inconsistent')
        if len(group_df) > 1:
            m, b = np.polyfit(group_df['eerie_score'], group_df['trust_score'], 1)
            axes[3].plot(group_df['eerie_score'], m*group_df['eerie_score']+b, color=colors[is_consistent], linestyle='--')
            corr, _ = pearsonr(group_df['eerie_score'], group_df['trust_score'])
            axes[3].annotate(f'{"Consistent" if is_consistent else "Inconsistent"} Correlation: {corr:.2f}',
                             xy=(0.02, 0.92 if is_consistent else 0.85), xycoords='axes fraction', color=colors[is_consistent])
    axes[3].set_xlabel('Eeriness Score')
    axes[3].set_ylabel('Trust Score')
    axes[3].set_title('Trust vs Eeriness by Consistency Type (Interaction)')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('research_questions_summary.png', dpi=300)
    plt.close()

def main():
    # Load and clean data
    df = load_and_clean_data()
    df_clean = check_data_quality(df)
    df_final = prepare_for_modeling(df_clean)
    
    # Ensure change_ranking is binary
    df_final['change_ranking_binary'] = df_final['change_ranking'].astype(int)
    
    # Create visualizations
    plot_composite_scores(df_final)
    plot_tertiles(df_final)
    plot_demographics(df_final)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for dimension in ['credibility', 'trust', 'eerie']:
        print(f"\n{dimension.capitalize()} Scores:")
        print(df_final.groupby('condition')[f'{dimension}_score'].describe())
    
    # Perform statistical tests
    print("\nStatistical Tests:")
    for dimension in ['credibility', 'trust', 'eerie']:
        print(f"\n{dimension.capitalize()} Score ANOVA:")
        f_stat, p_value = stats.f_oneway(
            *[group[f'{dimension}_score'] for name, group in df_final.groupby('condition')]
        )
        print(f"F-statistic: {f_stat:.3f}")
        print(f"P-value: {p_value:.3f}")
    
    # Analyze research questions
    analyze_rq1(df_final)
    analyze_rq2(df_final)
    analyze_rq3(df_final)
    
    # Multi-panel summary plot
    plot_research_questions_summary(df_final)

if __name__ == "__main__":
    main() 