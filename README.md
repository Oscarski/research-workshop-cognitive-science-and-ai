# 🔬 Virtual Agent Perception Analysis

This project analyzes survey data related to virtual agent (VA) perceptions across different conditions. The analysis focuses on three main research questions regarding consistency, credibility, trust, and eeriness in VA interactions.

## 📁 Project Structure

```
Research_final/
├── 📊 data_analysis.py         # Main analysis script
├── 🧹 data_cleaning.py         # Data cleaning and preparation functions
├── 📑 cleaned_data.csv         # Processed dataset
├── 📂 Version 1-4/*.csv        # Raw survey data files
├── 🎨 Visualizations/
│   ├── 📈 composite_scores.png    # Composite scores by condition
│   ├── 📊 tertiles.png           # Distribution of tertiles
│   ├── 👥 demographics.png       # Age and gender distributions
│   ├── 🔄 rq1_consistency_persuasion.png  # RQ1 analysis
│   ├── 🔗 rq2_mediation_analysis.png      # RQ2 analysis
│   ├── 👻 rq3_eeriness_trust.png          # RQ3 analysis
│   └── 📊 research_questions_summary.png   # Total summary
└── 📝 README.md               # This documentation
```

## 🔄 Data Processing

The analysis pipeline includes the following steps:

1. **Data Loading and Cleaning** (`data_cleaning.py`)
   - 📥 Loads survey data from multiple versions
   - ⏱️ Processes duration and response data
   - 🧹 Handles missing values and data type conversions
   - 🆔 Creates unique participant IDs

2. **Data Quality Checks**
   - 🚫 Removes fastest responders from specific conditions
   - 👥 Preserves participants who identify as "other" gender
   - ⚖️ Ensures balanced participant distribution

3. **Data Preparation**
   - 📊 Creates composite scores for:
     - 🤝 Credibility
     - 🤝 Trust
     - 👻 Eeriness
   - 📊 Categorizes scores into tertiles (low, medium, high)
   - 📈 Prepares data for statistical analysis

## ❓ Research Questions

The analysis addresses three main research questions:

### 🔄 RQ1: Consistency and Persuasion
- 🔍 Analyzes the relationship between VA consistency and opinion change
- ↔️ Compares consistent (male-male, female-female) vs. inconsistent conditions (male-female, female-male)
- 📊 Key findings:
  - 📈 Mixed effects model with condition as random effect
  - 📊 Fixed effect of consistency: β = 0.150 (SE = 0.250, p = 0.548)
  - 📈 Odds ratio: 1.162 for opinion change in consistent conditions
  - 🔄 Random effect variance: 0.044
  - ⚠️ Effect not statistically significant at conventional levels

### 🔗 RQ2: Credibility & Trust Mediation
- 🔍 Examines the mediating role of credibility in trust formation
- 📊 Key findings:
  - ✅ Significant path from consistency to credibility (β = 0.140, p = 0.042)
  - 💪 Strong relationship between credibility and trust (β = 0.846, p < 0.001)
  - ✅ Significant relationship between trust and change (β = 8.630, p = 0.007)

### 👻 RQ3: Inconsistency, Eeriness, and Trust
- 🔍 Investigates the relationship between inconsistency, eeriness, and trust
- 📊 Key findings:
  - ❌ No significant effect of consistency on eeriness (p = 0.848)
  - ✅ Significant effect of consistency on trust (β = 0.149, p = 0.046)
  - ❌ No significant interaction between eeriness and consistency (p = 0.210)

## 📊 Analysis Methods

The project employs various statistical and visualization techniques:

1. **Statistical Analysis**
   - 📊 ANOVA tests for composite scores
   - 📈 Mixed logistic regression for opinion change (accounting for condition-level random effects)
   - 🔗 Mediation analysis
   - 📈 Correlation analysis

2. **Visualization**
   - 📊 Bar charts for condition comparisons
   - 📦 Box plots for score distributions
   - 📊 Error bar plots for mean comparisons with confidence intervals
   - 📊 Multi-panel summary plots with model predictions

## 📊 Results

The analysis produces:
- 📑 Cleaned and processed dataset (`cleaned_data.csv`)
- 🎨 Seven visualization files:
  1. 📈 Composite scores by condition
  2. 📊 Tertile distributions
  3. 👥 Demographic distributions
  4. 🔄 RQ1: Consistency and persuasion analysis
  5. 🔗 RQ2: Mediation analysis
  6. 👻 RQ3: Eeriness and trust analysis
  7. 📊 Research questions summary
- 📊 Statistical test results
- ⚖️ Balance checks across conditions

## 🚀 Usage

To run the analysis:
1. 📦 Ensure required Python packages are installed
2. ▶️ Run `data_analysis.py`
3. 📊 Review generated visualizations and statistical outputs

## 📦 Dependencies

- 🐍 Python 3.x
- 📊 pandas
- 🔢 numpy
- 📈 matplotlib
- 🎨 seaborn
- 📊 scipy
- 📊 statsmodels

## 🔒 Data Privacy

The analysis maintains participant anonymity by:
- 🆔 Using unique participant IDs
- 📊 Aggregating data for analysis
- 🚫 Not storing personally identifiable information 