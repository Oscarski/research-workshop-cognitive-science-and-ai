# 🔬 Virtual Agent Perception Analysis

This project analyzes survey data related to virtual agent (VA) perceptions across different conditions. The analysis focuses on the relationship between consistency, credibility, and opinion change in VA interactions.

## 📁 Project Structure

```
Research_final/
├── 📊 data_analysis.py         # Main analysis script
├── 🧹 data_cleaning.py         # Data cleaning and preparation functions
├── 📑 cleaned_data.csv         # Processed dataset
├── 📂 Version 1-4/*.csv        # Raw survey data files
├── 🎨 Visualizations/
│   ├── 📈 composite_scores.png    # Composite scores by condition
│   ├── 👥 demographics.png       # Age and gender distributions
│   └── 🔄 rq1_consistency_persuasion.png  # Opinion change by consistency
└── 📝 README.md               # This documentation
```

## 🔄 Data Processing

The analysis pipeline includes the following steps:

1. **Data Loading and Cleaning** (`data_cleaning.py`)  
   * 📥 Loads survey data from multiple versions  
   * ⏱️ Processes duration and response data  
   * 🧹 Handles missing values and data type conversions  
   * 🆔 Creates unique participant IDs

2. **Data Quality Checks**  
   * 🚫 Removes fastest responders from specific conditions  
   * 👥 Preserves participants who identify as "other" gender  
   * ⚖️ Ensures balanced participant distribution

3. **Data Preparation**  
   * 📊 Creates composite scores for:  
         * 🤝 Credibility  
         * 🤝 Trust  
         * 👻 Eeriness  
   * 📊 Categorizes scores into tertiles (low, medium, high)

## ❓ Research Questions

The analysis addresses two main research questions:

### 🔄 RQ1: Consistency and Persuasion

* 🔍 Analyzes the relationship between VA consistency and opinion change
* ↔️ Compares consistent (male-male, female-female) vs. inconsistent conditions (male-female, female-male)
* 📊 Key findings:  
   * 📈 Logistic regression model
   * 📊 Coefficient for consistency: β = 0.767 (p = 0.293)
   * 📈 Odds ratio: 2.15 for opinion change in consistent conditions
   * ⚠️ Effect not statistically significant at conventional levels

### 🔗 RQ2: Credibility Analysis

* 🔍 Examines the relationship between consistency and credibility
* 📊 Key findings:  
   * 📈 Ordinal regression model
   * 📊 Category 1 (low to medium): β = 0.811 (p = 0.313)
   * 📊 Category 2 (medium to high): β = 1.622 (p = 0.056)
   * ✅ Trend toward higher credibility in consistent conditions

## 📊 Analysis Methods

The project employs various statistical and visualization techniques:

1. **Statistical Analysis**  
   * 📊 Logistic regression for opinion change
   * 📈 Ordinal regression for credibility analysis
   * 📊 Descriptive statistics for composite scores

2. **Visualization**  
   * 📊 Box plots for composite scores by condition
   * 📈 Bar charts for opinion change analysis
   * 📊 Demographics plots

## 📊 Results

The analysis produces:

* 📑 Cleaned and processed dataset (`cleaned_data.csv`)
* 🎨 Three visualization files:  
   1. 📈 Composite scores by condition  
   2. 👥 Demographic distributions  
   3. 🔄 Opinion change by consistency analysis

## 🚀 Usage

To run the analysis:

1. 📦 Ensure required Python packages are installed
2. ▶️ Run `data_analysis.py`
3. 📊 Review generated visualizations and statistical outputs

## 📦 Dependencies

* 🐍 Python 3.x
* 📊 pandas
* 🔢 numpy
* 📈 matplotlib
* 🎨 seaborn
* 📊 scipy
* 📊 statsmodels

## 🔒 Data Privacy

The analysis maintains participant anonymity by:

* 🆔 Using unique participant IDs
* 📊 Aggregating data for analysis
* 🚫 Not storing personally identifiable information 