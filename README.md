# Virtual Agent Perception Analysis

This project analyzes survey data related to virtual agent (VA) perceptions across different conditions. The analysis focuses on three main research questions regarding consistency, credibility, trust, and eeriness in VA interactions.

## Project Structure

```
Research_final/
├── data_analysis.py      # Main analysis script
├── cleaned_data.csv      # Processed dataset
├── research_questions_analysis.png  # Visualizations of research questions
└── README.md            # This documentation
```

## Data Processing

The analysis pipeline includes the following steps:

1. **Data Loading and Cleaning**
   - Loads survey data from multiple versions
   - Processes duration and response data
   - Handles missing values and data type conversions
   - Creates unique participant IDs

2. **Data Quality Checks**
   - Removes fastest responders from specific conditions
   - Preserves participants who identify as "other" gender
   - Ensures balanced participant distribution

3. **Data Preparation**
   - Creates composite scores for:
     - Credibility
     - Trust
     - Eeriness
   - Categorizes scores into tertiles (low, medium, high)
   - Prepares data for statistical analysis

## Research Questions

The analysis addresses three main research questions:

### RQ1: Consistency and Persuasion
- Analyzes the relationship between VA consistency and opinion change
- Compares consistent vs. inconsistent conditions
- Visualizes opinion change patterns

### RQ2: Credibility & Trust Mediation
- Examines the mediating role of credibility in trust formation
- Analyzes trust scores across credibility categories
- Tests correlation between credibility and trust

### RQ3: Inconsistency, Eeriness, and Trust
- Investigates the relationship between inconsistency, eeriness, and trust
- Analyzes trust scores across eeriness categories
- Tests correlation between eeriness and trust

## Analysis Methods

The project employs various statistical and visualization techniques:

1. **Statistical Analysis**
   - ANOVA tests for composite scores
   - Chi-square tests for tertile distributions
   - Correlation analysis
   - Regression analysis

2. **Visualization**
   - Bar charts for condition comparisons
   - Scatter plots with regression lines
   - Error bar plots for mean comparisons
   - Distribution plots

## Results

The analysis produces:
- Cleaned and processed dataset (`cleaned_data.csv`)
- Visualizations of research questions (`research_questions_analysis.png`)
- Statistical test results
- Balance checks across conditions

## Usage

To run the analysis:
1. Ensure required Python packages are installed
2. Run `data_analysis.py`
3. Review generated visualizations and statistical outputs

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Data Privacy

The analysis maintains participant anonymity by:
- Using unique participant IDs
- Aggregating data for analysis
- Not storing personally identifiable information 