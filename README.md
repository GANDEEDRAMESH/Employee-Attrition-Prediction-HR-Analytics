# Employee Attrition Prediction & HR Analytics

## Project Overview

This project analyzes an HR dataset to build predictive models that help organizations understand and predict employee attrition. The analysis provides data-driven insights for the Human Resources department to improve employee retention strategies.

## Business Problem

**Research Question:** What factors are most likely to lead an employee to leave the company?

Salifort Motors sought to understand employee satisfaction and retention patterns in their organization. The ability to predict and address potential attrition provides significant value, as recruiting, interviewing, and training new employees is both time-consuming and costly.

## Dataset

- **Source:** [Kaggle HR Analytics Dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)
- **Rows:** 14,999 employees (after removing duplicates: 11,991)
- **Features:** 10 variables

### Features Description

| Variable | Description |
|----------|-------------|
| `satisfaction_level` | Employee-reported job satisfaction level (0–1) |
| `last_evaluation` | Score of employee's last performance review (0–1) |
| `number_project` | Number of projects the employee contributes to |
| `average_monthly_hours` | Average number of hours worked per month |
| `tenure` | Employee tenure in years |
| `work_accident` | Whether employee experienced an accident at work (0/1) |
| `left` | **Target Variable** - Whether employee left company (0/1) |
| `promotion_last_5years` | Whether employee was promoted in last 5 years (0/1) |
| `department` | Employee's department |
| `salary` | Employee's salary tier (low, medium, high) |

## Key Findings

### Data Insights

1. **Workload & Attrition Correlation**
   - Employees working 240+ hours/month showed significantly higher attrition
   - All employees assigned to 7 projects left the company
   - Optimal project load: 3-4 projects (low attrition rate)

2. **Satisfaction & Retention**
   - Mean satisfaction for employees who left: 0.44
   - Mean satisfaction for employees who stayed: 0.67
   - Strong negative correlation between satisfaction and attrition

3. **Tenure Patterns**
   - Employees with 4+ years tenure who left showed unusually low satisfaction
   - Long-tenured employees (6+ years) rarely left
   - Relatively few long-tenure employees suggest higher-paying positions

4. **Evaluation & Promotion**
   - Overworked employees with high evaluations were more likely to leave (burnout)
   - Very few employees working extreme hours were promoted
   - Disconnect between hard work and career advancement

5. **Workload Reality**
   - Benchmark: ~167 hours/month (40-hour weeks, 2-week vacation)
   - Most employees work 200+ hours/month (well above benchmark)
   - Indicates systemic overwork across the organization

## Methodology

### Data Processing
- **Data Cleaning:** Removed 3,008 duplicate rows (20% of data)
- **Outlier Detection:** Identified tenure outliers using IQR method
- **Feature Encoding:**
  - Ordinal encoding for `salary` (preserves hierarchy)
  - One-hot encoding for `department`

### Exploratory Data Analysis (EDA)
- Univariate analysis: distributions of key variables
- Bivariate analysis: relationships with attrition
- Correlation analysis: feature relationships
- Visualizations: boxplots, histograms, scatterplots, heatmaps

### Modeling Approaches

#### 1. **Logistic Regression Model**
- **Assumptions Checked:**
  - Binary outcome variable ✓
  - Independent observations ✓
  - No severe multicollinearity ✓
  - Sufficiently large sample size ✓
  
- **Performance:**
  - Accuracy: 82%
  - Precision: 79%
  - Recall: 82%
  - F1-Score: 80%

#### 2. **Decision Tree Model**
- **Grid Search Parameters:**
  - `max_depth`: [4, 6, 8, None]
  - `min_samples_leaf`: [1, 2, 5]
  - `min_samples_split`: [2, 4, 6]
  
- **Best Parameters:** (Determined via cross-validation)
  - Strong cross-validation scores across all metrics

#### 3. **Random Forest Model**
- **Grid Search Parameters:**
  - `max_depth`: [3, 5, None]
  - `max_features`: [1.0]
  - `max_samples`: [0.7, 1.0]
  - `min_samples_leaf`: [1, 2, 3]
  - `min_samples_split`: [2, 3, 4]
  - `n_estimators`: [300, 500]
  
- **Advantages:**
  - Reduces overfitting compared to single decision trees
  - Multiple trees provide more robust predictions

## Files in Repository

```
.
├── README.md                                              # This file
├── Employee Attrition Prediction & HR Analytics.ipynb    # Main analysis notebook
├── HR_comma_sep.csv                                       # Raw dataset
└── [Additional resources and outputs]
```

## How to Use

### Prerequisites
```
Python 3.7+
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

### Installation
```bash
# Clone the repository
git clone https://github.com/GANDEEDRAMESH/Employee-Attrition-Prediction-HR-Analytics.git

# Navigate to the project directory
cd Employee-Attrition-Prediction-HR-Analytics

# Install required packages
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Running the Analysis
1. Open `Employee Attrition Prediction & HR Analytics.ipynb`
2. Run cells sequentially from top to bottom
3. Review EDA visualizations and model performance metrics
4. Examine feature importance and model predictions

## Key Libraries Used

- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, xgboost
- **Metrics:** scikit-learn (classification_report, confusion_matrix, roc_auc_score)

## Recommendations for HR

Based on the analysis, the following actions can improve employee retention:

1. **Workload Management**
   - Cap projects at 4 per employee
   - Maintain 160-200 hours/month as target
   - Identify and redistribute overloaded assignments

2. **Compensation & Promotion**
   - Link promotions more directly to performance and tenure
   - Review salary progression, especially for tenure-based roles
   - Ensure high performers receive career advancement

3. **Satisfaction Monitoring**
   - Regularly assess employee satisfaction levels
   - Investigate specific events around the 4-year mark
   - Implement interventions for declining satisfaction

4. **Department-Level Review**
   - While attrition varies by department, review individual roles
   - Identify burnout patterns within teams
   - Support managers in workload distribution

## Model Selection & Performance

The Random Forest and Decision Tree models outperformed Logistic Regression due to their ability to capture non-linear relationships in employee behavior. The ensemble approach of Random Forest provides robustness against overfitting while maintaining high predictive accuracy.

**Recommended Model:** Random Forest (balances accuracy, interpretability, and generalization)

## Ethical Considerations

- **Privacy:** All data is anonymized and aggregated
- **Fairness:** Models evaluated for potential bias across departments
- **Transparency:** Model predictions can be explained through feature importance
- **Application:** Predictions should support retention efforts, not punitive actions

## Limitations

1. Dataset appears to contain some synthetic/manipulated values (unusual distribution shapes)
2. Cross-sectional data; temporal patterns not captured
3. Missing context on major organizational events
4. Limited demographic variables for bias assessment

## Future Work

- Time-series analysis of employee satisfaction trends
- Deep learning models for improved predictions
- Explainability analysis (SHAP, LIME) for model interpretability
- Clustering analysis to identify employee personas
- Causal analysis to determine true drivers vs. correlations

## Resources & References

- [Kaggle HR Analytics Dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Employee Attrition Analysis Best Practices](https://en.wikipedia.org/wiki/Attrition_(employment))

## Author

**GANDEEDRAMESH**

## License

This project is open source and available under the MIT License.

## Contact & Support

For questions or feedback, please open an issue on the GitHub repository.

---

**Last Updated:** December 2025

**Project Status:** Under Progress
