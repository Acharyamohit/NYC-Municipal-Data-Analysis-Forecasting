# NYC-Municipal-Data-Analysis-Forecasting

# Predictive Analysis of NYC Shooting Incidents: Temporal and Severity Modeling

## Project Overview
This repository contains a comprehensive data science workflow analyzing the NYPD Shooting Incident Data (Historic). The project addresses two primary classification challenges: 
1. **Temporal Prediction**: Classifying the time of day (Morning, Afternoon, Evening, Night, Late Night) based on incident characteristics.
2. **Severity Prediction**: Developing a binary classifier to predict the lethality of an incident (`STAT_MURDER_FLG`).

## Dataset and Data Engineering
The analysis is based on a refined dataset of approximately 23,000 records derived from the NYC Open Data portal.

### Data Integration and Cleaning
* **Join Logic**: Data was integrated from three source tables (Incidents, Victims, and Location). An inner join was utilized to ensure data integrity across all features, resulting in 23,000 high-fidelity rows.
* **Handling Missing Values**: Categorical nulls were audited; features like Victim Sex and Race showed significant skews (e.g., 26,126 Males vs 2,610 Females).
* **Feature Engineering**: 
    * Raw timestamps were binned into five categorical time slots.
    * `pd.get_dummies` with `drop_first=True` was implemented for categorical encoding to prevent multicollinearity in linear models.

## Predictive Modeling and Evaluation

### Phase 1: Temporal Classification
Both **Multinomial Logistic Regression** and **Random Forest** models were evaluated.
* **Accuracy Baseline**: Both models converged at approximately 35-36% accuracy.
* **Diagnostic**: Confusion matrix analysis revealed a heavy bias toward "Late Night" predictions. This stems from class imbalance where the model defaults to the majority class to maximize global accuracy.

### Phase 2: Severity Classification (Fatal vs. Non-Fatal)
Given the 5:1 imbalance (23,966 Non-Fatal vs 4,781 Fatal), standard accuracy was discarded as a success metric in favor of **Recall** and **F1-Score**.
* **Model**: Random Forest Classifier with `class_weight='balanced'`.
* **Result**: Achieved 56.99% accuracy.
* **Trade-off Analysis**: By balancing the model, we achieved a 0.56 Recall for fatalities. While this lowered overall precision (0.23), it significantly reduced False Negatives, which is critical in public safety modeling.

## Key Visualizations
* **Geospatial Scatter Plots**: Visualizing incidents by Latitude and Longitude to identify high-density clusters.
* **Confusion Matrices**: Used to diagnose model "laziness" and class bias.
* **Feature Importance**: Identifying which demographic or geographic variables most heavily influence the model's decision-making process.

## Technical Environment
* **Language**: Python 3.13
* **Libraries**: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
* **Tools**: Jupyter Notebook, Anaconda Distribution

## Future Improvements
* **Synthetic Over-sampling**: Applying SMOTE to the training set to increase the signal of minority classes (Morning shootings and Fatalities).
* **Hyperparameter Optimization**: Implementing `GridSearchCV` to tune `n_estimators` beyond the current baseline of 200.
* **Recovery of Dropped Records**: Re-evaluating the join strategy (Left Join) to recover the 12,000 records lost during initial data integration.

