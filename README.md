# NYPD Shooting Incident Analysis: Relational Data Modeling & Predictive Analytics

## 1. Project Overview
This project presents a rigorous analysis of New York City’s historical shooting data to identify patterns in gun violence and assess the predictability of incident outcomes. Utilizing a relational database approach, the study investigates two primary domains:
1.  **Temporal Dynamics**: Identifying if incident characteristics can predict the `TIME_OF_DAY`.
2.  **Severity Assessment**: Developing a binary classifier to predict the probability of a fatal outcome (`STAT_MURDER_FLG`).

## 2. Data Engineering and Lineage
The project utilizes the **NYPD Shooting Incident Data (Historic)**. To ensure a comprehensive feature set for machine learning, a relational integration strategy was employed across three primary source entities:

* **Source Entities**:
    * **Shootings (Transaction Table)**: Contains the unique `INCIDENT_KEY`, location-based data (Borough, Precinct), and coordinate data.
    * **Shootings_Offenders**: Detailed demographic and descriptive profiles of the suspected perpetrators.
    * **Shootings_Victims**: Detailed victim demographics and the definitive binary flag for mortality (`STAT_MURDER_FLG`).
* **ETL & Join Logic**: An **Inner Join** was utilized across these three tables to create a "Gold Standard" analytical dataset. While the raw shooting data comprised ~35,000 records, this strategy yielded **23,000 high-fidelity rows** containing verified data across all dimensions (Location, Victim, and Offender).



## 3. Data Dictionary and Feature Selection
Data was curated based on official NYPD specifications to ensure domain accuracy:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `BORO` | Categorical | Borough where the incident occurred (BRONX, BROOKLYN, etc.) |
| `VIC_AGE_GROUP` | Categorical | Specific age bin of the victim at the time of the incident |
| `STAT_MURDER_FLG` | Binary | Target variable: Indicates if the shooting resulted in a fatality |
| `PRECINCT` | Numerical | Specific NYPD Precinct of occurrence |

## 4. Machine Learning Methodology

### Phase I: Temporal Prediction (Multiclass)
**Objective**: Predict the categorical `TIME_OF_DAY` (Morning, Afternoon, Evening, Night, Late Night).
* **Models Evaluated**: Multinomial Logistic Regression and Random Forest Classifier.
* **Key Findings**: Accuracy reached a baseline of **35.88%**. Analysis of the confusion matrix indicated that the model developed a bias toward the "Late Night" class, which significantly outnumbered other categories, representing a classic class imbalance challenge.

### Phase II: Severity Prediction (Binary)
**Objective**: Predict whether an incident results in death (`STAT_MURDER_FLG`).
* **Imbalance Handling**: The dataset exhibited a 5:1 ratio of non-fatal to fatal incidents.
* **Model Optimization**: Implemented a **Random Forest Classifier** with `class_weight='balanced'`. 
* **Results**: Achieved an **Accuracy of 56.99%**.
* **Performance Insight**: By prioritizing **Recall (0.56)** over pure Accuracy, the model was tuned to minimize False Negatives—ensuring that fatal incidents are detected even at the cost of higher False Positives.



## 5. Model Interpretability and Feature Importance
Using Random Forest's Gini Importance, the model identified that **Borough** and **Victim Age** are stronger indicators of incident severity than the specific time of occurrence. This suggests that socioeconomic and geographic factors carry more predictive weight in lethality than temporal environmental factors.

## 6. Technical Stack
* **Environment**: Python 3.x, Jupyter Notebook.
* **Libraries**: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`.
* **Standards**: NYPD Open Data Governance Standards.

## 7. Future Work and Optimization
* **Resampling**: Implementing **SMOTE** (Synthetic Minority Over-sampling Technique) to address the minority class scarcity in "Morning" shootings.
* **Feature Expansion**: Re-evaluating the join strategy to recover the 12,000 records lost during the inner join through a **Left Join** strategy with missing-value imputation.

---
**Author**: Mohit Acharya
