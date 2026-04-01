Here is the complete **Model Card** formatted in Markdown. You can copy this directly into a file named `model_card.md` in your project repository.

---

# Model Card - Census Income Prediction

## Model Details
* **Developed by:** Simon Jeffery
* **Model date:** March 2026
* **Model version:** 1.0.0
* **Model type:** **Random Forest Classifier**
* **Framework:** Scikit-learn
* **Information:** This model is a binary classifier trained on the 1994 Census database. It uses a series of demographic and financial features to predict if an individual's annual income exceeds **$50,000**. The model utilizes a `OneHotEncoder` for categorical preprocessing and a `LabelBinarizer` for the target variable.

## Intended Use
* **Primary intended uses:** This model was developed as part of an ML Ops pipeline project to demonstrate automated testing, API deployment with FastAPI, and CI/CD integration.
* **Primary intended users:** Data science researchers and socioeconomic analysts interested in legacy census trends.
* **Out-of-scope use cases:** This model **must not** be used to determine creditworthiness, insurance premiums, or hiring suitability. The data is historically biased and does not reflect the current 2026 economic landscape.

## Training Data
* **Source:** UCI Machine Learning Repository - [Census Income (Adult) Dataset](https://archive.ics.uci.edu/ml/datasets/adult).
* **Size:** Approximately 32,561 records (80% used for training).
* **Features:** 14 features including `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, and `hours-per-week`.
* **Preprocessing:** Cleaned for leading/trailing whitespace and missing values (marked as `?`).

## Evaluation Data
* **Details:** 20% of the dataset (approx. 6,512 records) was held out as a test set.
* **Method:** Stratified random split to ensure the distribution of the target class remains consistent between training and testing.

## Metrics
_The following metrics were achieved on the hold-out test set:_

| Metric | Value |
| :--- | :--- |
| **Precision** | [0.74] |
| **Recall** | [0.62] |
| **F1-Score** | [0.67] |

*The model was further evaluated on **data slices** for categorical features (e.g., Education, Race, Sex) to monitor for performance disparities across demographic groups.*

## Ethical Considerations
* **Data Bias:** The 1994 Census data contains inherent biases regarding gender and race, reflecting the social and systemic inequalities of that era. 
* **Fairness:** Automated slicing analysis was performed to identify if the model's predictive power varies significantly across protected classes.
* **Privacy:** The data is public, anonymized, and aggregated; no PII (Personally Identifiable Information) is included in the training process.

## Caveats and Recommendations
* **Temporal Drift:** Because the data is over 30 years old, the relationship between features like `education` and `income` has shifted. This model should be treated as a historical baseline rather than a modern predictor.
* **Categorical Imbalance:** Some countries and occupations have very few samples, leading to lower reliability in predictions for those specific "slices."
* **Recommendation:** For production-grade socioeconomic modeling in 2026, the model should be retrained on the latest American Community Survey (ACS) data.

---

**Would you like me to show you how to generate the specific numbers for the "Metrics" section using the `compute_model_metrics` function we wrote earlier?**