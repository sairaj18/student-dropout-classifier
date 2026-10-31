# Student Dropout & Academic Success – Classification and Streamlit Deployment

## a. Problem Statement

Higher-education institutions lose students to dropout, and identifying at-risk students early allows timely intervention. This project builds and compares five machine learning classifiers that predict a student's final outcome (**Dropout**, **Enrolled**, or **Graduate**) from demographic, socio-economic, and academic information available at enrolment and after the first two semesters. The best-performing models are packaged into an interactive Streamlit web app that evaluates any uploaded test set using pre-trained models.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository – *Predict Students' Dropout and Academic Success* (dataset id 697)
- **Type:** Multi-class classification (3 classes)
- **Instances:** 4,424
- **Features:** 36 (demographics, application details, socio-economic indicators, and curricular units/grades for the 1st and 2nd semesters)
- **Target column:** `Target` – `Graduate` (2,209), `Dropout` (1,421), `Enrolled` (794)
- **Class balance:** Moderately imbalanced; `Enrolled` is the minority class.
- **Missing values:** None.
- **Train/test split:** 80/20 stratified split with `random_state=42` (3,539 training rows, 885 test rows). The test set is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` is applied inside the pipelines of Logistic Regression, KNN and SVM (scale-sensitive models). Decision Tree and Naive Bayes use raw features. The target is label-encoded with `LabelEncoder`.

## c. GitHub Repository Link

https://github.com/sairaj18/student-dropout-classifier

### Repository structure

```
student-outcome-classifier/
│-- app.py                  # Streamlit app
│-- train_models.py         # trains, evaluates and saves all 5 models
│-- metrics_utils.py        # shared evaluation-metric helper
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test data (features + Target)
│-- students_raw.csv        # full dataset
│-- model/
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- support_vector_machine.joblib
│   │-- label_encoder.joblib
│   │-- feature_columns.joblib
│   └-- results_table.csv
```

### How to run

```bash
pip install -r requirements.txt
python train_models.py      # trains models and writes model/ and test_data.csv
streamlit run app.py        # launches the app
```

## d. Models Used

All five models were trained on the same training split and evaluated on the same held-out test set (885 rows). Precision, Recall and F1 use **weighted averaging**; AUC is the **one-vs-rest weighted** AUC computed from predicted class probabilities.

### Comparison Table

| ML Model Name          | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
|:-----------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression    |   0.7684 | 0.8961 |    0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree          |   0.7288 | 0.8347 |    0.7264 | 0.7288 | 0.7250 | 0.5542 |
| KNN                    |   0.6734 | 0.8154 |    0.6530 | 0.6734 | 0.6576 | 0.4529 |
| Naive Bayes            |   0.6599 | 0.8131 |    0.6347 | 0.6599 | 0.6432 | 0.4303 |
| Support Vector Machine |   0.7582 | 0.8865 |    0.7465 | 0.7582 | 0.7454 | 0.5986 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|:--------------|:------------------------------------|
| Logistic Regression | Best model overall, leading on all six metrics (Accuracy 0.768, AUC 0.896, MCC 0.615). With standardised features it fits this dataset well, which suggests the class boundaries are largely linear in the academic-performance features. It is also fast, stable and easy to interpret. |
| Decision Tree | Third-best on most metrics (Accuracy 0.729, MCC 0.554). It captures some non-linear rules, but its AUC (0.835) is clearly below Logistic Regression and SVM because a single tree gives coarse, step-like probability estimates. Depth was limited to 8 to control overfitting. |
| KNN | Weak performance (Accuracy 0.673, MCC 0.453). With 36 features, distances become less informative (curse of dimensionality), and many features are binary or categorical codes, so Euclidean distance is a poor similarity measure. It is also sensitive to the class imbalance, since majority-class neighbours dominate. |
| Naive Bayes | Lowest performance (Accuracy 0.660, MCC 0.430). The Gaussian and feature-independence assumptions are violated here, since many features are strongly correlated (for example, 1st and 2nd semester units approved and grades) and many are not normally distributed. It remains very fast to train. |
| Support Vector Machine | Close second to Logistic Regression (Accuracy 0.758, AUC 0.887, MCC 0.599). The RBF kernel handles non-linearity well after scaling, but with default hyperparameters it does not beat the linear model. Tuning `C` and `gamma` may improve it, at a higher training cost. |
| **Overall, winner for this dataset?** | **Logistic Regression** – it achieves the highest Accuracy, AUC, Precision, Recall, F1 and MCC, while being the simplest and fastest of the strong models. SVM is a close runner-up, while KNN and Naive Bayes are the weakest. |

## Streamlit App Features

- **CSV upload** of the test data (must contain all 36 feature columns and the `Target` column)
- **Model selection dropdown** for the five trained models
- **Evaluation metrics** display: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** and **classification report** for the selected model
- **Comparison table and chart** of all five models on the uploaded data

The app loads the saved models from `model/` and does not retrain on the uploaded data.
