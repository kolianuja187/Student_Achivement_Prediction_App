
# **Student Achievement Prediction App – Machine Learning Project Report**

## **1. Introduction**

The **Student Achievement App** is a machine learning application designed to predict students’ academic performance based on various factors such as study habits, demographic information, and socio-economic status. This tool can assist educators, counselors, and academic institutions in identifying students at risk and implementing early interventions.

---

## **2. Project Objectives**

* Predict students’ academic achievement (grades or performance levels).
* Explore and understand the relationship between academic factors and student outcomes.
* Build a robust, scalable prediction pipeline.
* Provide a reusable machine learning model and structured preprocessing.

---

## **3. Project Structure and Files**

| **File Name**                      | **Description**                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `Students_Performance_Dataset.xls` | Raw dataset containing student performance and background information.                      |
| `Data_Cleaning1.ipynb`             | Jupyter Notebook used for initial data cleaning (handling nulls, formatting, encoding).     |
| `EDA-02.ipynb`                     | Exploratory Data Analysis notebook to uncover patterns and relationships between variables. |
| `academic_achievement_cleaned.xls` | Output file containing cleaned and preprocessed dataset.                                    |
| `Model_Selection (1).ipynb`        | Notebook for evaluating and selecting the best-performing machine learning models.          |
| `feature_columns (1).json`         | JSON file containing list of selected feature columns used for training the model.          |
| `StudentGradePipeline (1).pkl`     | Final trained model (pipeline) serialized using `pickle` for deployment.                    |

---

## **4. Data Cleaning and Preprocessing**

**Performed in:** `Data_Cleaning1.ipynb`

### Key Cleaning Steps:

* Removed or imputed missing values in critical columns (e.g., scores, attendance).
* Converted categorical features (e.g., gender, school type, parental education) using label encoding and one-hot encoding.
* Normalized numerical features like study time, absences.
* Exported the cleaned dataset as `academic_achievement_cleaned.xls`.

---

## **5. Exploratory Data Analysis (EDA)**

**Performed in:** `EDA-02.ipynb`

### Highlights:

* Visualized score distributions using histograms and box plots.
* Correlation heatmaps showed strong links between parental education, study time, and student performance.
* Clustered students based on performance tiers for better understanding of behavior and demographics.
* Identified outliers and anomalies in score data.

---

## **6. Feature Engineering**

* Selected the most impactful features from the dataset using feature importance metrics.
* Features were saved in `feature_columns (1).json` to ensure consistency during deployment.

**Key Features Included:**

* Gender, Age, Study Time, Parental Education, Lunch Type, Test Preparation Course, etc.

---

## **7. Model Selection and Training**

**Performed in:** `Model_Selection (1).ipynb`

### Models Evaluated:

* Logistic Regression (for classification)
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting
* XGBoost
* Support Vector Machine

### Evaluation Metrics:

* **Accuracy**
* **Precision/Recall**
* **F1-Score**
* **Confusion Matrix**

**Best Model:** Random Forest Classifier with highest accuracy and generalization ability

* Final model and preprocessing steps were wrapped into a pipeline and saved as `StudentGradePipeline (1).pkl`.

---

## **8. Deployment Readiness**

The model pipeline (`StudentGradePipeline (1).pkl`) can be integrated into a web app or API that:

* Accepts input from users (student data),
* Applies consistent preprocessing using saved `feature_columns`,
* Returns the predicted academic performance.

---

## **9. Results and Insights**

* Students who studied more than 2 hours daily and had completed test preparation courses consistently performed better.
* Parental education and access to free lunch showed correlations with performance gaps.
* The app can help identify students needing support based on their predicted achievement level.

---

## **10. Future Improvements**

* Deploy model using Flask or Streamlit for interactive use.
* Expand dataset with real-time or more recent records.
* Include temporal data (e.g., performance over time).
* Introduce deep learning for complex classification tasks.

---

## **11.output

<img width="1036" height="800" alt="image" src="https://github.com/user-attachments/assets/51d689ec-dce8-4ba3-88c9-d72d7fe4b80c" />


## **12. Conclusion**

The **Student Achievement App** effectively uses machine learning to predict academic performance with high accuracy. This project demonstrates the power of data-driven education systems and highlights the potential of ML tools to support student success through early intervention and informed decision-making.

---

