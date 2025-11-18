# Healthcare Data Analysis and Prediction Notebook

This notebook demonstrates a comprehensive analysis of a healthcare dataset, including data loading, exploratory data analysis (EDA), data preprocessing, supervised machine learning for predicting test results, anomaly detection in billing amounts, and setting up an AI-powered doctor recommendation generator.

## 1. Load and Inspect Data
- The `healthcare_dataset.csv` file is loaded into a pandas DataFrame named `df`.
- Initial inspection includes displaying the first 5 rows (`df.head()`), a concise summary of the DataFrame (`df.info()`), and descriptive statistics for numerical columns (`df.describe()`).

## 2. Exploratory Data Analysis (EDA)
- Visualizations are created using `matplotlib` and `seaborn` to understand data distributions and frequencies.
- **Histograms** are generated for numerical features:
    - `Age`: Distribution of patient ages.
    - `Billing Amount`: Distribution of billing amounts.
    - `Room Number`: Distribution of room numbers.
- **Count plots** are generated for key categorical features:
    - `Medical Condition`: Frequency of different medical conditions.
    - `Admission Type`: Frequency of various admission types.
    - `Medication`: Frequency of prescribed medications.
- *Note: Addressed `FutureWarning` in `sns.countplot` by explicitly setting `hue` and `legend=False`.*

## 3. Prepare Data for Supervised Learning
- **Date Feature Engineering**:
    - `Date of Admission` and `Discharge Date` columns are converted to datetime objects.
    - A new feature, `Length of Stay`, is calculated as the difference in days between `Discharge Date` and `Date of Admission`.
- **Column Dropping**:
    - Irrelevant columns such as `Date of Admission`, `Discharge Date`, `Name`, `Doctor`, `Hospital`, and `Insurance Provider` are dropped from the DataFrame.
- **Categorical Feature Encoding**:
    - One-hot encoding is applied to all remaining categorical columns (e.g., `Gender`, `Blood Type`, `Medical Condition`, `Admission Type`, `Medication`) using `pd.get_dummies()`.
    - The target variable, `Test Results`, is label encoded using `sklearn.preprocessing.LabelEncoder` to convert 'Abnormal', 'Inconclusive', 'Normal' into numerical values (e.g., 0, 1, 2).
- **Feature and Target Separation**:
    - The preprocessed data is split into features (`X`) and the target variable (`y`).

## 4. Train and Evaluate Supervised Model
- **Data Splitting**:
    - The dataset is split into training (80%) and testing (20%) sets using `train_test_split` from `sklearn.model_selection` with `random_state=42` for reproducibility.
- **Model Training**:
    - A `RandomForestClassifier` is initialized and trained on the training data (`X_train`, `y_train`).
- **Model Evaluation**:
    - Predictions are made on the test set (`X_test`).
    - **Accuracy Score**: Overall model accuracy is calculated.
    - **Classification Report**: Precision, recall, and F1-score for each class ('Abnormal', 'Inconclusive', 'Normal') are provided.
    - **Confusion Matrix**: A heatmap visualization of the confusion matrix is generated to show true vs. predicted labels.

## 5. Anomaly Detection in Billing Amounts
- **Isolation Forest Model**:
    - An `IsolationForest` model is used to detect anomalies in the `Billing Amount` column.
    - `contamination` is set to `0.01` (1% anomalies).
- **Anomaly Identification**:
    - The model fits to the `Billing Amount` and identifies 545 anomalous entries.
- **Findings**:
    - Anomalies include both negative and extremely high billing amounts, indicating potential data quality issues, errors, or unusual financial scenarios.

## 6. AI Doctor Recommendation Generator Setup
- **Gemini API Integration**:
    - The notebook configures the `google.generativeai` library to use the `gemini-2.5-flash` model for generating doctor recommendations.
- **`generate_doctor_recommendation` Function**:
    - A function is defined to create concise, clinical-style health advice based on `predicted_test_result`, `age`, `medical_condition`, and `medication`.
    - It formulates a prompt for the LLM and handles potential API call exceptions gracefully.
- **Example Recommendation**:
    - An example run demonstrates how the function generates a recommendation for a patient with 'Normal' test results, 'Seasonal Allergies', and 'Loratadine PRN' medication.
