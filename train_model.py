import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter dependency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.combine import SMOTEENN
import joblib
import warnings
import time  # Added for measuring computation time

warnings.filterwarnings('ignore')
np.random.seed(42)

# Record the start time for overall computation
overall_start_time = time.time()

# === 1. LOAD & PREPARE DATA (CRISP-DM: Business Understanding, Data Understanding) ===
print("=== Data Understanding ===")
# Load the dataset (150,000 instances)
data = pd.read_csv('Final_healthcare_big_data.csv')
print("Dataset loaded successfully.")

# Data Overview Summary
print("\nData Overview Summary:")
print(f"Dataset Shape: {data.shape}")  # Should show (150000, number_of_columns)
print("\nBasic Statistics for Numerical Columns:")
print(data.describe())
print("\nColumn Information:")
print(data.info())
print("\nClass Distribution of Target Variable ('Outcome'):")
print(data['Outcome'].value_counts())  # Should show Improved: 9045, Unchanged: 52965, Worsened: 87990

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)
print("\nStatement: The dataset has no missing values." if missing_values.sum() == 0 else "\nStatement: The dataset contains missing values.")

# Check for duplicates and drop them
initial_shape = data.shape
data = data.drop_duplicates()
final_shape = data.shape
print(f"\nInitial dataset shape: {initial_shape}")
print(f"Dataset shape after dropping duplicates: {final_shape}")
print(f"Number of duplicates dropped: {initial_shape[0] - final_shape[0]}")

# Visualize dataset datatypes
print("\nDataset Column Datatypes:")
print(data.dtypes)
dtype_counts = data.dtypes.value_counts()
print("\nDatatype Counts:")
print(dtype_counts)

# Figure: Visualize column datatypes
plt.figure(figsize=(8, 6))
sns.barplot(x=dtype_counts.values, y=dtype_counts.index, hue=dtype_counts.index, palette='viridis', legend=False)
plt.title('Dataset Column Datatypes (Two Object and Int64 Types)')
plt.xlabel('Number of Columns')
plt.ylabel('Datatype')
plt.tight_layout()
plt.savefig('dataset_datatypes.png')
plt.close()
print("\nFigure saved as 'dataset_datatypes.png' showing columns with two object and int64 datatypes.")

# Feature engineering (CRISP-DM: Data Preparation)
print("\n=== Data Preparation ===")
data['Resource_Bed_Interaction'] = data['Resource_Utilization'] * data['Bed_Occupancy_Rate']
data['Wait_Length_Interaction'] = data['Wait_Time_Minutes'] * data['Length_of_Stay']
data['Log_Wait_Time'] = np.log1p(data['Wait_Time_Minutes'])
data['Log_Resource_Utilization'] = np.log1p(data['Resource_Utilization'])
data['Satisfaction_Staff_Interaction'] = data['Satisfaction_Rating'] * data['Staff_Count']
print("Feature engineering completed.")

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nLabel Encoding Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# === 2. PREPROCESSING (CRISP-DM: Data Preparation) ===
numerical_features = ['Daily_Patient_Inflow', 'Emergency_Response_Time_Minutes', 'Staff_Count', 
                      'Bed_Occupancy_Rate', 'Visit_Frequency', 'Wait_Time_Minutes', 
                      'Length_of_Stay', 'Previous_Visits', 'Resource_Utilization', 'Age',
                      'Resource_Bed_Interaction', 'Wait_Length_Interaction', 'Log_Wait_Time', 
                      'Log_Resource_Utilization', 'Satisfaction_Staff_Interaction']
categorical_features = ['Treatment_Outcome', 'Equipment_Availability', 'Medicine_Stock_Level', 
                        'Comorbidities', 'Disease_Category']
ordinal_features = ['Satisfaction_Rating']

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                     ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
ord_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('scaler', StandardScaler())])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numerical_features),
    ('cat', cat_pipe, categorical_features),
    ('ord', ord_pipe, ordinal_features)
])

X_processed = preprocessor.fit_transform(X)
cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = numerical_features + list(cat_feature_names) + ordinal_features
print(f"\nNumber of features after preprocessing: {len(feature_names)}")
print("Feature names:", feature_names)
print(f"Shape of X_processed: {X_processed.shape}")

# Drop redundant features
X_df = pd.DataFrame(X_processed, columns=feature_names)
X_df.drop(['Wait_Time_Minutes', 'Resource_Utilization'], axis=1, inplace=True, errors='ignore')
print("\nDropped redundant features ('Wait_Time_Minutes', 'Resource_Utilization').")
print(f"Shape of X_df after dropping redundant features: {X_df.shape}")

# === 3. BALANCE DATA AND SPLIT INTO TRAINING, VALIDATION, AND TEST SETS (CRISP-DM: Data Preparation) ===
smote_enn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smote_enn.fit_resample(X_df.values, y_encoded)
print("\nClass distribution after SMOTEENN balancing:")
print(pd.Series(y_balanced).value_counts())

# Split data into training, validation, and test sets (60-20-20 split)
# First, split into training+validation (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, 
                                                  stratify=y_balanced, random_state=42)

# Then, split training+validation into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, 
                                                  stratify=y_temp, random_state=42)

print(f"\nShape of X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"Class distribution in training set:\n{pd.Series(y_train).value_counts()}")
print(f"Class distribution in validation set:\n{pd.Series(y_val).value_counts()}")
print(f"Class distribution in test set:\n{pd.Series(y_test).value_counts()}")

# === 4. DEFINE AND TRAIN STACKING CLASSIFIER (CRISP-DM: Modeling) ===
print("\n=== Modeling ===")
class_weights = {0: 5.0, 1: 1.0, 2: 1.0}
rf_model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=2, max_features='log2', 
                                  class_weight=class_weights, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, 
                          n_estimators=1000, max_depth=10, learning_rate=0.1, 
                          subsample=0.9, colsample_bytree=1.0, reg_lambda=1, reg_alpha=1)
lgbm_model = LGBMClassifier(class_weight=class_weights, random_state=42)
catboost_model = CatBoostClassifier(verbose=0, early_stopping_rounds=10, auto_class_weights='Balanced')

stacking = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('cat', catboost_model)
    ],
    final_estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    cv=5, n_jobs=1
)

print(f"Processing Training Set: Training Stacking Classifier on {X_train.shape[0]} instances...")
stacking_start_time = time.time()  # Record start time for Stacking Classifier
stacking.fit(X_train, y_train)
stacking_end_time = time.time()  # Record end time for Stacking Classifier
stacking_training_time = stacking_end_time - stacking_start_time
print(f"Training time for Stacking Classifier: {stacking_training_time:.2f} seconds")
print("Stacking classifier trained successfully.")

# Evaluate on validation set (for monitoring)
print(f"Processing Validation Set: Evaluating Stacking Classifier on {X_val.shape[0]} instances...")
y_val_pred = stacking.predict(X_val)
print("\nValidation Classification Report for Stacking Classifier:")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

# === 5. EVALUATION (CRISP-DM: Evaluation) ===
print("\n=== Evaluation ===")
print(f"Processing Test Set: Evaluating Stacking Classifier on {X_test.shape[0]} instances...")
y_pred = stacking.predict(X_test)
y_pred_proba = stacking.predict_proba(X_test)

print("Evaluation Phase: Generating classification report for Stacking Classifier on test set...")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === 6. THRESHOLD ADJUSTMENT ===
print("\n=== Threshold Adjustment ===")
thresholds = {}
y_pred_adj = np.copy(y_pred)
for cls in range(3):
    precision, recall, thresh = precision_recall_curve(y_test == cls, y_pred_proba[:, cls])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    thresholds[cls] = thresh[np.argmax(f1_scores)]

for i in range(len(y_pred_proba)):
    for cls in range(3):
        if y_pred_proba[i, cls] >= thresholds[cls]:
            y_pred_adj[i] = cls
            break

print("Evaluation Phase: Generating classification report after threshold adjustment...")
print("\nClassification Report after Threshold Adjustment:")
print(classification_report(y_test, y_pred_adj, target_names=label_encoder.classes_))

# === 7. SHAP INTERPRETABILITY (CRISP-DM: Evaluation) ===
print("\nComputing SHAP values for interpretability...")
xgb_model.fit(X_train, y_train)  # Retrain for SHAP
shap_explainer = shap.TreeExplainer(xgb_model)
X_test_df = pd.DataFrame(X_test, columns=X_df.columns)
shap_values = shap_explainer.shap_values(X_test_df)

# SHAP Feature Importance
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig('shap_feature_importance.png')
plt.close()

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.close()

# === 8. VISUALIZATIONS (CRISP-DM: Evaluation) ===
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_adj)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Threshold Adjusted)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# ROC Curves
plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

# === 9. SAVE MODELS (CRISP-DM: Deployment) ===
print("\n=== Deployment ===")
joblib.dump(stacking, 'final_healthcare_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\nModels and preprocessors saved successfully.")

# Record the end time for overall computation
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"\nOverall Computation Time: {overall_time:.2f} seconds")

# Save training time for thesis documentation
training_times_df = pd.DataFrame({'Model': ['Stacking Classifier'], 'Training Time (seconds)': [stacking_training_time]})
training_times_df['Training Time (seconds)'] = training_times_df['Training Time (seconds)'].round(2)
training_times_df.to_csv('training_times_final_stacked.csv', index=False)
print("\nTraining time saved as 'training_times_final_stacked.csv'")
print(training_times_df)
