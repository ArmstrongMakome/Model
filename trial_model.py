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

# === 3. BALANCE DATA (CRISP-DM: Data Preparation) ===
smote_enn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smote_enn.fit_resample(X_df.values, y_encoded)
print("\nClass distribution after SMOTEENN balancing:")
print(pd.Series(y_balanced).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, 
                                                    stratify=y_balanced, random_state=42)
print(f"\nShape of X_train: {X_train.shape}, X_test: {X_test.shape}")

# === 4. DEFINE AND EVALUATE INDIVIDUAL MODELS (CRISP-DM: Modeling & Evaluation) ===
print("\n=== Individual Model Evaluation ===")

# Define the individual models (same as in the stacking classifier)
class_weights = {0: 5.0, 1: 1.0, 2: 1.0}
rf_model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=2, max_features='log2', 
                                  class_weight=class_weights, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, 
                          n_estimators=1000, max_depth=10, learning_rate=0.1, 
                          subsample=0.9, colsample_bytree=1.0, reg_lambda=1, reg_alpha=1)
lgbm_model = LGBMClassifier(class_weight=class_weights, random_state=42)
catboost_model = CatBoostClassifier(verbose=0, early_stopping_rounds=10, auto_class_weights='Balanced')

# List of models to evaluate
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model,
    'CatBoost': catboost_model
}

# Dictionary to store performance metrics, confusion matrices, and training times
performance_results = []
confusion_matrices = {}
training_times = {}

# Evaluate each model independently
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    start_time = time.time()  # Record start time for this model
    model.fit(X_train, y_train)
    end_time = time.time()  # Record end time
    training_time = end_time - start_time
    training_times[model_name] = training_time
    print(f"Training time for {model_name}: {training_time:.2f} seconds")
    
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = cm
    
    # Generate and save heatmap for confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, 
                annot_kws={"size": 12}, cbar_kws={'label': 'Number of Instances'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap for {model_name} as 'confusion_matrix_{model_name.lower().replace(' ', '_')}.png'")
    
    # Compute and print detailed classification report
    print(f"\nDetailed Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Extract only overall weighted average metrics for combined table
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    metrics = {
        'Model': model_name,
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score'],
        'Training Time (seconds)': training_time
    }
    performance_results.append(metrics)

# === 5. TRAIN AND EVALUATE STACKING CLASSIFIER (CRISP-DM: Modeling & Evaluation) ===
print("\n=== Stacking Classifier Modeling & Evaluation ===")
stacking_start_time = time.time()  # Record start time for stacking classifier
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

stacking.fit(X_train, y_train)
stacking_end_time = time.time()  # Record end time for stacking classifier
stacking_training_time = stacking_end_time - stacking_start_time
training_times['Stacking Classifier'] = stacking_training_time
print(f"Training time for Stacking Classifier: {stacking_training_time:.2f} seconds")
print("Stacking classifier trained successfully.")

# Evaluate Stacking Classifier
y_pred_stacking = stacking.predict(X_test)
y_pred_proba_stacking = stacking.predict_proba(X_test)

# Compute and print detailed classification report
print("\nDetailed Classification Report for Stacking Classifier:")
print(classification_report(y_test, y_pred_stacking, target_names=label_encoder.classes_))

# Extract only overall weighted average metrics for combined table
report_stacking = classification_report(y_test, y_pred_stacking, target_names=label_encoder.classes_, output_dict=True)
metrics_stacking = {
    'Model': 'Stacking Classifier',
    'Accuracy': report_stacking['accuracy'],
    'Precision': report_stacking['weighted avg']['precision'],
    'Recall': report_stacking['weighted avg']['recall'],
    'F1-Score': report_stacking['weighted avg']['f1-score'],
    'Training Time (seconds)': stacking_training_time
}
performance_results.append(metrics_stacking)

# Convert performance results to a DataFrame and save as a combined table
performance_df = pd.DataFrame(performance_results)
performance_df = performance_df.round(3)  # Round to 3 decimal places for readability
print("\nCombined Performance Evaluation Table for All Models (Including Training Times):")
print(performance_df)
performance_df.to_csv('combined_model_performance_with_times.csv', index=False)
print("Combined performance table saved as 'combined_model_performance_with_times.csv'")

# === 6. CREATE BAR GRAPH COMPARISON BASED ON ACCURACY ===
print("\n=== Generating Bar Graph Comparison Based on Accuracy ===")
models = performance_df['Model']
accuracies = performance_df['Accuracy']

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Ensure y-axis ranges from 0 to 1

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
             f'{height:.1%}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_accuracy_comparison.png', dpi=300)
plt.close()
print("Bar graph saved as 'model_accuracy_comparison.png'")

# === 7. THRESHOLD ADJUSTMENT ===
print("\n=== Threshold Adjustment ===")
thresholds = {}
y_pred_adj = np.copy(y_pred_stacking)
for cls in range(3):
    precision, recall, thresh = precision_recall_curve(y_test == cls, y_pred_proba_stacking[:, cls])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    thresholds[cls] = thresh[np.argmax(f1_scores)]

for i in range(len(y_pred_proba_stacking)):
    for cls in range(3):
        if y_pred_proba_stacking[i, cls] >= thresholds[cls]:
            y_pred_adj[i] = cls
            break

print("\nClassification Report for Stacking Classifier after Threshold Adjustment:")
print(classification_report(y_test, y_pred_adj, target_names=label_encoder.classes_))

# Confusion Matrix for Stacking Classifier (Threshold Adjusted)
cm_stacking = confusion_matrix(y_test, y_pred_adj)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Stacking Classifier - Threshold Adjusted)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_stacking_trail.png')
plt.close()
print("Saved confusion matrix heatmap for Stacking Classifier as 'confusion_matrix_stacking_trail.png'")

# === 8. SHAP INTERPRETABILITY (CRISP-DM: Evaluation) ===
print("\nComputing SHAP values for interpretability...")
xgb_model.fit(X_train, y_train)  # Retrain for SHAP
shap_explainer = shap.TreeExplainer(xgb_model)
X_test_df = pd.DataFrame(X_test, columns=X_df.columns)
shap_values = shap_explainer.shap_values(X_test_df)

# SHAP Feature Importance
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig('shap_feature_importance_trail.png')
plt.close()

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig('shap_summary_plot_trail.png')
plt.close()

# === 9. VISUALIZATIONS (CRISP-DM: Evaluation) ===
# ROC Curves
plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba_stacking[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig('roc_curves_trail_.png')
plt.close()

# === 10. SAVE MODELS (CRISP-DM: Deployment) ===
print("\n=== Deployment ===")
joblib.dump(stacking, 'final_healthcare_model_trial.pkl')
joblib.dump(preprocessor, 'preprocessor_trial.pkl')
joblib.dump(label_encoder, 'label_encoder.trial.pkl')
print("\nModels and preprocessors saved successfully.")

# Record the end time for overall computation
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"\nOverall Computation Time: {overall_time:.2f} seconds")

# Save training times to a file for thesis documentation (optional, can be removed if combined table suffices)
training_times_df = pd.DataFrame(list(training_times.items()), columns=['Model', 'Training Time (seconds)'])
training_times_df['Training Time (seconds)'] = training_times_df['Training Time (seconds)'].round(2)
training_times_df.to_csv('training_times_trial.csv', index=False)
print("\nTraining times saved as 'training_times_trial.csv'")
print(training_times_df)