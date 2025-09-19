import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import joblib
import time
import io
import os
import requests

# Download model from Hugging Face if not present
model_url = "https://huggingface.co/ArmstrongM/Healthcare/resolve/main/final_healthcare_model_1.pkl"
model_path = "final_healthcare_model_1.pkl"
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

model = joblib.load(model_path)

preprocessor = joblib.load('preprocessor_1.pkl')
label_encoder = joblib.load('label_encoder_1.pkl')

# Define feature lists (same as in the model)
numerical_features = ['Daily_Patient_Inflow', 'Emergency_Response_Time_Minutes', 'Staff_Count', 
                      'Bed_Occupancy_Rate', 'Visit_Frequency', 'Wait_Time_Minutes', 
                      'Length_of_Stay', 'Previous_Visits', 'Resource_Utilization', 'Age',
                      'Resource_Bed_Interaction', 'Wait_Length_Interaction', 'Log_Wait_Time', 
                      'Log_Resource_Utilization', 'Satisfaction_Staff_Interaction']
categorical_features = ['Treatment_Outcome', 'Equipment_Availability', 'Medicine_Stock_Level', 
                        'Comorbidities', 'Disease_Category']
ordinal_features = ['Satisfaction_Rating']

# Define feature names (replicating the training script's logic)
cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = numerical_features + list(cat_feature_names) + ordinal_features

# Dark Mode Toggle
st.sidebar.subheader("Theme Settings")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

# Apply custom CSS based on theme selection
if theme == "Dark":
    st.markdown("""
        <style>
        /* Main app background and default text color */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF !important;
        }
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }
        /* General text elements */
        p, div, span, label {
            color: #FFFFFF !important;
        }
        /* Form container */
        .stForm {
            background-color: #1E1E1E;
            color: #FFFFFF !important;
        }
        /* Text input fields */
        .stTextInput > div > div > input {
            background-color: #2E2E2E;
            color: #FFFFFF !important;
            border: 1px solid #555555;
        }
        /* Placeholder text for inputs */
        .stTextInput > div > div > input::placeholder {
            color: #AAAAAA !important;
        }
        /* Labels for text inputs */
        .stTextInput > div > div > label {
            color: #FFFFFF !important;
        }
        /* Selectbox (dropdowns) */
        .stSelectbox > div > div > select {
            background-color: #2E2E2E;
            color: #FFFFFF !important;
            border: 1px solid #555555;
        }
        /* Labels for selectbox */
        .stSelectbox > div > div > label {
            color: #FFFFFF !important;
        }
        /* Selectbox options (dropdown menu items) */
        .stSelectbox > div > div > select > option {
            background-color: #2E2E2E;
            color: #FFFFFF !important;
        }
        /* Slider */
        .stSlider > div > div > div > div {
            background-color: #2E2E2E;
            color: #FFFFFF !important;
        }
        /* Slider labels (the numbers/values shown) */
        .stSlider > div > div > div > div > div {
            color: #FFFFFF !important;
        }
        /* Labels for sliders */
        .stSlider > div > div > label {
            color: #FFFFFF !important;
        }
        /* Buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
        }
        /* DataFrame */
        .stDataFrame {
            background-color: #2E2E2E;
            color: #FFFFFF !important;
        }
        /* DataFrame text */
        .stDataFrame table {
            color: #FFFFFF !important;
        }
        /* Fallback for any other form-related text */
        .stForm * {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black;
        }
        .stTextInput > div > div > input {
            background-color: white;
            color: black;
        }
        .stSelectbox > div > div > select {
            background-color: white;
            color: black;
        }
        .stSlider > div > div > div > div {
            background-color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .stDataFrame {
            background-color: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("Healthcare Outcome Prediction Model Interface")
st.write("Enter patient details manually or upload a file to predict the healthcare outcome (Improved, Unchanged, Worsened).")

# Function to convert Matplotlib figure to Streamlit image (used for Prediction Probabilities graph)
def fig_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# Initialize session state for storing prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# File upload section
st.subheader("Upload Patient Data")
st.markdown("**Supported files**: CSV, XLS, ODS (OpenDocument Spreadsheet). Ensure the file contains columns matching the required features.")
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xls', 'ods'])

# Process uploaded file
if uploaded_file is not None:
    start_time = time.time()
    try:
        # Read file based on extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            input_data = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            input_data = pd.read_excel(uploaded_file)
        elif file_extension == 'ods':
            input_data = pd.read_excel(uploaded_file, engine='odf')
        
        # Validate required columns
        required_columns = numerical_features + categorical_features + ordinal_features
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            st.success("File uploaded successfully! Processing data...")
            
            # Interactive Data Exploration for Uploaded Files
            st.subheader("Explore Uploaded Data")
            st.dataframe(input_data)

            column_to_plot = st.selectbox("Select a column to visualize", input_data.columns)
            fig, ax = plt.subplots(figsize=(8, 4))
            input_data[column_to_plot].hist(ax=ax, bins=20, color='purple')
            ax.set_title(f"Histogram of {column_to_plot}")
            ax.set_xlabel(column_to_plot)
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            st.image(fig_to_streamlit(fig), caption=f"Distribution of {column_to_plot}")
            plt.close(fig)

            # Compute derived features
            input_data['Resource_Bed_Interaction'] = input_data['Resource_Utilization'] * (input_data['Bed_Occupancy_Rate'] / 100)
            input_data['Wait_Length_Interaction'] = input_data['Wait_Time_Minutes'] * input_data['Length_of_Stay']
            input_data['Log_Wait_Time'] = np.log1p(input_data['Wait_Time_Minutes'])
            input_data['Log_Resource_Utilization'] = np.log1p(input_data['Resource_Utilization'])
            input_data['Satisfaction_Staff_Interaction'] = input_data['Satisfaction_Rating'] * input_data['Staff_Count']
            
            # Preprocess and predict
            prediction_start_time = time.time()
            input_processed = preprocessor.transform(input_data)
            input_df = pd.DataFrame(input_processed, columns=feature_names)
            input_df.drop(['Wait_Time_Minutes', 'Resource_Utilization'], axis=1, inplace=True, errors='ignore')
            
            predictions = model.predict(input_df)
            predictions_proba = model.predict_proba(input_df)
            predicted_labels = label_encoder.inverse_transform(predictions)
            prediction_time = time.time() - prediction_start_time
            
            # Display results
            st.subheader("Prediction Results")
            for i, (label, proba) in enumerate(zip(predicted_labels, predictions_proba)):
                st.write(f"Record {i+1}: Predicted Outcome: **{label}**")
                st.write("Prediction Probabilities:")
                for class_label, prob in zip(label_encoder.classes_, proba):
                    st.write(f"{class_label}: {prob:.2f}")
                
                # Probability bar chart for first record
                if i == 0:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(label_encoder.classes_, proba, color='skyblue')
                    ax.set_title(f"Prediction Probabilities for Record {i+1}")
                    ax.set_xlabel("Outcome")
                    ax.set_ylabel("Probability")
                    plt.tight_layout()
                    st.image(fig_to_streamlit(fig), caption=f"Prediction Probabilities for Record {i+1}")
                    plt.close(fig)
            
            # Downloadable predictions
            results = pd.DataFrame({
                'Predicted_Label': predicted_labels,
                **{f'Prob_{label}': predictions_proba[:, i] for i, label in enumerate(label_encoder.classes_)}
            })
            st.download_button("Download Predictions", results.to_csv(index=False), "predictions.csv")
            
            # SHAP explanation for first record
            st.subheader("Model Explanation (SHAP) - First Record")
            shap_start_time = time.time()
            explainer = shap.TreeExplainer(model.estimators_[1])  # Use XGBoost for SHAP
            shap_values = explainer.shap_values(input_df.iloc[[0]])
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, input_df.iloc[[0]], plot_type="bar", show=False)
            plt.title("SHAP Feature Importance for First Record")
            plt.tight_layout()
            plt.savefig('shap_plot.png')
            plt.close()
            st.image('shap_plot.png', caption="SHAP Feature Importance")
            
            shap_time = time.time() - shap_start_time
            
            # Display timing information
            st.write(f"Prediction Time: {prediction_time:.2f} seconds")
            st.write(f"SHAP Computation Time: {shap_time:.2f} seconds")
            st.write(f"Total Processing Time: {time.time() - start_time:.2f} seconds")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Input form (only shown if no file is uploaded)
if uploaded_file is None:
    with st.form("patient_form"):
        st.subheader("Patient Information")
        
        # Numerical inputs
        daily_patient_inflow = st.number_input("Daily Patient Inflow", min_value=0, value=100)
        emergency_response_time = st.number_input("Emergency Response Time (Minutes)", min_value=0.0, value=15.0)
        staff_count = st.number_input("Staff Count", min_value=0, value=50)
        bed_occupancy_rate = st.number_input("Bed Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=75.0)
        visit_frequency = st.number_input("Visit Frequency", min_value=0, value=3)
        wait_time_minutes = st.number_input("Wait Time (Minutes)", min_value=0.0, value=30.0)
        length_of_stay = st.number_input("Length of Stay (Days)", min_value=0.0, value=5.0)
        previous_visits = st.number_input("Previous Visits", min_value=0, value=2)
        resource_utilization = st.number_input("Resource Utilization", min_value=0.0, value=80.0)
        age = st.number_input("Age", min_value=0, value=45)
        
        # Derived features
        resource_bed_interaction = resource_utilization * (bed_occupancy_rate / 100)
        wait_length_interaction = wait_time_minutes * length_of_stay
        log_wait_time = np.log1p(wait_time_minutes)
        log_resource_utilization = np.log1p(resource_utilization)
        
        # Categorical inputs
        treatment_outcome = st.selectbox("Treatment Outcome", ["Improved", "Unchanged", "Worsened"])
        equipment_availability = st.selectbox("Equipment Availability", ["Available", "Limited", "Unavailable"])
        medicine_stock_level = st.selectbox("Medicine Stock Level", ["Adequate", "Low"])
        comorbidities = st.selectbox("Comorbidities", ["Yes", "No"])
        disease_category = st.selectbox("Disease Category", ["Other", "Cardiovascular", "Diabetes", "Respiratory"])
        
        # Ordinal input
        satisfaction_rating = st.slider("Satisfaction Rating", min_value=1, max_value=5, value=3)
        satisfaction_staff_interaction = satisfaction_rating * staff_count
        
        # Submit button
        submitted = st.form_submit_button("Predict Outcome")

    # Process manual input and predict
    if submitted:
        start_time = time.time()
        # Create input dataframe
        input_data = pd.DataFrame({
            'Daily_Patient_Inflow': [daily_patient_inflow],
            'Emergency_Response_Time_Minutes': [emergency_response_time],
            'Staff_Count': [staff_count],
            'Bed_Occupancy_Rate': [bed_occupancy_rate / 100],
            'Visit_Frequency': [visit_frequency],
            'Wait_Time_Minutes': [wait_time_minutes],
            'Length_of_Stay': [length_of_stay],
            'Previous_Visits': [previous_visits],
            'Resource_Utilization': [resource_utilization],
            'Age': [age],
            'Resource_Bed_Interaction': [resource_bed_interaction],
            'Wait_Length_Interaction': [wait_length_interaction],
            'Log_Wait_Time': [log_wait_time],
            'Log_Resource_Utilization': [log_resource_utilization],
            'Treatment_Outcome': [treatment_outcome],
            'Equipment_Availability': [equipment_availability],
            'Medicine_Stock_Level': [medicine_stock_level],
            'Comorbidities': [comorbidities],
            'Disease_Category': [disease_category],
            'Satisfaction_Rating': [satisfaction_rating],
            'Satisfaction_Staff_Interaction': [satisfaction_staff_interaction]
        })
        
        # Preprocess and predict
        prediction_start_time = time.time()
        input_processed = preprocessor.transform(input_data)
        input_df = pd.DataFrame(input_processed, columns=feature_names)
        input_df.drop(['Wait_Time_Minutes', 'Resource_Utilization'], axis=1, inplace=True, errors='ignore')
        
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0]
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        prediction_time = time.time() - prediction_start_time
        
        # Save to prediction history
        history_entry = {
            'Predicted_Label': predicted_label,
            'Prob_Improved': prediction_proba[0],
            'Prob_Unchanged': prediction_proba[1],
            'Prob_Worsened': prediction_proba[2],
            'Wait_Time_Minutes': wait_time_minutes,
            'Staff_Count': staff_count,
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.prediction_history.append(history_entry)

        # Display prediction
        st.subheader("Prediction Result")
        st.write(f"Predicted Outcome: **{predicted_label}**")
        st.write("Prediction Probabilities:")
        for label, prob in zip(label_encoder.classes_, prediction_proba):
            st.write(f"{label}: {prob:.2f}")
        
        # Model Confidence Indicator
        confidence = max(prediction_proba) * 100
        st.write(f"Model Confidence: {confidence:.2f}%")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(label_encoder.classes_, prediction_proba, color='skyblue')
        ax.set_title("Prediction Probabilities")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Probability")
        plt.tight_layout()
        st.image(fig_to_streamlit(fig), caption="Prediction Probabilities")
        plt.close(fig)
        
        # Downloadable prediction
        results = pd.DataFrame({
            'Predicted_Label': [predicted_label],
            **{f'Prob_{label}': [prob] for label, prob in zip(label_encoder.classes_, prediction_proba)}
        })
        st.download_button("Download Prediction", results.to_csv(index=False), "prediction.csv")
        
        # Visualize the input values that led to the prediction
        st.subheader("Input Values Visualization")

        # Numerical Features (Bar Chart)
        numerical_values = {
            'Daily Patient Inflow': daily_patient_inflow,
            'Emergency Response Time (Min)': emergency_response_time,
            'Staff Count': staff_count,
            'Bed Occupancy Rate (%)': bed_occupancy_rate,
            'Visit Frequency': visit_frequency,
            'Wait Time (Min)': wait_time_minutes,
            'Length of Stay (Days)': length_of_stay,
            'Previous Visits': previous_visits,
            'Resource Utilization': resource_utilization,
            'Age': age,
            'Satisfaction Rating': satisfaction_rating
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(numerical_values.keys()), list(numerical_values.values()), color='lightgreen')
        ax.set_title("Numerical Input Features")
        ax.set_xlabel("Value")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        st.image(fig_to_streamlit(fig), caption="Numerical Features Used for Prediction")
        plt.close(fig)

        # Categorical Features (Bar Chart for simplicity)
        categorical_values = {
            'Treatment Outcome': treatment_outcome,
            'Equipment Availability': equipment_availability,
            'Medicine Stock Level': medicine_stock_level,
            'Comorbidities': comorbidities,
            'Disease Category': disease_category
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(categorical_values.keys()), [1] * len(categorical_values), color='lightcoral')
        for i, (key, value) in enumerate(categorical_values.items()):
            ax.text(0.5, i, value, ha='center', va='center', color='black', fontweight='bold')
        ax.set_title("Categorical Input Features")
        ax.set_xlabel("Category (Value Displayed)")
        ax.set_ylabel("Feature")
        ax.set_xlim(0, 1.5)
        plt.tight_layout()
        st.image(fig_to_streamlit(fig), caption="Categorical Features Used for Prediction")
        plt.close(fig)

        # Benchmarking Against Averages
        averages = {
            'Daily Patient Inflow': 80,
            'Emergency Response Time (Min)': 12.0,
            'Staff Count': 60,
            'Bed Occupancy Rate (%)': 70.0,
            'Wait Time (Min)': 20.0
        }

        st.subheader("Benchmarking Against Averages")
        comparison = {
            'Feature': [],
            'Your Value': [],
            'Average': [],
            'Difference': []
        }

        for feature, avg in averages.items():
            user_value = numerical_values[feature]
            comparison['Feature'].append(feature)
            comparison['Your Value'].append(user_value)
            comparison['Average'].append(avg)
            comparison['Difference'].append(user_value - avg)

        comparison_df = pd.DataFrame(comparison)
        st.table(comparison_df)

        # SHAP explanation
        st.subheader("Model Explanation (SHAP)")
        shap_start_time = time.time()
        explainer = shap.TreeExplainer(model.estimators_[1])
        shap_values = explainer.shap_values(input_df)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance for Prediction")
        plt.tight_layout()
        plt.savefig('shap_plot.png')
        plt.close()
        st.image('shap_plot.png', caption="SHAP Feature Importance")
        
        shap_time = time.time() - shap_start_time

        # Actionable Recommendations
        st.subheader("Actionable Recommendations")
        # Aggregate SHAP values across classes by taking the mean of absolute values
        shap_values_aggregated = np.abs(shap_values).mean(axis=2)  # Shape: (1, 24)
        shap_values_df = pd.DataFrame(shap_values_aggregated, columns=input_df.columns)
        top_negative_features = shap_values_df.iloc[0].sort_values(ascending=False).head(3).index.tolist()

        st.write("Top features impacting the outcome (based on SHAP importance):")
        for feature in top_negative_features:
            st.write(f"- {feature}: This feature has a high impact on the prediction. Consider reviewing its value.")

        # Specific suggestion for Wait_Time_Minutes if it's a top feature
        if 'Wait_Time_Minutes' in top_negative_features:
            suggested_wait_time = wait_time_minutes * 0.8  # Reduce by 20%
            st.write(f"Suggestion: Reducing Wait Time to {suggested_wait_time:.1f} minutes may improve the outcome.")

        # Interactive Feature Impact Simulator
        st.subheader("Feature Impact Simulator")
        with st.expander("Adjust Key Features"):
            new_wait_time = st.slider("Adjust Wait Time (Minutes)", min_value=0.0, max_value=60.0, value=float(wait_time_minutes))
            new_staff_count = st.slider("Adjust Staff Count", min_value=0, max_value=100, value=int(staff_count))

            if st.button("Simulate New Outcome"):
                input_data['Wait_Time_Minutes'] = [new_wait_time]
                input_data['Staff_Count'] = [new_staff_count]
                input_data['Wait_Length_Interaction'] = [new_wait_time * length_of_stay]
                input_data['Log_Wait_Time'] = [np.log1p(new_wait_time)]
                input_data['Satisfaction_Staff_Interaction'] = [satisfaction_rating * new_staff_count]

                input_processed = preprocessor.transform(input_data)
                input_df = pd.DataFrame(input_processed, columns=feature_names)
                input_df.drop(['Wait_Time_Minutes', 'Resource_Utilization'], axis=1, inplace=True, errors='ignore')

                new_prediction = model.predict(input_df)
                new_prediction_proba = model.predict_proba(input_df)[0]
                new_predicted_label = label_encoder.inverse_transform(new_prediction)[0]

                st.write(f"New Predicted Outcome: **{new_predicted_label}**")
                st.write("New Prediction Probabilities:")
                for label, prob in zip(label_encoder.classes_, new_prediction_proba):
                    st.write(f"{label}: {prob:.2f}")

        # Historical Prediction Dashboard
        st.subheader("Prediction History")
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history_df['Timestamp'], history_df['Prob_Improved'], label='Improved', marker='o')
            ax.plot(history_df['Timestamp'], history_df['Prob_Unchanged'], label='Unchanged', marker='o')
            ax.plot(history_df['Timestamp'], history_df['Prob_Worsened'], label='Worsened', marker='o')
            ax.set_title("Prediction Probabilities Over Time")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Probability")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.image(fig_to_streamlit(fig), caption="Prediction Trends")
            plt.close(fig)

        # Real-Time Feedback
        st.subheader("Feedback")
        with st.form("feedback_form"):
            rating = st.slider("Rate the prediction accuracy (1-5)", 1, 5, 3)
            comments = st.text_area("Comments (optional)")
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                st.success("Thank you for your feedback!")

        # Display timing information
        st.write(f"Prediction Time: {prediction_time:.2f} seconds")
        st.write(f"SHAP Computation Time: {shap_time:.2f} seconds")
        st.write(f"Total Processing Time: {time.time() - start_time:.2f} seconds")
