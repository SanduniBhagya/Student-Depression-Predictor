# app_streamlit.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🦋",
    layout="wide", 
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    .stApp {
        /* REPLACED background-color and background-image with a subtle linear gradient */
        background: linear-gradient(to right,#F0E8F0, #D8BFD8, #E0B0FF);
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
        background-attachment: fixed;
    }
    /* Increased padding-top for the main content block to prevent title cutoff */
    .main .block-container {
        padding-top: 4rem; /* Increased from 3rem */
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* NEW: Set general text color for better readability */
    .stApp,
    .stApp label,
    .stApp p,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #262626; /* Very dark gray for all text, almost black */
    }

    /* Style for the main form container - enhanced shade */
    div[data-testid="stForm"] {
        background-color: #ffffff; /* White background for the form box */
        padding: 2.5rem; /* More padding inside the form box */
        border-radius: 0.75rem; /* Rounded corners for the box */
        /* ENHANCED SHADOW: Larger and slightly darker for more prominence */
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.08);
        margin-top: 2rem; /* Space above the form box */
        margin-bottom: 2rem; /* Space below the form box */
        border: 1px solid #e0e0e0; /* Subtle border for definition */
    }

    .stButton>button {
        background-color: #2563eb; /* Dark blue background */
        color: #FFFFFF !important; /* Force button text color to white */
        font-weight: bold;
        /* EXTREMELY ENHANCED BUTTON SIZE AND SHADOW FOR MAXIMUM PROMINENCE */
        padding: 2rem 5rem !important; /* Made significantly larger with !important */
        font-size: 2rem !important; /* Made significantly larger text with !important */
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.5), 0 8px 12px -4px rgba(0, 0, 0, 0.3); /* Even more prominent shadow */
        /* END EXTREMELY ENHANCED BUTTON */
        
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        
        /* Styles to center the button */
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 2rem; /* Add some space above the button */
    }
    .stButton>button:hover {
        background-color: #1d4ed8; /* Darker blue on hover */
        transform: translateY(-4px); /* Even more lift on hover */
        box-shadow: 0 25px 40px -12px rgba(0, 0, 0, 0.6), 0 10px 15px -5px rgba(0, 0, 0, 0.4); /* Very aggressive hover shadow */
    }
    .prediction-box {
        background-color: #e0f2f7; /* Light cyan for result */
        border-left: 8px solid #0288d1; /* Darker blue border */
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        text-align: center;
    }
    .prediction-text {
        font-size: 1.875rem; /* text-3xl */
        font-weight: bold;
        color: #0369a1; /* Darker blue for prediction result, keep this distinct */
        margin-bottom: 0.5rem;
    }
    .probability-text {
        font-size: 1.25rem; /* text-xl */
        color: #4a5568; /* Gray 700 for probability text */
    }
    /* Adjust input field border-radius for consistency */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input { /* Also target number input */
        border-radius: 0.5rem;
        color: #262626; /* Ensure input text itself is dark */
    }
    /* Target the selected option text in a selectbox */
    .stSelectbox>div>div>div[data-baseweb="select"] > div:first-child {
        color: #262626;
    }
    /* Specific styling for radio buttons to make them clearer */
    .stRadio > label {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# The rest of your Python code remains the same.

st.title("🦋 Student Depression Predictor")
st.write("Enter student information below to get a depression risk prediction.")

# --- Load the saved model and scaler ---
# Ensure these files are in the same directory as app_streamlit.py
@st.cache_resource # Cache the model loading for efficiency
def load_model_and_scaler():
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, 'depression_prediction_model.pkl')
        scaler_path = os.path.join(script_dir, 'scaler.pkl')

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        st.success("Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        st.error(
            "Error: Model or scaler files not found. "
            "Please ensure 'depression_prediction_model.pkl' and 'scaler.pkl' "
            "are in the same directory as this script.", icon="❌"
        )
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}", icon="❗")
        st.stop() # Stop the app
model, scaler = load_model_and_scaler()


# --- Define the EXACT order of features (MUST match training data) ---
# REPLACE THIS LIST with the output from X.columns.tolist() from your Jupyter Notebook
feature_columns =['Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'City_Agra', 'City_Ahmedabad', 'City_Bangalore', 'City_Bhavna', 'City_Bhopal', 'City_Chennai', 'City_City', 'City_Delhi', 'City_Faridabad', 'City_Gaurav', 'City_Ghaziabad', 'City_Harsh', 'City_Harsha', 'City_Hyderabad', 'City_Indore', 'City_Jaipur', 'City_Kalyan', 'City_Kanpur', 'City_Khaziabad', 'City_Kibara', 'City_Kolkata', 'City_Less Delhi', 'City_Less than 5 Kalyan', 'City_Lucknow', 'City_Ludhiana', 'City_M.Com', 'City_M.Tech', 'City_ME', 'City_Meerut', 'City_Mihir', 'City_Mira', 'City_Mumbai', 'City_Nagpur', 'City_Nalini', 'City_Nalyan', 'City_Nandini', 'City_Nashik', 'City_Patna', 'City_Pune', 'City_Rajkot', 'City_Rashi', 'City_Reyansh', 'City_Saanvi', 'City_Srinagar', 'City_Surat', 'City_Thane', 'City_Vaanya', 'City_Vadodara', 'City_Varanasi', 'City_Vasai-Virar', 'City_Visakhapatnam', 'Profession_Chef', 'Profession_Civil Engineer', 'Profession_Content Writer', 'Profession_Digital Marketer', 'Profession_Doctor', 'Profession_Educational Consultant', 'Profession_Entrepreneur', 'Profession_Lawyer', 'Profession_Manager', 'Profession_Pharmacist', 'Profession_Student', 'Profession_Teacher', 'Profession_UX/UI Designer', 'Sleep Duration_7-8 hours', 'Sleep Duration_Less than 5 hours', 'Sleep Duration_More than 8 hours', 'Sleep Duration_Others', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Degree_B.Com', 'Degree_B.Ed', 'Degree_B.Pharm', 'Degree_B.Tech', 'Degree_BA', 'Degree_BBA', 'Degree_BCA', 'Degree_BE', 'Degree_BHM', 'Degree_BSc', 'Degree_Class 12', 'Degree_LLB', 'Degree_LLM', 'Degree_M.Com', 'Degree_M.Ed', 'Degree_M.Pharm', 'Degree_M.Tech', 'Degree_MA', 'Degree_MBA', 'Degree_MBBS', 'Degree_MCA', 'Degree_MD', 'Degree_ME', 'Degree_MHM', 'Degree_MSc', 'Degree_Others', 'Degree_PhD']

# --- Define numerical columns that need scaling (MUST match preprocessing) ---
numerical_cols_to_scale = [
    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
    'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress'
]

# --- Define valid options for categorical inputs ---
valid_genders = ['Male', 'Female']
valid_cities = ["--- Select an Option ---"] + sorted(['Kalyan', 'Srinagar', 'Hyderabad', 'Vasai-Virar', 'Lucknow', 'Thane', 'Ludhiana', 'Agra', 'Surat', 'Kolkata', 'Jaipur', 'Patna', 'Visakhapatnam', 'Pune', 'Ahmedabad', 'Bhopal', 'Chennai', 'Meerut', 'Rajkot', 'Delhi', 'Bangalore', 'Ghaziabad', 'Mumbai', 'Vadodara', 'Varanasi', 'Nagpur', 'Indore', 'Kanpur', 'Nashik', 'Faridabad', 'Saanvi', 'Bhavna', 'City', 'Harsha', 'Kibara', 'Nandini', 'Nalini', 'Mihir', 'Nalyan', 'M.Com', 'ME', 'Rashi', 'Gaurav', 'Reyansh', 'Harsh', 'Vaanya', 'Mira', 'Less than 5 Kalyan', '3.0', 'Less Delhi', 'M.Tech', 'Khaziabad'])
valid_professions = ["--- Select an Option ---"] + sorted(['Student', 'Architect', 'Teacher', 'Digital Marketer', 'Content Writer', 'Chef', 'Doctor', 'Pharmacist', 'Civil Engineer', 'UX/UI Designer', 'Educational Consultant', 'Manager', 'Lawyer', 'Entrepreneur'])
valid_sleep_durations = ["--- Select an Option ---"] + sorted(['Less than 5 hours', '7-8 hours', '5-6 hours', 'More than 8 hours', 'Others'])
valid_dietary_habits = ["--- Select an Option ---"] + sorted(['Unhealthy', 'Moderate', 'Healthy', 'Others'])
valid_degrees = ["--- Select an Option ---"] + sorted(['Class 12', 'B.Ed', 'B.Com', 'B.Arch', 'BCA', 'MSc', 'B.Tech', 'MCA', 'M.Tech', 'BHM', 'BSc', 'M.Ed', 'B.Pharm', 'M.Com', 'MBBS', 'BBA', 'LLB', 'BE', 'BA', 'M.Pharm', 'MD', 'MBA', 'MA', 'PhD', 'LLM', 'MHM', 'ME', 'Others'])
valid_suicidal_thoughts_options = ['Yes', 'No']
valid_family_history_options = ['Yes', 'No']

# --- Function to preprocess new raw input and make a prediction ---
def predict_new_student_depression(raw_input_data):
    # 1. Create a DataFrame for the new input, ensuring all columns are present and initialized to 0
    new_data_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # 2. Populate the DataFrame with the raw input data
    # Handle binary mapped columns (Gender, Suicidal Thoughts, Family History)
    new_data_df['Gender'] = 1 if raw_input_data.get('gender', 'Male').lower() == 'male' else 0
    new_data_df['Have you ever had suicidal thoughts ?'] = 1 if raw_input_data.get('suicidal_thoughts', 'No').lower() == 'yes' else 0
    new_data_df['Family History of Mental Illness'] = 1 if raw_input_data.get('family_history', 'No').lower() == 'yes' else 0

    # Populate numerical columns
    for col in numerical_cols_to_scale:
        val = raw_input_data.get(col)
        if val is not None:
            new_data_df[col] = float(val)

    # Populate one-hot encoded columns (set to 1 for the chosen category)
    # Check for placeholder value and ignore it
    city_val = raw_input_data.get('City')
    if city_val and city_val != "--- Select an Option ---":
        col_name = f'City_{city_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1

    profession_val = raw_input_data.get('Profession')
    if profession_val and profession_val != "--- Select an Option ---":
        col_name = f'Profession_{profession_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1

    sleep_duration_val = raw_input_data.get('Sleep Duration')
    if sleep_duration_val and sleep_duration_val != "--- Select an Option ---":
        col_name = f'Sleep Duration_{sleep_duration_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1

    dietary_habits_val = raw_input_data.get('Dietary Habits')
    if dietary_habits_val and dietary_habits_val != "--- Select an Option ---":
        col_name = f'Dietary Habits_{dietary_habits_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1

    degree_val = raw_input_data.get('Degree')
    if degree_val and degree_val != "--- Select an Option ---":
        col_name = f'Degree_{degree_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1
        elif degree_val == 'M.C.A' and 'M.C.A' in new_data_df.columns: # Handle M.C.A if it's a direct column
             new_data_df['M.C.A'] = 1


    # 3. Scale numerical features using the LOADED scaler
    new_data_df[numerical_cols_to_scale] = scaler.transform(new_data_df[numerical_cols_to_scale])

    # 4. Make prediction
    prediction = model.predict(new_data_df)[0]
    prediction_proba = model.predict_proba(new_data_df)[:, 1][0]

    return prediction, prediction_proba

# --- Streamlit Form ---
with st.form("depression_prediction_form"):
    st.header("Student Information")
    st.write("Please provide the following details about the student for a depression risk prediction:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("1. Select Gender:", valid_genders, help="Choose the student's gender.")
        age = st.number_input("2. Enter Age:", min_value=18, max_value=60, value=18, help="Student's age in years.")
        city = st.selectbox("3. Select City:", valid_cities, index=0, help="The city where the student resides.")
        profession = st.selectbox("4. Select Profession:", valid_professions, index=0, help="The student's primary profession or field of study.")
        academic_pressure = st.slider("5. Academic Pressure (0.0 - 5.0):", 0.0, 5.0, 3.0, 0.1, help="Rate academic pressure from 0 (very low) to 5 (very high).")
        work_pressure = st.slider("6. Work Pressure (0.0 - 5.0):", 0.0, 5.0, 1.0, 0.1, help="Rate work pressure from 0 (very low/none) to 5 (very high).")
        cgpa = st.slider("7. CGPA (0.0 - 10.0):", 0.0, 10.0, 7.5, 0.01, help="Current Cumulative Grade Point Average (CGPA).")

    with col2:
        study_satisfaction = st.slider("8. Study Satisfaction (0.0 - 5.0):", 0.0, 5.0, 3.0, 0.1, help="Rate satisfaction with studies from 0 (very dissatisfied) to 5 (very satisfied).")
        job_satisfaction = st.slider("9. Job Satisfaction (0.0 - 5.0, 0 if N/A):", 0.0, 5.0, 0.0, 0.1, help="Rate satisfaction with current job (if any) from 0 (N/A or very dissatisfied) to 5 (very satisfied).")
        sleep_duration = st.selectbox("10. Select Sleep Duration:", valid_sleep_durations, index=0, help="Average daily hours of sleep.")
        dietary_habits = st.selectbox("11. Select Dietary Habits:", valid_dietary_habits, index=0, help="General dietary patterns (e.g., Healthy, Moderate, Unhealthy).")
        degree = st.selectbox("12. Select Degree Program:", valid_degrees, index=0, help="The student's current or highest degree program.")
        suicidal_thoughts = st.radio("13. Ever had suicidal thoughts?", valid_suicidal_thoughts_options, index=1, help="Has the student ever experienced suicidal thoughts?")
        work_study_hours = st.number_input("14. Daily Work/Study Hours:", min_value=0.0, value=8.0, step=0.5, help="Average total hours spent on work and study daily.")
        financial_stress = st.slider("15. Financial Stress (0.0 - 5.0):", 0.0, 5.0, 3.0, 0.1, help="Rate financial stress from 0 (no stress) to 5 (very high stress).")
        family_history = st.radio("16. Family History of Mental Illness?", valid_family_history_options, index=1, help="Is there a family history of mental illness?")

    submitted = st.form_submit_button("Predict Depression")

    if submitted:
        # Basic validation for select boxes that are required
        if any(val == "--- Select an Option ---" for val in [city, profession, sleep_duration, dietary_habits, degree]):
            st.error("❌ Please ensure all required fields (marked with '--- Select an Option ---') are filled.", icon="❗")
            st.stop() # Stop execution if validation fails

        input_data = {
            'gender': gender,
            'Age': age,
            'City': city,
            'Profession': profession,
            'Academic Pressure': academic_pressure,
            'Work Pressure': work_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction,
            'Job Satisfaction': job_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Degree': degree,
            'suicidal_thoughts': suicidal_thoughts,
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress,
            'family_history': family_history
        }

        # Make prediction
        predicted_label, probability_of_depression = predict_new_student_depression(input_data)

        prediction_result = "Depressed" if predicted_label == 1 else "Not Depressed"
        
        st.markdown(
            f"""
            <div class="prediction-box">
                <p class="prediction-text">Prediction: {prediction_result}</p>
                <p class="probability-text">Probability of Depression: {probability_of_depression:.2f}</p>
            </div>
            """, unsafe_allow_html=True
        )
        if predicted_label == 1:
            st.warning("Recommendation: Please consider seeking professional help or support.",icon="🫂")
        else:
            st.success("Based on the provided data, you are not depressed.", icon="💜")