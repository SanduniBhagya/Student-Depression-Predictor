# predict_student_depression.py
("Script started successfully!")
import pickle
import pandas as pd
import numpy as np

# --- Load the saved model and scaler ---
try:
    with open('depression_prediction_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print('Model and scaler loaded successfully.')
except FileNotFoundError:
    print('Error: Model or scaler files not found. Make sure they are in the same directory as this script.')
    exit()
except Exception as e:
    print(f'An error occurred while loading files: {e}')
    exit()

# --- Define the EXACT order of features (MUST match training data) ---
feature_columns = ['Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'City_Agra', 'City_Ahmedabad', 'City_Bangalore', 'City_Bhavna', 'City_Bhopal', 'City_Chennai', 'City_City', 'City_Delhi', 'City_Faridabad', 'City_Gaurav', 'City_Ghaziabad', 'City_Harsh', 'City_Harsha', 'City_Hyderabad', 'City_Indore', 'City_Jaipur', 'City_Kalyan', 'City_Kanpur', 'City_Khaziabad', 'City_Kibara', 'City_Kolkata', 'City_Less Delhi', 'City_Less than 5 Kalyan', 'City_Lucknow', 'City_Ludhiana', 'City_M.Com', 'City_M.Tech', 'City_ME', 'City_Meerut', 'City_Mihir', 'City_Mira', 'City_Mumbai', 'City_Nagpur', 'City_Nalini', 'City_Nalyan', 'City_Nandini', 'City_Nashik', 'City_Patna', 'City_Pune', 'City_Rajkot', 'City_Rashi', 'City_Reyansh', 'City_Saanvi', 'City_Srinagar', 'City_Surat', 'City_Thane', 'City_Vaanya', 'City_Vadodara', 'City_Varanasi', 'City_Vasai-Virar', 'City_Visakhapatnam', 'Profession_Chef', 'Profession_Civil Engineer', 'Profession_Content Writer', 'Profession_Digital Marketer', 'Profession_Doctor', 'Profession_Educational Consultant', 'Profession_Entrepreneur', 'Profession_Lawyer', 'Profession_Manager', 'Profession_Pharmacist', 'Profession_Student', 'Profession_Teacher', 'Profession_UX/UI Designer', 'Sleep Duration_7-8 hours', 'Sleep Duration_Less than 5 hours', 'Sleep Duration_More than 8 hours', 'Sleep Duration_Others', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Degree_B.Com', 'Degree_B.Ed', 'Degree_B.Pharm', 'Degree_B.Tech', 'Degree_BA', 'Degree_BBA', 'Degree_BCA', 'Degree_BE', 'Degree_BHM', 'Degree_BSc', 'Degree_Class 12', 'Degree_LLB', 'Degree_LLM', 'Degree_M.Com', 'Degree_M.Ed', 'Degree_M.Pharm', 'Degree_M.Tech', 'Degree_MA', 'Degree_MBA', 'Degree_MBBS', 'Degree_MCA', 'Degree_MD', 'Degree_ME', 'Degree_MHM', 'Degree_MSc', 'Degree_Others', 'Degree_PhD']

# --- Define numerical columns that need scaling (MUST match preprocessing) ---
numerical_cols_to_scale = [
    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
    'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress'
]

# --- Function to preprocess new raw input and make a prediction ---
def predict_new_student_depression(raw_input_data):
    # 1. Create a DataFrame for the new input, ensuring all columns are present and initialized to 0
    new_data_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # 2. Populate the DataFrame with the raw input data
    # Handle binary mapped columns (Gender, Suicidal Thoughts, Family History)
    new_data_df['Gender'] = 1 if raw_input_data.get('Gender', 'Male').lower() == 'male' else 0
    new_data_df['Have you ever had suicidal thoughts ?'] = 1 if raw_input_data.get('Have you ever had suicidal thoughts ?', 'No').lower() == 'yes' else 0
    new_data_df['Family History of Mental Illness'] = 1 if raw_input_data.get('Family History of Mental Illness', 'No').lower() == 'yes' else 0

    # Populate numerical columns
    for col in numerical_cols_to_scale:
        if col in raw_input_data and raw_input_data[col] is not None:
            new_data_df[col] = raw_input_data[col]

    # Populate one-hot encoded columns (set to 1 for the chosen category)
    city_val = raw_input_data.get('City')
    if city_val:
        col_name = f'City_{city_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1
        else:
            print(f"Warning: City '{city_val}' not found in training data. This feature will be ignored.")

    profession_val = raw_input_data.get('Profession')
    if profession_val:
        col_name = f'Profession_{profession_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1
        else:
            print(f"Warning: Profession '{profession_val}' not found in training data. This feature will be ignored.")

    sleep_duration_val = raw_input_data.get('Sleep Duration')
    if sleep_duration_val:
        col_name = f'Sleep Duration_{sleep_duration_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1
        else:
            print(f"Warning: Sleep Duration '{sleep_duration_val}' not found in training data. This feature will be ignored.")

    dietary_habits_val = raw_input_data.get('Dietary Habits')
    if dietary_habits_val:
        col_name = f'Dietary Habits_{dietary_habits_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1
        else:
            print(f"Warning: Dietary Habits '{dietary_habits_val}' not found in training data. This feature will be ignored.")

    degree_val = raw_input_data.get('Degree')
    if degree_val:
        col_name = f'Degree_{degree_val}'
        if col_name in new_data_df.columns:
            new_data_df[col_name] = 1
        else:
            print(f"Warning: Degree '{degree_val}' not found in training data. This feature will be ignored.")

    # 3. Scale numerical features using the LOADED scaler
    new_data_df[numerical_cols_to_scale] = scaler.transform(new_data_df[numerical_cols_to_scale])

    # 4. Make prediction
    prediction = model.predict(new_data_df)[0]
    prediction_proba = model.predict_proba(new_data_df)[:, 1][0]

    return prediction, prediction_proba

if __name__ == '__main__':
    print('\n--- Enter Student Information for Depression Prediction ---')
    print('Please provide the following details:')

    input_data = {}

    # Gender (Binary)
    while True:
        gender = input("Gender (Male/Female): ").strip()
        if gender.lower() in ['male', 'female']:
            input_data['Gender'] = gender
            break
        else:
            print("Invalid input. Please enter 'Male' or 'Female'.")

    # Age (Numerical)
    while True:
        try:
            age = int(input("Age: "))
            input_data['Age'] = age
            break
        except ValueError:
            print("Invalid input. Please enter a number for age.")

    # Academic Pressure (Numerical)
    while True:
        try:
            ap = float(input("Academic Pressure (0.0 - 5.0, e.g., 3.5): "))
            if 0.0 <= ap <= 5.0:
                input_data['Academic Pressure'] = ap
                break
            else:
                print("Invalid input. Please enter a value between 0.0 and 5.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Work Pressure (Numerical)
    while True:
        try:
            wp = float(input("Work Pressure (0.0 - 5.0, e.g., 2.0): "))
            if 0.0 <= wp <= 5.0:
                input_data['Work Pressure'] = wp
                break
            else:
                print("Invalid input. Please enter a value between 0.0 and 5.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # CGPA (Numerical)
    while True:
        try:
            cgpa = float(input("CGPA (0.0 - 10.0, e.g., 7.8): "))
            if 0.0 <= cgpa <= 10.0:
                input_data['CGPA'] = cgpa
                break
            else:
                print("Invalid input. Please enter a value between 0.0 and 10.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Study Satisfaction (Numerical)
    while True:
        try:
            ss = float(input("Study Satisfaction (0.0 - 5.0, e.g., 4.0): "))
            if 0.0 <= ss <= 5.0:
                input_data['Study Satisfaction'] = ss
                break
            else:
                print("Invalid input. Please enter a value between 0.0 and 5.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Job Satisfaction (Numerical, handle optional for students)
    while True:
        try:
            js_input = input("Job Satisfaction (0.0 - 5.0, or leave blank if N/A): ").strip()
            if js_input == '':
                input_data['Job Satisfaction'] = 0.0 # Or median/mean from training data if you prefer
                break
            js = float(js_input)
            if 0.0 <= js <= 5.0:
                input_data['Job Satisfaction'] = js
                break
            else:
                print("Invalid input. Please enter a value between 0.0 and 5.0, or leave blank.")
        except ValueError:
            print("Invalid input. Please enter a number or leave blank.")

    # Work/Study Hours (Numerical)
    while True:
        try:
            wsh = float(input("Work/Study Hours (e.g., 8.0): "))
            if wsh >= 0:
                input_data['Work/Study Hours'] = wsh
                break
            else:
                print("Invalid input. Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Financial Stress (Numerical)
    while True:
        try:
            fs = float(input("Financial Stress (0.0 - 5.0, e.g., 4.5): "))
            if 0.0 <= fs <= 5.0:
                input_data['Financial Stress'] = fs
                break
            else:
                print("Invalid input. Please enter a value between 0.0 and 5.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Suicidal Thoughts (Binary)
    while True:
        st = input("Have you ever had suicidal thoughts? (Yes/No): ").strip()
        if st.lower() in ['yes', 'no']:
            input_data['Have you ever had suicidal thoughts ?'] = st
            break
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")

    # Family History of Mental Illness (Binary)
    while True:
        fh = input("Family History of Mental Illness? (Yes/No): ").strip()
        if fh.lower() in ['yes', 'no']:
            input_data['Family History of Mental Illness'] = fh
            break
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")


    # Example for City
    valid_cities = [col.replace('Kalyan','Srinagar','Hyderabad','Vasai-Virar','Lucknow','Thane','Ludhiana','Agra','Surat','Kolkata','Jaipur','Patna','Visakhapatnam','Pune','Ahmedabad','Bhopal','Chennai','Meerut','Rajkot','Delhi','Bangalore','Ghaziabad','Mumbai','Vadodara','Varanasi','Nagpur','Indore','Kanpur','Nashik','Faridabad','Saanvi','Bhavna','City','Harsha','Kibara','Nandini','Nalini','Mihir','Nalyan','M.Com','ME','Rashi','Gaurav','Reyansh','Harsh','Vaanya','Mira','Less than 5 Kalyan','3.0','Less Delhi','M.Tech','Khaziabad'
) for col in feature_columns if col.startswith('City_')]
    print(f"\nValid Cities: {', '.join(valid_cities)}")
    while True:
        city = input("City: ").strip()
        if f'City_{city}' in feature_columns:
            input_data['City'] = city
            break
        else:
            print(f"Invalid city. Please choose from: {', '.join(valid_cities)}")

    # Example for Profession
    valid_professions = [col.replace('Student', 'Architect', 'Teacher', 'Digital Marketer', 'Content Writer', 'Chef', 'Doctor', 'Pharmacist', 'Civil Engineer', 'UX/UI Designer', 'Educational Consultant', 'Manager', 'Lawyer', 'Entrepreneur') for col in feature_columns if col.startswith('Profession_')]
    print(f"\nValid Professions: {', '.join(valid_professions)}")
    while True:
        profession = input("Profession: ").strip()
        if f'Profession_{profession}' in feature_columns:
            input_data['Profession'] = profession
            break
        else:
            print(f"Invalid profession. Please choose from: {', '.join(valid_professions)}")

    # Example for Sleep Duration
    valid_sleep_durations = [col.replace('Less than 5 hours', '7-8 hours', '5-6 hours', 'More than 8 hours', 'Others') for col in feature_columns if col.startswith('Sleep Duration_')]
    print(f"\nValid Sleep Durations: {', '.join(valid_sleep_durations)}")
    while True:
        sd = input("Sleep Duration: ").strip()
        if f'Sleep Duration_{sd}' in feature_columns:
            input_data['Sleep Duration'] = sd
            break
        else:
            print(f"Invalid sleep duration. Please choose from: {', '.join(valid_sleep_durations)}")

    # Example for Dietary Habits
    valid_dietary_habits = [col.replace('Unhealthy', 'Moderate', 'Healthy', 'Others') for col in feature_columns if col.startswith('Dietary Habits_')]
    print(f"\nValid Dietary Habits: {', '.join(valid_dietary_habits)}")
    while True:
        dh = input("Dietary Habits (Healthy/Moderate/Unhealthy): ").strip()
        if f'Dietary Habits_{dh}' in feature_columns:
            input_data['Dietary Habits'] = dh
            break
        else:
            print(f"Invalid dietary habit. Please choose from: {', '.join(valid_dietary_habits)}")

    valid_degrees = [col.replace('Class 12', 'B.Ed', 'B.Com', 'B.Arch', 'BCA', 'MSc', 'B.Tech', 'MCA', 'M.Tech', 'BHM', 'BSc', 'M.Ed', 'B.Pharm', 'M.Com', 'MBBS', 'BBA', 'LLB', 'BE', 'BA', 'M.Pharm', 'MD', 'MBA', 'MA', 'PhD', 'LLM', 'MHM', 'ME', 'Others') for col in feature_columns if col.startswith('Degree_')]
    if 'M.C.A' in feature_columns:
        valid_degrees.append('M.C.A')
    
    print(f"\nValid Degrees: {', '.join(valid_degrees)}")
    while True:
        degree = input("Degree: ").strip()
        if f'Degree_{degree}' in feature_columns or degree == 'M.C.A': # Handle 'M.C.A' directly
            input_data['Degree'] = degree
            break
        else:
            print(f"Invalid degree. Please choose from: {', '.join(valid_degrees)}")


    print("\n--- Processing your input ---")
    predicted_label, probability_of_depression = predict_new_student_depression(input_data)

    print(f'\nPrediction: {"Depressed" if predicted_label == 1 else "Not Depressed"}')
    print(f'Probability of Depression: {probability_of_depression:.4f}')