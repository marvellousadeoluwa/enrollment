import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("cost_sensitive_rf_model_final.pkl")

st.title("Student Transfer Prediction App")
st.write("Predict whether a student might consider transferring to another institution.")

# Final top 12 list used.
feature_list = [
    'ReasonForChoosingUniv',
    'FacultyDepartment',
    'EnrollmentAge',
    'CurrentLevel',
    'TransferReason',
    'CGPARange',
    'Department',
    'StateOfOrigin',
    'EnrollmentYear',
    'TuitionFundingSource',
    'StudentSupportRating',
    'HadAcademicChallenges'
]


# UI for top 12 features
input_data = {
    'Name': st.text_input('What is your Name: '),
    'ReasonForChoosingUniv': st.selectbox("Reason for Choosing University",
                                           ["Quality education",
                                           "Academic Excellence",
                                           "Affordability",
                                           "Religious Affiliation",
                                           "Scholarship",
                                           "Unknown"]),
    
    'Faculty': st.selectbox("Faculty", ["Select from option",
        'COLLEGE OF BASIC MEDICAL AND HEALTH SCIENCES',
        'College of Natural and applied sciences',
        'Law',
        'Natural and Applied Sciences',
    ]),
    
    'Department': st.selectbox("Department",
                                ["Computer science",
                                "Islamic law",
                                "Mass communication",
                                "Mathematical and Computer Science",
                                "Nursing science",
                                "Unknown"]),
    
    'EnrollmentYear': st.selectbox("Enrollment Year", [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, "<=2020"]),
    
    'AgeGroup': st.selectbox("Age Group", ['<18', '18-22', '23-27', '28-32', '33+']),
    
    'CurrentLevel': st.selectbox("Current Level", [100, 200, 300, 400, 'Graduate',"None"]),
    
    'TransferReason': st.selectbox("Reason for Previous Transfer (if any)",["Academic Challenges", "Better Opportunities","Financial","Unknown", "None"]),
    
    'CGPARange': st.selectbox("CGPA Range", ['2.4 - 3.49', '3.5 - 4.49', '4.5 - 5.0', "Unknown"]),
    
    'StateOfOrigin': st.selectbox("State of Origin", ['Oyo', 'Lagos', 'Osun', 'Ogun', 'Ondo','Kogi', "Others"]),
    
    'TuitionFundingSource': st.selectbox("Tuition Funding Source", ['Parents', 'Scholarship', 'Self', 'Loan', 'Other']),
    
    'StudentSupportRating': st.selectbox("Student Support Rating", ['Excellent', 'Good', 'Fair', 'Poor']),
    
    'AcademicChallenges': st.selectbox("Academic Challenges", [
        'None', 'Poor Facilities', 'Financial', 'Lecturer Availability', 'Time management', 'Religion Palava'
    ])
}


# Age mapping
age_mapping = {'<18': 16, '18-22': 20, '23-27': 25, '28-32': 30, '33+': 33}

# Derived features
input_data['EnrollmentAge'] = (2025 - input_data['EnrollmentYear']) + age_mapping[input_data['AgeGroup']]
input_data['FacultyDepartment'] = input_data['Faculty'] + "_" + input_data['Department']
input_data['HadAcademicChallenges'] = 0 if input_data['AcademicChallenges'] == 'None' else 1

# Construct DataFrame
filtered_input_data = {key: input_data[key] for key in feature_list if key in input_data}

input_df = pd.DataFrame([filtered_input_data])

# Encode categorical values
for col in input_df.columns:
    input_df[col] = input_df[col].astype('category').cat.codes

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = f"{input_data['Name']} is ✅ Likely to Stay" if prediction[0] == 0 else f"{input_data['Name']} is ⚠️ Likely to Consider Transfer"
    st.subheader("Prediction Result")
    st.success(result)
    st.write("\n**Model Input Summary:**")
    st.dataframe(input_df)
