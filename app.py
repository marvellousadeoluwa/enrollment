import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("cost_sensitive_rf_model.pkl")

st.title("Student Transfer Prediction App")
st.write("Predict whether a student might consider transferring to another institution.")

# Input UI
age_group = st.selectbox("Age Group", ['<18', '18-22', '23-27', '28-32', '33+'])
gender = st.radio("Gender", ['Male', 'Female'])
state = st.selectbox("State of Origin", ['Oyo', 'Lagos', 'Osun', 'Ogun', 'Kogi '])
current_level = st.selectbox("Current Level", [100, 200, 300, 400, 'Graduate'])
enrollment_year = st.selectbox("Enrollment Year", [2019, 2021, 2022, 2023, 2024])
faculty = st.selectbox("Faculty", [
    'COLLEGE OF BASIC MEDICAL AND HEALTH SCIENCES',
    'College of Natural and applied sciences ',
    'Law', 'Unknown', 'Natural and Applied Sciences '])
admission_mode = st.selectbox("Admission Mode", ['UTME', 'Direct Entry', 'Conversion '])
changed_course = st.radio("Changed Course?", ['Yes', 'No', 'Unknown'])
cgpa_range = st.selectbox("CGPA Range", ['2.4 - 3.49', '3.5 - 4.49', '4.5 - 5.0', 'Unknown'])
academic_challenges = st.selectbox("Academic Challenges", [
    'None', 'Poor Facilities', 'Financial', 'Lecturer Availability', 'Unknown'])
student_support = st.selectbox("Student Support Rating", ['Excellent', 'Good', 'Fair', 'Poor'])
postgrad_plan = st.radio("Plan Postgrad at FU?", ['Yes', 'No', 'Maybe'])

# Construct input DataFrame
input_dict = {
    'AgeGroup': age_group,
    'Gender': gender,
    'StateOfOrigin': state,
    'CurrentLevel': str(current_level),
    'EnrollmentYear': enrollment_year,
    'Faculty': faculty,
    'AdmissionMode': admission_mode,
    'ChangedCourse': changed_course,
    'CGPARange': cgpa_range,
    'AcademicChallenges': academic_challenges,
    'StudentSupportRating': student_support,
    'PlanPostgradAtFU': postgrad_plan
}

input_df = pd.DataFrame([input_dict])

# Dummy encoding (assuming LabelEncoder or OneHotEncoder used)
# In production, ensure encoding matches training
# For this sample, we use label encoding simulation

# Load encoders if available or apply consistent mapping
# For simplicity, convert categorical features to category codes
for col in input_df.columns:
    input_df[col] = input_df[col].astype('category').cat.codes

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "✅ Likely to Stay" if prediction[0] == 0 else "⚠️ Likely to Consider Transfer"
    st.subheader("Prediction Result")
    st.success(result)

    st.write("\n**Model Input Summary:**")
    st.dataframe(input_df)
