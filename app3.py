import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("cost_sensitive_rf_model.pkl")

st.title("Student Transfer Prediction - Full Feature App")
st.write("This version uses the 26 features that were used to train the model.")

# Input UI for the 26 training features
input_data = {
    'AgeGroup': st.selectbox("Age Group", ['<18', '18-22', '23-27', '28-32', '33+']),
    'Gender': st.radio("Gender", ['Male', 'Female']),
    'StateOfOrigin': st.selectbox("State of Origin", ['Oyo', 'Lagos', 'Osun', 'Ogun', 'Kogi ']),
    'CurrentLevel': st.selectbox("Current Level", [100, 200, 300, 400, 'Graduate']),
    'EnrollmentYear': st.selectbox("Enrollment Year", [2019, 2021, 2022, 2023, 2024]),
    'Faculty': st.selectbox("Faculty", [
        'COLLEGE OF BASIC MEDICAL AND HEALTH SCIENCES',
        'College of Natural and applied sciences ',
        'Law', 'Unknown', 'Natural and Applied Sciences ']),
    'Department': st.text_input("Department", "Biochemistry"),
    'AdmissionMode': st.selectbox("Admission Mode", ['UTME', 'Direct Entry', 'Conversion ']),
    'ChangedCourse': st.radio("Changed Course?", ['Yes', 'No', 'Unknown']),
    'ReasonForChoosingUniv': st.text_input("Reason for Choosing University", "Quality education"),
    'OnScholarship': st.radio("On Scholarship?", ['Yes', 'No', 'Unknown']),
    'TuitionFundingSource': st.selectbox("Tuition Funding Source", ['Parents', 'Scholarship', 'Self', 'Loan', 'Other']),
    'TransferReason': st.text_input("Reason for Previous Transfer (if any)", "None"),
    'CGPARange': st.selectbox("CGPA Range", ['2.4 - 3.49', '3.5 - 4.49', '4.5 - 5.0', 'Unknown']),
    'AcademicResourcesAdequate': st.radio("Are Academic Resources Adequate?", ['Yes', 'No', 'Unknown']),
    'StudentSupportRating': st.selectbox("Student Support Rating", ['Excellent', 'Good', 'Fair', 'Poor']),
    'AcademicChallenges': st.selectbox("Academic Challenges", [
        'None', 'Poor Facilities', 'Financial', 'Lecturer Availability', 'Unknown']),
    'FirstHeardAboutFU': st.selectbox("How did you first hear about FU?", ['Friend', 'Family', 'Social Media', 'School Visit', 'Other']),
    'AdmissionFinalFactor': st.text_input("Final Factor That Influenced Admission", "Proximity to home"),
    'DeferredOrBreak': st.radio("Taken a Break or Deferred?", ['Yes', 'No', 'Unknown']),
    'ReasonForDeferral': st.text_input("Reason for Deferral (if any)", "None"),
    'PlanPostgradAtFU': st.radio("Plan to do Postgraduate at FU?", ['Yes', 'No', 'Maybe']),
    'ClassSizeAdequate': st.radio("Is Class Size Adequate?", ['Yes', 'No', 'Unknown']),
    'FacultyDistributionAdequate': st.radio("Is Faculty Distribution Adequate?", ['Yes', 'No', 'Unknown']),
    'AccommodationDifficulty': st.selectbox("Accommodation Difficulty", ['None', 'Mild', 'Moderate', 'Severe']),
    'PracticalSessionsAdequate': st.radio("Are Practical Sessions Adequate?", ['Yes', 'No', 'Unknown'])
}

# Construct DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical values
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
