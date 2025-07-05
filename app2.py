import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("cost_sensitive_rf_model_final.pkl")

st.title("Student Transfer Prediction App")
st.write("Predict whether a student might consider transferring to another institution.")

# Full list of 26 model features used during training

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
    'StateOfOrigin': st.selectbox("State of Origin", ['Oyo', 'Lagos', 'Osun', 'Ogun', 'Kogi ']),
    'CurrentLevel': st.selectbox("Current Level", [100, 200, 300, 400, 'Graduate']),
    'EnrollmentYear': st.selectbox("Enrollment Year", [2019, 2021, 2022, 2023, 2024]),
    'Department': st.text_input("Department", "Biochemistry"),
    'ReasonForChoosingUniv': st.text_input("Reason for Choosing University", "Quality education"),
    'TuitionFundingSource': st.selectbox("Tuition Funding Source", ['Parents', 'Scholarship', 'Self', 'Loan', 'Other']),
    'TransferReason': st.text_input("Reason for Previous Transfer (if any)", "None"),
    'CGPARange': st.selectbox("CGPA Range", ['2.4 - 3.49', '3.5 - 4.49', '4.5 - 5.0', 'Unknown']),
    'StudentSupportRating': st.selectbox("Student Support Rating", ['Excellent', 'Good', 'Fair', 'Poor']),
    'AcademicChallenges': st.selectbox("Academic Challenges", [
        'None', 'Poor Facilities', 'Financial', 'Lecturer Availability', 'Unknown'
    ])
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
