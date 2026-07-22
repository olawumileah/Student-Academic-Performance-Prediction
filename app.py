import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Academic Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Academic Performance Prediction System")

st.write("""
Predict whether a student is likely to **Pass**
or become **At-Risk** using a trained
Decision Tree Machine Learning model.
""")

@st.cache_resource
def load_model():
    model = joblib.load("models/decision_tree_model.pkl")
    return model

@st.cache_data
def load_dataset():
    return pd.read_excel("data/processed/cleaned_dataset.xlsx")

model = load_model()
df = load_dataset()

faculty_encoder = joblib.load("models/faculty_encoder.pkl")
department_encoder = joblib.load("models/department_encoder.pkl")

gender_map = {
    "Female": 0,
    "Male": 1
}

level_map = {
    "100 Level": 1,
    "200 Level": 2,
    "300 Level": 3,
    "400 Level": 4,
    "500 Level": 5
}

cgpa_map = {
    "Below 1.50": 1,
    "1.50 - 2.49": 2,
    "2.50 - 3.49": 3,
    "3.50 - 4.49": 4,
    "4.50 - 5.00": 5
}

attendance_map = {
    "Below 50%": 1,
    "50% - 59%": 2,
    "60% - 69%": 3,
    "70% - 79%": 4,
    "80% - 89%": 5,
    "90% - 100%": 6
}

passed_map = {"No": 0, "Yes": 1}

performance_map = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5
}

device_map = {
    "Smartphone": 0,
    "Laptop": 1,
    "Desktop Computer": 2,
    "Tablet": 3
}

online_resource_map = {
    "No": 0,
    "Yes": 1
}

resource_frequency_map = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Very Often": 5,
    "Always": 6
}

likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

assignment_map = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

discussion_map = {
    "No": 0,
    "Yes": 1
}

time_map = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5
}
st.sidebar.title("Prediction Menu")
st.sidebar.success("Enter Student Information")
st.sidebar.success("✅ Decision Tree Model Loaded")

# Form starts here
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].unique())
        )

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=40,
            value=20
        )

        level = st.selectbox(
            "Level of Study",
            sorted(df["Level"].unique())
        )

        faculty = st.selectbox(
            "Faculty",
            sorted(df["Faculty"].unique())
        )

        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique())
        )

        cgpa = st.selectbox(
            "Current CGPA",
            sorted(df["CGPA"].unique())
        )

        ca_score = st.number_input(
            "Average CA Score",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

        attendance = st.selectbox(
            "Attendance",
            sorted(df["Attendance"].unique())
        )

    with col2:

        passed_all = st.selectbox(
            "Passed All Courses",
            sorted(df["Passed_All"].unique())
        )

        academic_performance = st.selectbox(
            "Academic Performance",
            sorted(df["Academic_Performance"].unique())
        )

        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=10.0,
            value=3.0
        )

        social_media = st.number_input(
            "Social Media Hours",
            min_value=0.0,
            max_value=15.0,
            value=3.0
        )

        internet = st.number_input(
            "Internet Research Hours",
            min_value=0.0,
            max_value=10.0,
            value=2.0
        )

        device = st.selectbox(
            "Learning Device",
            sorted(df["Learning_Device"].unique())
        )

        online_resources = st.selectbox(
            "Uses Online Resources",
            sorted(df["Uses_Online_Resources"].unique())
        )

        resource_frequency = st.selectbox(
            "Resource Frequency",
            sorted(df["Resource_Frequency"].unique())
        )

        distraction = st.selectbox(
            "Social Media Distraction",
            sorted(df["Social_Media_Distraction"].unique())
        )

        assignment = st.selectbox(
            "Assignment Completion",
            sorted(df["Assignment_Completion"].unique())
        )

        online_discussion = st.selectbox(
            "Online Discussion",
            sorted(df["Online_Discussion"].unique())
        )

        time_management = st.selectbox(
            "Time Management",
            sorted(df["Time_Management"].unique())
        )

    predict = st.form_submit_button(
        "🎯 Predict Academic Performance",
        use_container_width=True
    )

if predict:

    # Encode Faculty and Department
    faculty_encoded = faculty_encoder.transform([faculty])[0]
    department_encoded = department_encoder.transform([department])[0]

    # Create input dataframe
    input_data = pd.DataFrame({
        "Gender": [gender_map[gender]],
        "Age": [age],
        "Level": [level_map[level]],
        "Faculty": [faculty_encoded],
        "Department": [department_encoded],
        "CGPA": [cgpa_map[cgpa]],
        "CA_Score": [ca_score],
        "Attendance": [attendance_map[attendance]],
        "Passed_All": [passed_map[passed_all]],
        "Academic_Performance": [performance_map[academic_performance]],
        "Study_Hours": [study_hours],
        "Social_Media_Hours": [social_media],
        "Internet_Research_Hours": [internet],
        "Learning_Device": [device_map[device]],
        "Uses_Online_Resources": [online_resource_map[online_resources]],
        "Resource_Frequency": [resource_frequency_map[resource_frequency]],
        "Social_Media_Distraction": [likert_map[distraction]],
        "Assignment_Completion": [assignment_map[assignment]],
        "Online_Discussion": [discussion_map[online_discussion]],
        "Time_Management": [time_map[time_management]]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🎉 The student is likely to PASS the semester.")
    else:
        st.error("⚠️ The student is AT RISK of poor academic performance.")

    confidence = max(probability)

    st.subheader("Prediction Confidence")

    st.progress(float(confidence))

    st.write(f"Confidence: {confidence * 100:.2f}%")

    probability_df = pd.DataFrame({
        "Class": ["At Risk", "Pass"],
        "Probability": probability
    })

    st.subheader("Class Probabilities")

    st.dataframe(probability_df, use_container_width=True)