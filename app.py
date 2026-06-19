import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Title

st.title("🎓 Student Performance Predictor")

st.write("Predict student marks using Machine Learning")

# Load dataset

data = pd.read_csv("student_data.csv")

# Features and target

X = data[['studyhours', 'attendance', 'assignments']]

y = data['marks']

# Train model

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)

model = LinearRegression()

model.fit(X_train, y_train)

# Inputs

study_hours = st.slider("Study Hours", 0, 12, 5)

attendance = st.slider("Attendance (%)", 0, 100, 75)

assignments = st.slider("Assignments Completed", 0, 10, 5)

# Predict button

if st.button("Predict Performance"):

    new_data = pd.DataFrame({

        "studyhours": [study_hours],

        "attendance": [attendance],

        "assignments": [assignments]

    })

    prediction = model.predict(new_data)

    st.success(f"Predicted Marks: {prediction[0]:.2f}")

    if prediction[0] >= 90:

        st.info("Grade: A+")

    elif prediction[0] >= 75:

        st.info("Grade: A")

    elif prediction[0] >= 60:

        st.info("Grade: B")

    else:

        st.info("Grade: C")

    accuracy = model.score(X_test, y_test)

    st.write(f"Model Accuracy: {accuracy*100:.2f}%")

    # Graph

    st.subheader("Study Hours vs Marks")

    fig, ax = plt.subplots()

    ax.scatter(data["studyhours"], data["marks"])

    # Show user's prediction point

    ax.scatter(

        study_hours,

        prediction[0],

        s=150

    )

    ax.set_xlabel("Study Hours")

    ax.set_ylabel("Marks")

    ax.set_title("Student Performance Prediction")

    st.pyplot(fig)

# Dataset preview

st.subheader("Dataset")
st.dataframe(data)