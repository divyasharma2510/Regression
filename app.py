#Simple linear regression model to preduc the salary based on years of experience

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Simple Linear Regression - Salary Prediction")

#Loading the dataset
st.header("Step 1: Load and Display Data")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)

    #Step 2: Visualize the dataset
    st.header("Step 2: Visualize Salary vs Experience")
    fig, ax = plt.subplots()
    ax.scatter(data["YearsExperience"], data["Salary"],color='blue')
    ax.set_xlabel("Years of Experience") #Independent Variable
    ax.set_ylabel("Salary") #Dependent Variable 
    ax.set_title("Scatter Plot")
    st.pyplot(fig)

    #Step 3: Train Model
    st.header("Step 3: Train Linear Regression Model")
    X = data[["YearsExperience"]] #input data 
    y = data["Salary"]

    #st.write(X)
    #st.write(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train holds 80% data of i/p
    #X_test holds 20% data of i/p
    #y_train holds 80% data of o/p
    #y_test holds 20% data of o/p

    model = LinearRegression() #LinearRegression: class, creating obj: 'model' for that class so we can we the methods of that class
    model.fit(X_train, y_train)
    
    st.success("Model Trained Successfully")

    #Step 4: Visualize Regression Line
    st.header("Step 4: Visualize the Regression line")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='green', label='Actual Data')
    ax2.plot(X["YearsExperience"], model.predict(X), color='red', label='Regression Line')
    ax2.set_xlabel("Years of Experience") #Independent Variable
    ax2.set_ylabel("Salary") #Dependent Variable 
    ax2.legend()
    st.pyplot(fig2)

    #Step 5: Make Predictions
    st.header("Step 5: Predict Salary")
    experience = st.number_input("Enter years of experience: ", min_value=0.0, step=0.1)
    if st.button("Predict"):
        prediction = model.predict([[experience]])
        st.success(f"Predicted Salary: {prediction[0]:,.2f}")


    