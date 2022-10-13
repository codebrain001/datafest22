import streamlit as st  # Import the streamlit package (as st by convention)
st.set_page_config(layout="wide") # Increase the width of the web application
import pandas as pd # For data wrangling operations
import numpy as np # For scientific computing operations
import pickle # To load pickle file
from PIL import Image # Using the Python Imaging Library to load images
from pathlib import Path

app_path = Path(__file__).parents[0]
pkl_path = str(app_path) + "/pipeline.pkl"

# load the model pipeline from the app directory
model_pipeline = pickle.load(open(pkl_path, "rb"))

def main():
    # Setting application title
    st.title("Bank Customers Churn Prediction App")
    # Setting Application description
    st.markdown(
        """
        :money: This streamlit app predicts the bank's customer churn status. The application is functional for both online and batch predictions. \n
        The project and dataset can be found on GitHub <a href="https://github.com/codebrain001/datafest22"> here</a> and Kaggle <a href="https://github.com/codebrain001/datafest22"> here</a> respectively.
        """, unsafe_allow_html=True
    )

    # Setting Application sidebar default
    app_image = Image.open("images/datafest-app.jpg")    # loading image from image folder
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Online", "Batch")
    )
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(app_image)

    if add_selectbox == "Online":
        st.info("Input data below")

        # Create form to capture input data
        with st.form("my_form"):
            id = st.number_input('ID',  min_value=0, max_value=20000000, value=0)
            surname = st.text_input('Surname')
            creditScore = st.number_input('Credit score',  min_value=0, max_value=1000, value=0)
            geography = st.selectbox("Geography", ('France', 'Spain', 'Germany'))
            gender = st.radio('Gender',('Male', 'Female'))
            age = st.number_input('Age',  min_value=0, max_value=100, value=0)
            tenure = st.slider('Account tenure (years)', min_value=0, max_value=20, value=0)
            balance = st.number_input('Account balance',  min_value=0.00, max_value=500000.00, value=0.00, format="%.2f")
            numOfProducts= st.slider('Number of bank products', min_value=0, max_value=5, value=0)
            creditCard = st.radio('Credit Card',('Yes', 'No'))
            activeMember = st.radio('Active member',('Yes', 'No'))
            estimatedSalary = st.number_input('Estimated salary',  min_value=0.00, max_value=500000.00, value=0.00, format="%.2f")

            # Creating form dictionary of inputs
            data = {
                "CustomerId": [id],
                "Surname": [surname],
                "CreditScore": [creditScore],
                "Geography": [geography],
                "Gender": [gender],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [numOfProducts],
                "HasCrCard": [creditCard],
                "IsActiveMember": [activeMember],
                "EstimatedSalary": [estimatedSalary],
            }

            # Create form dataframe to apply model pipeline
            form_df = pd.DataFrame(data)
            # Dropping customer IDs and surname (PII)
            form_df.drop(columns=['CustomerId', 'Surname'], inplace=True)
            # Encoding features (Recall from the initial dataframe and description on the dataset on Kaggle, some features were numeric feature, we just made it categorical in the form for intuition)
            form_df[["HasCrCard", "IsActiveMember"]] = form_df[["HasCrCard", "IsActiveMember"]].replace({'Yes': 1, 'No': 0})

            # applying model pipeline from form dataframe to make prediction
            prediction = model_pipeline.predict(form_df)

            # Every form must have a submit button.
            submitted = st.form_submit_button("Predict")

            if submitted:
                if prediction == 1:
                    st.warning("The customer will churn")
                else:
                    st.success("The customer is satisfied with the bank services")

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.subheader('Data overview')
            st.write(batch_data.style.format(subset=['Balance', 'EstimatedSalary'], formatter="{:.2f}"))

        # Submit batch data to predict
        if st.button('Predict'):
            if uploaded_file is None:
                st.warning("No data upload, please upload batch data",  icon="⚠️")
            else:
                # Get batch prediction
                batch_data['prediction'] = model_pipeline.predict(batch_data)
                # select customerID & prediction columns
                predictions_df = batch_data[["CustomerId", "prediction"]]
                predictions_df = predictions_df.replace({1:'The customer will churn',
                                                    0:'The customer is satisfied with the bank services'})

                st.subheader('Prediction')
                st.write(predictions_df)

if __name__ == '__main__':
    main()