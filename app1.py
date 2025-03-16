import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load the model
loaded_model = pickle.load(open('wine_classifier.sav', 'rb'))

# Load the dataset
wine = pd.read_csv('wine.csv')

# Function for wine prediction
def wine_prediction(input_data):
    convert_data_to_numpy = np.asarray(input_data)
    reshape_input_data = convert_data_to_numpy.reshape(1, -1)
    prediction = loaded_model.predict(reshape_input_data)
    return "This Wine Quality is Bad." if prediction == 0 else "This Wine Quality is Good."

# Chat page (Charts and Insights)
def chat_page():
    st.title("Chart Page")
    
    # Count wine quality occurrences
    wine_counts = wine["quality"].value_counts().reset_index()
    wine_counts.columns = ["quality", "count"]
    
    # Bar plot for wine quality
    fig = px.bar(wine_counts, x="quality", y="count", labels={"quality": "Quality", "count": "Count"}, title="Wine Quality Status")
    st.plotly_chart(fig)
    
    # Pie chart for Alcohol Content vs Wine Quality
    wine_alcohol = wine.groupby('quality')['alcohol'].sum().reset_index()
    fig = px.pie(wine_alcohol, names='quality', values='alcohol', title='Alcohol Content vs Wine Quality', color_discrete_sequence=['lightblue', 'darkblue'])
    st.plotly_chart(fig)
    
    # Pie chart for Volatile Acidity vs Wine Quality
    wine_acidity = wine.groupby('quality')['volatile_acidity'].sum().reset_index()
    fig = px.pie(wine_acidity, names='quality', values='volatile_acidity', title='Volatile Acidity vs Wine Quality', color_discrete_sequence=['darkblue', 'skyblue'])
    st.plotly_chart(fig)
    
    # Display value counts
    st.write("Wine Quality Count:")
    st.write(wine['quality'].value_counts())
    
    # Problem Statement
    st.subheader("Problem Statement")
    st.markdown("""
    - Wine Quality Classification: Predict wine quality based on its chemical features using a classification model, with 0.6 as the benchmark.
    """)
    
    # Objective
    st.subheader("Problem Objective")
    st.markdown("""
    - Objective: Develop a machine learning model that classifies wine quality using physicochemical properties.
    """)
    
    # Observations
    st.subheader("Observation")
    st.markdown("""
    - Wine Quality Status: The bar chart shows a relatively balanced dataset with high-quality and low-quality wines.
    - Alcohol Content vs. Wine Quality: A higher percentage of good-quality wines have higher alcohol content.
    - Volatile Acidity vs. Wine Quality: Lower volatile acidity is slightly more associated with high-quality wines.
    """)
    
    # Conclusion
    st.subheader("Conclusion")
    st.markdown("""
    - The analysis highlights the key factors influencing wine quality, with alcohol content and volatile acidity playing significant roles.
    - The classification model can be used to predict wine quality effectively, providing useful insights for wine producers and consumers.
    """)

# Dashboard page (User Input & Prediction)
def dashboard_page():
    st.title("Dashboard Page")
    st.markdown("<h3 style='text-align: center;'>Wine Quality Prediction</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: green; font-size: 16px;'>Input the required values:</h5>", unsafe_allow_html=True)

    # User input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.number_input('Fixed Acidity', value=0.0)
        volatile_acidity = st.number_input('Volatile Acidity', value=0.0)
        citric_acid = st.number_input('Citric Acid', value=0.0)
    
    with col2:
        residual_sugar = st.number_input('Residual Sugar', value=0.0)
        chlorides = st.number_input('Chlorides', value=0.0)
        sulphates = st.number_input('Sulphates', value=0.0)
    
    with col3:
        density = st.number_input('Density', value=0.0)
        pH = st.number_input('pH', value=0.0)
        alcohol = st.number_input('Alcohol', value=0.0)
    
    # Additional inputs
    free_sulfur_dioxide = st.selectbox('Free Sulfur Dioxide', [0.0, 10.0, 20.0, 30.0])   
    total_sulfur_dioxide = st.selectbox('Total Sulfur Dioxide', [0.0, 10.0, 20.0, 30.0], index=0)
    
    # Predict button
    if st.button('Wine Quality Application'):
        try:
            input_data = [
                float(fixed_acidity), float(volatile_acidity), float(citric_acid),
                float(residual_sugar), float(chlorides), float(density),
                float(free_sulfur_dioxide), float(total_sulfur_dioxide),
                float(pH), float(sulphates), float(alcohol)
            ]
            result = wine_prediction(input_data)
            st.success(result)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Main function to switch between pages
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Chart", "Dashboard"])
    if page == "Chart":
        chat_page()
    elif page == "Dashboard":
        dashboard_page()

# Run the app
if __name__ == "__main__":
    main()
