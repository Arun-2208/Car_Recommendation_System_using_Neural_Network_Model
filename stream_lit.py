import streamlit as st
import numpy as np

# Define the car attributes matching the dataset used during training
car_attributes = [
    'Year of Manufacture', 'Price ($)', 'Maintenance Cost ($/yr)', 'Engine Type', 'Fuel Economy (km/l)',
    'Body Type', 'Transmission Type', 'User Rating', 'Safety Rating', 'Comfort Level',
    'Performance Rating', 'Warranty Period (years)', 'Seating Capacity', 'Battery Capacity (kWh)', 'Drive Type'
]

# Sample options for user selections
engine_options = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
body_type_options = ['Sedan', 'Hatchback', 'SUV', 'Convertible', 'Coupe', 'Minivan', 'Wagon', 'Truck']
transmission_options = ['Manual', 'Automatic']
drive_type_options = ['FWD', 'RWD', 'AWD']

# Set page configuration and styling
st.set_page_config(page_title="Car Purchase Recommendation System", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4fa;
        font-family: Arial, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #333;
        margin-bottom: 20px;
    }
    
    .question-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        padding: 10px 20px;
        border-radius: 20px;
        background-color: #2b67f6;
        color: white;
        text-align: center;
    }
    .confirmation-pill {
        display: inline-block;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 25px;
        background-color: #ffcccb;
        color: #333;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display a large car image at the top center
#st.image("image.png", width=150)  # Replace "image.png" with the correct path if needed

# Display the main title
st.markdown('<div class="main-title">Car Purchase Recommendation System</div>', unsafe_allow_html=True)

# Function to get user input
def get_user_input():
    user_input = {}

    # Using session state to track progress through questions
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.answers = {}

    # Skip the Battery Capacity question if engine type is not Electric or Hybrid
    battery_capacity_question = None
    if st.session_state.answers.get('Engine Type') in ['Electric', 'Hybrid']:
        battery_capacity_question = ("If considering an electric or hybrid car, what is the minimum battery capacity (in kWh) you require?", st.slider, {'label': 'Battery Capacity (kWh)', 'min_value': 20, 'max_value': 100, 'value': 50})

    # Define the questions
    questions = [
        ("What is your preferred year of manufacture for the car?", st.slider, {'label': 'Year of Manufacture', 'min_value': 2010, 'max_value': 2024, 'value': 2020}),
        ("What is your budget for the ex-showroom price of the car?", st.slider, {'label': 'Ex-Showroom Price ($)', 'min_value': 10000, 'max_value': 200000, 'value': 50000, 'step': 5000}),
        ("What is the maximum annual maintenance cost you are willing to incur?", st.slider, {'label': 'Maintenance Cost ($/yr)', 'min_value': 500, 'max_value': 7000, 'value': 1500, 'step': 500}),
        ("Which engine type do you prefer?", st.selectbox, {'label': 'Engine Type', 'options': engine_options}),
        ("What is the minimum fuel economy (in km/l) you are looking for?", st.slider, {'label': 'Fuel Economy (km/l)', 'min_value': 8, 'max_value': 30, 'value': 15}),
        ("What type of car body are you interested in?", st.selectbox, {'label': 'Body Type', 'options': body_type_options}),
        ("Do you prefer an automatic or manual transmission?", st.selectbox, {'label': 'Transmission Type', 'options': transmission_options}),
        ("What is the minimum user rating (out of 10) you are willing to consider?", st.slider, {'label': 'User Rating', 'min_value': 1, 'max_value': 10, 'value': 5}),
        ("How important is safety to you? What is the minimum safety rating you require?", st.slider, {'label': 'Safety Rating', 'min_value': 1, 'max_value': 10, 'value': 5}),
        ("What level of comfort (out of 10) are you looking for in your car?", st.slider, {'label': 'Comfort Level', 'min_value': 1, 'max_value': 10, 'value': 5}),
        ("Are there specific performance metrics that are important to you?", st.slider, {'label': 'Performance Rating', 'min_value': 1, 'max_value': 10, 'value': 5}),
        ("How long of a warranty period are you looking for?", st.slider, {'label': 'Warranty Period (years)', 'min_value': 1, 'max_value': 5, 'value': 3}),
        ("What is the minimum seating capacity you require in a car?", st.selectbox, {'label': 'Seating Capacity', 'options': [4, 5, 7]}),
        battery_capacity_question,
        ("Do you have a preference for the drive type?", st.selectbox, {'label': 'Drive Type', 'options': drive_type_options}),
    ]

    # Remove None values from the questions list
    questions = [q for q in questions if q is not None]

    # Safeguard to prevent index error when all questions are completed
    if st.session_state.question_index < len(questions):
        current_question, input_function, kwargs = questions[st.session_state.question_index]

        if kwargs:
            st.markdown('<div class="question-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="question-title">{current_question}</div>', unsafe_allow_html=True)
            answer = input_function(**kwargs)

            if st.button("Next"):
                st.session_state.answers[car_attributes[st.session_state.question_index]] = answer
                st.session_state.question_index += 1
            st.markdown('</div>', unsafe_allow_html=True)

    # If all questions are answered, show the submit button
    elif st.session_state.question_index >= len(questions):
        if st.button("Submit"):
            st.write("Your preferences have been recorded. Calculating the best car recommendation for you...")
            st.markdown("### Your Selected Preferences:")
            for key, value in st.session_state.answers.items():
                st.markdown(f'<span class="confirmation-pill">{key}: {value}</span>', unsafe_allow_html=True)

# Main execution flow
get_user_input()
