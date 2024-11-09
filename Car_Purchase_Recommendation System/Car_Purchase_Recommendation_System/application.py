import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Inject custom CSS styles
def add_custom_styles():
    st.markdown(
        """
        <style>
        /* Set background color */
        body {
            background-color: #a9a9a9;
        }
        
        /* Center the app content */
        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
            padding: 20px;
        }

        /* Style the buttons */
        .stButton>button {
            border-radius: 12px;
            background-color: #002366;
            color: white;
            padding: 10px 24px;
            text-align: center;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: white;
            color: black;
            border: 2px solid #4CAF50;
        }

        /* Style the headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Arial', sans-serif;
            color: #2c3e50;
        }

        /* Style for the question text */
        .question {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }

        /* Customize columns */
        [data-testid="column"] {
            padding: 10px;
        }

        /* Style the images */
        img {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 5px;
            width: 100%;
            transition: transform 0.2s ease-in-out;
        }

        img:hover {
            transform: scale(1.05);
            border-color: #4CAF50;
        }

        /* Style the main content container */
        .main {
            background-color: #ADD8E6;
            margin-top:20px;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# Add header function for the top heading
def add_header():
    st.markdown(
        """
        <h1 style='text-align: center; color: #800020; font-family: Arial, sans-serif;'>
            AI Car Purchase Recommendation System  🙂
        </h1>
        """, 
        unsafe_allow_html=True
    )

# Call the function to add the styles
add_custom_styles()

# Call the function to add the header at the top
add_header()

# Initialize session state for tracking answers and progress
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# Define 16 questions and their corresponding image options
questions = {
    1: {
        'question': "What type of car body do you prefer?",
        'options': {
            'Sedan': 'images/sedan.png',
            'SUV': 'images/suv.png',
            'Coupe': 'images/coupe.png',
            'Convertible': 'images/convertible.png'
        }
    },
    2: {
        'question': "What is your preferred engine type?",
        'options': {
            'Petrol': 'images/petrol_engine.png',
            'Diesel': 'images/diesel_engine.png',
            'Electric': 'images/electric_engine.png',
            'Hybrid': 'images/hybrid_engine.png'
        }
    },
    3: {
        'question': "What is your budget for the car ($)?",
        'options': {
            20000: 'images/budget.png',
            50000: 'images/budget.png',
            80000: 'images/budget.png',
            120000: 'images/budget.png'
        }
    },
    4: {
        'question': "What is your preferred year of manufacture?",
        'options': {
            2010: 'images/year_of_manufacture.png',
            2015: 'images/year_of_manufacture.png',
            2020: 'images/year_of_manufacture.png',
            2024: 'images/year_of_manufacture.png'
        }
    },
    5: {
        'question': "What transmission type do you prefer?",
        'options': {
            'Manual': 'images/manual_trx.png',
            'Automatic': 'images/automatic_trx.png'
        }
    },
    6: {
        'question': "What is your expected resale value ($)?",
        'options': {
            10000: 'images/resale_value.png',
            20000: 'images/resale_value.png',
            40000: 'images/resale_value.png',
            80000: 'images/resale_value.png'
        }
    },
    7: {
        'question': "What is your preferred fuel economy (km/l)?",
        'options': {
            6: 'images/fuel_economy.png',
            10: 'images/fuel_economy.png',
            15: 'images/fuel_economy.png',
            20: 'images/fuel_economy.png'
        }
    },
    8: {
        'question': "What performance rating do you expect (1-10)?",
        'options': {
            3: 'images/performance.png',
            6: 'images/performance.png',
            8: 'images/performance.png',
            12: 'images/performance.png'
        }
    },
    9: {
        'question': "What is your preferred user rating (1-10)?",
        'options': {
            3: 'images/low_rating.png',
            6: 'images/low_rating.png',
            8: 'images/high_rating.png',
            10: 'images/high_rating.png'
        }
    },
    10: {
        'question': "What safety rating do you prefer?",
        'options': {
            5: 'images/safety_rating.png',
            7: 'images/safety_rating.png',
            8: 'images/safety_rating.png',
            10: 'images/safety_rating.png'
        }
    },
    11: {
        'question': "What comfort level do you prefer?",
        'options': {
            4: 'images/low_comfort.png',
            7: 'images/moderate_comfort.png',
            10: 'images/high_comfort.png'
        }
    },
    12: {
        'question': "What maintenance cost range do you prefer ($/year)?",
        'options': {
            500: 'images/maintenance_cost.png',
            2000: 'images/maintenance_cost.png',
            3500: 'images/maintenance_cost.png',
            6500: 'images/maintenance_cost.png'
        }
    },
    13: {
        'question': "What warranty period are you looking for?",
        'options': {
            1: 'images/warranty.png',
            2: 'images/warranty.png',
            3: 'images/warranty.png',
            4: 'images/warranty.png'
        }
    },
    14: {
        'question': "What is your seating capacity preference?",
        'options': {
            2: 'images/2seat.png',
            4: 'images/4seat.png',
            5: 'images/5seat.png',
            7: 'images/7seat.png'
        }
    },
    15: {
        'question': "What battery capacity do you prefer (kWh)?",
        'options': {
            25: 'images/small_battery.png',
            45: 'images/small_battery.png',
            75: 'images/large_battery.png',
            100: 'images/large_battery.png'
        }
    },
    16: {
        'question': "What is your preferred drive type?",
        'options': {
            'AWD': 'images/awd.png',
            'FWD': 'images/fwd.png',
            'RWD': 'images/rwd.png'
        }
    }
}

# Function to save user data to CSV
def save_to_csv(user_inputs):
    # Define the expected columns
    columns = [
        "Price ($)", "Year of Manufacture", "Resale Value ($)", "Fuel Economy (km/l)",
        "Performance Rating", "User Rating", "Safety Rating", "Comfort Level",
        "Maintenance Cost ($/yr)", "Warranty Period (years)", "Seating Capacity",
        "Battery Capacity (kWh)", "Body Type", "Engine Type", "Transmission Type", "Drive Type"
    ]

    # Map the user's selections to the correct column names
    user_data = {
        "Price ($)": user_inputs.get(3, None),  # Assuming question 3 is about the price
        "Year of Manufacture": user_inputs.get(4, None),  # Assuming question 4 is about the year of manufacture
        "Resale Value ($)": user_inputs.get(6, None),  # Assuming question 6 is about resale value
        "Fuel Economy (km/l)": user_inputs.get(7, None),  # Assuming question 7 is about fuel economy
        "Performance Rating": user_inputs.get(8, None),  # Assuming question 8 is about performance rating
        "User Rating": user_inputs.get(9, None),  # Assuming question 9 is about user rating
        "Safety Rating": user_inputs.get(10, None),  # Assuming question 10 is about safety rating
        "Comfort Level": user_inputs.get(11, None),  # Assuming question 11 is about comfort level
        "Maintenance Cost ($/yr)": user_inputs.get(12, None),  # Assuming question 12 is about maintenance cost
        "Warranty Period (years)": user_inputs.get(13, None),  # Assuming question 13 is about warranty period
        "Seating Capacity": user_inputs.get(14, None),  # Assuming question 14 is about seating capacity
        "Battery Capacity (kWh)": user_inputs.get(15, None),  # Assuming question 15 is about battery capacity
        "Body Type": user_inputs.get(1, None),  # Assuming question 1 is about body type
        "Engine Type": user_inputs.get(2, None),  # Assuming question 2 is about engine type
        "Transmission Type": user_inputs.get(5, None),  # Assuming question 5 is about transmission type
        "Drive Type": user_inputs.get(16, None),  # Assuming question 16 is about drive type
    }

    # Create a DataFrame with the user's data
    df = pd.DataFrame([user_data], columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv('user_selections.csv', index=False)
    st.write("Data saved to `user_selections.csv`")

def clean_and_process_user_data(csv_path='user_selections.csv'):
    # Load user data from CSV
    df = pd.read_csv(csv_path)

    # Perform cleaning and preprocessing
    scaler = MinMaxScaler(feature_range=(0.00, 0.95))  # Set the range to avoid zero values
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Expected columns for scaling and encoding
    numerical_columns = [
        "Price ($)", "Year of Manufacture", "Resale Value ($)", "Fuel Economy (km/l)",
        "Performance Rating", "User Rating", "Safety Rating", "Comfort Level",
        "Maintenance Cost ($/yr)", "Warranty Period (years)", "Seating Capacity",
        "Battery Capacity (kWh)"
    ]

    categorical_columns = [
        "Body Type", "Engine Type", "Transmission Type", "Drive Type"
    ]

    # Check if all numerical columns exist in the dataframe
    if not all(col in df.columns for col in numerical_columns):
        raise ValueError("Some numerical columns are missing from the input data")

    # Fit the scaler to the numerical columns in the dataset
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Encode categorical columns if present
    if not all(col in df.columns for col in categorical_columns):
        raise ValueError("Some categorical columns are missing from the input data")

    encoded_cats = encoder.fit_transform(df[categorical_columns])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine scaled numerical columns and encoded categorical columns
    df_final = pd.concat([df[numerical_columns], encoded_cats_df], axis=1)

    # Define the expected columns for the model
    expected_columns = [
        "Price ($)", "Year of Manufacture", "Resale Value ($)", "Fuel Economy (km/l)", 
        "Performance Rating", "User Rating", "Safety Rating", "Comfort Level", 
        "Maintenance Cost ($/yr)", "Warranty Period (years)", "Seating Capacity", 
        "Battery Capacity (kWh)", "Body Type_Convertible", "Body Type_Coupe", "Body Type_Hatchback", 
        "Body Type_Minivan", "Body Type_SUV", "Body Type_Sedan", "Body Type_Wagon", 
        "Engine Type_Diesel", "Engine Type_Electric", "Engine Type_Hybrid", 
        "Engine Type_Petrol", "Transmission Type_Automatic", "Transmission Type_Manual", 
        "Drive Type_AWD", "Drive Type_FWD", "Drive Type_RWD"
    ]

    # Add any missing expected columns with a default value of 0
    for col in expected_columns:
        if col not in df_final.columns:
            df_final[col] = 0

    # Reorder columns to match the expected order
    df_final = df_final[expected_columns]

    # Check if the column order and shape are correct
    if df_final.shape[1] != 28:
        raise ValueError(f"Processed data does not have 28 columns. Current columns count: {df_final.shape[1]}")

    # Save the processed data to be used by the model
    df_final.to_csv('processed_user_data.csv', index=False)
    return df_final


def predict_car_recommendation(processed_data):
    from tensorflow.keras.models import load_model

    # Load the trained model
    model = load_model('fine_tuned_car_recommendation_model.keras')

    # Convert processed_data to a NumPy array of float32
    if isinstance(processed_data, pd.DataFrame):
        X = processed_data.to_numpy()
    else:
        X = np.array(processed_data)

    # Ensure it's a float32 type to avoid conversion issues
    X = X.astype(np.float32)

    # Check for NaN or Inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input data contains NaN or Inf values. Please handle them appropriately.")
    
    # Perform the prediction
    predictions = model.predict(X)[0]  # Assuming a single-row input

    # Get the index of the car with the highest predicted probability
    predicted_index = np.argmax(predictions)

    # Load the cleaned dataset to retrieve car names
    car_mapping = pd.read_csv('cleaned_dataset.csv')  # This dataset should have the mapping of indices to car names
    recommended_car = car_mapping.iloc[predicted_index]['Car']  # Get the car name

    return f"""
        <div style='text-align: center; background-color: #f0f8ff; padding: 20px; border-radius: 12px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);'>
            <h2 style='color: #0073e6; font-family: Arial, sans-serif;'>
                Hi buddy 🙂! , here is my personalized recommendation for you 🙂!
                 The most suitable car for you would be : 
            </h2>
            <h1 style='color: #ff4500; font-family: "Trebuchet MS", sans-serif;'>
                🚘 {recommended_car} 🚘
            </h1>
        </div>
    """


# Function to display 4 questions at a time with clickable image options
def show_question_with_images(question_number):
    question = questions[question_number]
    st.write(f'<div class="question">{question["question"]}</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    for idx, (option, img_url) in enumerate(question['options'].items()):
        key = f"q{question_number}_{idx}"  # Ensure unique keys
        if idx == 0:
            with col1:
                if st.button(str(option), key=key):
                    st.session_state.user_inputs[question_number] = option
                st.image(img_url, use_column_width=True)
        elif idx == 1:
            with col2:
                if st.button(str(option), key=key):
                    st.session_state.user_inputs[question_number] = option
                st.image(img_url, use_column_width=True)
        elif idx == 2:
            with col3:
                if st.button(str(option), key=key):
                    st.session_state.user_inputs[question_number] = option
                st.image(img_url, use_column_width=True)
        elif idx == 3:
            with col4:
                if st.button(str(option), key=key):
                    st.session_state.user_inputs[question_number] = option
                st.image(img_url, use_column_width=True)

# Function to display 4 questions at a time
def display_step(step):
    st.write('<div class="main">', unsafe_allow_html=True)
    if step == 1:
        st.write("<h2>Step 1: Choose Your Preferences (1-4)</h2>", unsafe_allow_html=True)
        show_question_with_images(1)
        show_question_with_images(2)
        show_question_with_images(3)
        show_question_with_images(4)
        if st.button("Next", key="next1"):
            st.session_state.step += 1

    elif step == 2:
        st.write("<h2>Step 2: Choose Your Preferences (5-8)</h2>", unsafe_allow_html=True)
        show_question_with_images(5)
        show_question_with_images(6)
        show_question_with_images(7)
        show_question_with_images(8)
        if st.button("Next", key="next2"):
            st.session_state.step += 1

    elif step == 3:
        st.write("<h2>Step 3: Choose Your Preferences (9-12)</h2>", unsafe_allow_html=True)
        show_question_with_images(9)
        show_question_with_images(10)
        show_question_with_images(11)
        show_question_with_images(12)
        if st.button("Next", key="next3"):
            st.session_state.step += 1

    elif step == 4:
        st.write("<h2>Step 4: Choose Your Preferences (13-16)</h2>", unsafe_allow_html=True)
        show_question_with_images(13)
        show_question_with_images(14)
        show_question_with_images(15)
        show_question_with_images(16)
        if st.button("Submit", key="submit"):
            # Save user selections to CSV
            save_to_csv(st.session_state.user_inputs)
            
            # Clean and process user data
            processed_data = clean_and_process_user_data('user_selections.csv')
            
            # After predicting car recommendation
            recommended_car_html = predict_car_recommendation(processed_data)

            # Display the styled HTML using Streamlit's markdown feature with the "unsafe_allow_html" parameter
            st.markdown(recommended_car_html, unsafe_allow_html=True)
            st.session_state.step += 1

    elif step == 5:
        st.write("<h2>Review Your Choices</h2>", unsafe_allow_html=True)
        
        # Apply custom CSS for the table
        st.markdown(
            """
            <style>
            /* Make the table rows and columns white */
            .review-table thead th {
                background-color: #FFFFFF;
                color: #2c3e50;
            }
            .review-table tbody tr {
                background-color: #FFFFFF;
                color: #2c3e50;
            }
            .review-table tbody tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        # Create a DataFrame from user inputs for better visualization
        review_data = pd.DataFrame.from_dict(st.session_state.user_inputs, orient='index', columns=['Selected Option'])
        
        # Reset the index to make the question numbers readable
        review_data.index.name = 'Question'
        review_data.reset_index(inplace=True)
        
        # Display the DataFrame as a table in Streamlit with a class for styling
        st.write(review_data.to_html(classes='review-table', index=False), unsafe_allow_html=True)

        if st.button("Start Over", key="start_over"):
            st.session_state.step = 1
            st.session_state.user_inputs = {}
        st.write('</div>', unsafe_allow_html=True)


# Display the current step
display_step(st.session_state.step)
