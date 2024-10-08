import pandas as pd
import numpy as np
import random

# Expanded lists of car makes and models for each category
high_end_cars = [
    'Audi A4', 'Audi Q7', 'BMW X5', 'BMW 7 Series', 'Mercedes C-Class', 'Mercedes S-Class', 
    'Jaguar XF', 'Jaguar F-Pace', 'Porsche 911', 'Porsche Cayenne', 'Lexus LS', 'Maserati Ghibli'
]

average_cars = [
    'Ford Focus', 'Ford Fusion', 'Suzuki Swift', 'Suzuki Baleno', 'Toyota Corolla', 'Toyota Camry', 
    'Honda Civic', 'Honda Accord', 'Nissan Altima', 'Nissan Sentra', 'Hyundai Elantra', 'Kia Forte'
]

performance_cars = [
    'Chevrolet Camaro', 'Chevrolet Corvette', 'Dodge Challenger', 'Dodge Charger', 
    'Ford Mustang', 'Subaru WRX', 'Mazda MX-5', 'Nissan GT-R', 'BMW M3', 'Porsche Boxster'
]

eco_friendly_cars = [
    'Tesla Model 3', 'Tesla Model S', 'Nissan Leaf', 'Toyota Prius', 'Hyundai Ioniq', 
    'Chevrolet Bolt', 'BMW i3', 'Kia Niro EV', 'Honda Clarity', 'Ford Mustang Mach-E'
]

family_cars = [
    'Toyota Highlander', 'Honda CR-V', 'Kia Sorento', 'Hyundai Tucson', 'Ford Explorer', 
    'Chevrolet Traverse', 'Subaru Outback', 'Mazda CX-5', 'Nissan Rogue', 'Volkswagen Tiguan'
]

# General attributes for other cars
other_cars = [
    'Chevrolet Malibu', 'Volkswagen Passat', 'Hyundai Kona', 'Kia Optima', 'Tata Harrier',
    'Jeep Grand Cherokee', 'GMC Acadia', 'Chrysler Pacifica', 'Dodge Durango', 'Buick Enclave'
]

# Define potential attributes logically suitable for each category
body_types_dict = {
    'high_end': ['Sedan', 'SUV', 'Coupe', 'Convertible'],
    'average': ['Sedan', 'Hatchback', 'Wagon'],
    'performance': ['Coupe', 'Convertible', 'Sedan'],
    'eco_friendly': ['Sedan', 'Hatchback', 'SUV'],
    'family': ['SUV', 'Minivan', 'Wagon'],
    'other': ['SUV', 'Sedan', 'Hatchback', 'Wagon']
}

engine_types_dict = {
    'high_end': ['Petrol', 'Diesel', 'Hybrid'],
    'average': ['Petrol', 'Diesel'],
    'performance': ['Petrol', 'Diesel'],
    'eco_friendly': ['Electric', 'Hybrid'],
    'family': ['Petrol', 'Diesel', 'Hybrid'],
    'other': ['Petrol', 'Diesel', 'Hybrid']
}

transmission_types_dict = {
    'high_end': ['Automatic'],
    'average': ['Manual', 'Automatic'],
    'performance': ['Manual', 'Automatic'],
    'eco_friendly': ['Automatic'],
    'family': ['Automatic'],
    'other': ['Manual', 'Automatic']
}

seating_capacity_dict = {
    'high_end': [4, 5],
    'average': [5],
    'performance': [2, 4],
    'eco_friendly': [4, 5],
    'family': [5, 7],
    'other': [4, 5, 7]
}

drive_types_dict = {
    'high_end': ['AWD', 'RWD'],
    'average': ['FWD'],
    'performance': ['RWD', 'AWD'],
    'eco_friendly': ['FWD', 'AWD'],
    'family': ['AWD', 'FWD'],
    'other': ['FWD', 'AWD']
}

comfort_levels = {
    'high_end': [8,9,10],
    'average': [4,5,6],
    'performance': [5,6,7],
    'eco_friendly': [6,7,8],
    'family': [8,9,10],
    'other': [5, 6, 7]
}


# Function to generate random sample values for each car
def generate_random_car_data(car_id):
    # Randomly choose a car make with adjusted attributes based on the type
    car = random.choice(high_end_cars + average_cars + performance_cars + eco_friendly_cars + family_cars + other_cars)
    
    # Determine the category of the car and set attributes logically
    if car in high_end_cars:
        category = 'high_end'
        price = round(random.uniform(100000, 200000), 2)  # High price for luxury cars
        safety_rating = random.randint(9, 10)  # High safety rating
        performance_rating = random.randint(8, 10)  # High performance rating
        resale_value = round(price * random.uniform(0.7, 0.9), 2)  # High resale value
        fuel_economy = round(random.uniform(8, 12), 1)  # Poor fuel economy
        maintenance_cost = round(random.uniform(4000, 7000), 2)  # High maintenance cost
    
    elif car in average_cars:
        category = 'average'
        price = round(random.uniform(15000, 30000), 2)  # Lower price for budget cars
        safety_rating = random.randint(4, 6)  # Average safety rating
        performance_rating = random.randint(4, 6)  # Average performance rating
        resale_value = round(price * random.uniform(0.4, 0.6), 2)  # Lower resale value
        fuel_economy = round(random.uniform(20, 30), 1)  # High fuel economy
        maintenance_cost = round(random.uniform(500, 1500), 2)  # Lower maintenance cost
    
    elif car in performance_cars:
        category = 'performance'
        price = round(random.uniform(60000, 100000), 2)  # High price for performance cars
        safety_rating = random.randint(6, 8)  # Moderate safety rating
        performance_rating = random.randint(9, 10)  # Very high performance rating
        resale_value = round(price * random.uniform(0.6, 0.8), 2)  # Moderate resale value
        fuel_economy = round(random.uniform(8, 15), 1)  # Low fuel economy
        maintenance_cost = round(random.uniform(2500, 5500), 2)  # Moderate to high maintenance cost
    
    elif car in eco_friendly_cars:
        category = 'eco_friendly'
        price = round(random.uniform(30000, 60000), 2)  # Moderate price for eco-friendly cars
        safety_rating = random.randint(6, 8)  # High safety rating
        performance_rating = random.randint(6, 8)  # Average performance rating
        resale_value = round(price * random.uniform(0.6, 0.8), 2)  # Moderate resale value
        fuel_economy = round(random.uniform(25, 30), 1)  # Very high fuel economy
        maintenance_cost = round(random.uniform(500, 2000), 2)  # Lower maintenance cost
    
    elif car in family_cars:
        category = 'family'
        price = round(random.uniform(25000, 50000), 2)  # Moderate price for family cars
        safety_rating = random.randint(8, 10)  # High safety rating
        performance_rating = random.randint(6, 8)  # Average performance rating
        resale_value = round(price * random.uniform(0.6, 0.8), 2)  # Moderate resale value
        fuel_economy = round(random.uniform(15, 25), 1)  # Good fuel economy
        maintenance_cost = round(random.uniform(1000, 2500), 2)  # Moderate maintenance cost
    
    else:
        category = 'other'
        price = round(random.uniform(30000, 60000), 2)  # Moderate price for other cars
        safety_rating = random.randint(6, 8)  # Moderate safety rating
        performance_rating = random.randint(6, 8)  # Moderate performance rating
        resale_value = round(price * random.uniform(0.6, 0.8), 2)  # Moderate resale value
        fuel_economy = round(random.uniform(15, 25), 1)  # Average fuel economy
        maintenance_cost = round(random.uniform(1500, 3000), 2)  # Moderate maintenance cost
    
    # Set logically matching attributes
    body_type = random.choice(body_types_dict[category])
    engine_type = random.choice(engine_types_dict[category])
    year = random.randint(2010, 2024)  # Year of manufacture
    transmission = random.choice(transmission_types_dict[category])
    user_rating = random.randint(5, 10)  # Random user rating
    comfort_level = random.choice(comfort_levels[category])
    warranty_period = random.randint(1, 5)  # Warranty period in years
    seating_capacity = random.choice(seating_capacity_dict[category])
    battery_capacity = round(random.uniform(20, 100), 1) if engine_type == 'Electric' else np.nan  # Set battery capacity only if the engine type is Electric
    drive_type = random.choice(drive_types_dict[category])

    return {
        'Car ID': car_id,
        'Car': car,
        'Body Type': body_type,
        'Engine Type': engine_type,
        'Price ($)': price,
        'Year of Manufacture': year,
        'Transmission Type': transmission,
        'Resale Value ($)': resale_value,
        'Fuel Economy (km/l)': fuel_economy,
        'Performance Rating': performance_rating,
        'User Rating': user_rating,
        'Safety Rating': safety_rating,
        'Comfort Level': comfort_level,
        'Maintenance Cost ($/yr)': maintenance_cost,
        'Warranty Period (years)': warranty_period,
        'Seating Capacity': seating_capacity,
        'Battery Capacity (kWh)': battery_capacity,
        'Drive Type': drive_type
    }

# Generating the dataset with 50 unique records for demonstration (adjust range for larger datasets)
large_car_dataset = pd.DataFrame([generate_random_car_data(i) for i in range(1, 500001)])

# Export the dataset to a CSV file
large_car_dataset.to_csv("new_dataset.csv", index=False)

print("New dataset generated and saved as 'new_dataset.csv'")

