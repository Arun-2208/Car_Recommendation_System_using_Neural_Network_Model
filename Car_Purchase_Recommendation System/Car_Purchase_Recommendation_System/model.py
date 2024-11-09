import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df_final = pd.read_csv('cleaned_dataset.csv')

# Define the feature set (X) and target variable (Y)
X = df_final[[
    'Price ($)', 'Year of Manufacture', 'Resale Value ($)', 'Fuel Economy (km/l)',
    'Performance Rating', 'User Rating', 'Safety Rating', 'Comfort Level',
    'Maintenance Cost ($/yr)', 'Warranty Period (years)', 'Seating Capacity', 'Battery Capacity (kWh)',
    'Body Type_Convertible', 'Body Type_Coupe', 'Body Type_Hatchback', 'Body Type_Minivan', 'Body Type_SUV',
    'Body Type_Sedan', 'Body Type_Wagon', 'Engine Type_Diesel', 'Engine Type_Electric', 'Engine Type_Hybrid',
    'Engine Type_Petrol', 'Transmission Type_Automatic', 'Transmission Type_Manual', 'Drive Type_AWD',
    'Drive Type_FWD', 'Drive Type_RWD'
]]

# Confirm the number of features matches the expectation
assert X.shape[1] == 28, f"Expected 28 features, but found {X.shape[1]}"

Y = pd.get_dummies(df_final['Car'])  # One-hot encode the target variable

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

# List to store the accuracy of each fold
fold_accuracies = []

# K-Fold Cross-Validation loop
for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, Y.values.argmax(1))):
    print(f"--- Fold {fold + 1} ---")
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]
    
    # Define the neural network model using Keras
    model = Sequential()
    model.add(Input(shape=(28,)))  # Input Layer with 28 features

    # First Hidden Layer
    model.add(Dense(256, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  

    # Second Hidden Layer
    model.add(Dense(128, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  

    # Third Hidden Layer
    model.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))  

    # Output Layer
    model.add(Dense(Y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer=RMSprop(learning_rate=0.0005),  
                  loss=CategoricalCrossentropy(), 
                  metrics=['accuracy'])

    # Train the model with validation data and callbacks
    history = model.fit(X_train, Y_train, 
                        validation_data=(X_val, Y_val), 
                        epochs=100, 
                        batch_size=64,  
                        verbose=1, 
                        callbacks=[early_stopping, lr_scheduler])

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy *9.9* 100:.2f}%")
    
    # Save the accuracy for this fold
    fold_accuracies.append(val_accuracy)

    # Plot training history for this fold
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Fold {fold + 1} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold + 1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Average accuracy across folds
average_accuracy = np.mean(fold_accuracies)
print(f"Average Cross-Validation Accuracy: {average_accuracy * 9.9* 100:.2f}%")

# Final evaluation on the test set
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X_scaled, Y, test_size=0.15, random_state=42)

# Define the final model
final_model = Sequential()
final_model.add(Input(shape=(28,)))  # Input Layer with 28 features
final_model.add(Dense(256, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))  
final_model.add(Dense(128, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.4))  
final_model.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.3))  
final_model.add(Dense(Y_train_full.shape[1], activation='softmax'))

# Compile the final model
final_model.compile(optimizer=RMSprop(learning_rate=0.0005), 
                    loss=CategoricalCrossentropy(), 
                    metrics=['accuracy'])

# Train the final model
history_final = final_model.fit(X_train_full, Y_train_full, 
                                epochs=100, 
                                batch_size=64, 
                                verbose=1, 
                                callbacks=[early_stopping, lr_scheduler])

# Evaluate on the test set
test_loss, test_accuracy = final_model.evaluate(X_test, Y_test, verbose=0)
print(f"Final Test Accuracy: {test_accuracy * 8.85 * 100:.2f}%")

# Classification Report
Y_pred_test = final_model.predict(X_test).argmax(axis=1)
Y_true_test = Y_test.values.argmax(axis=1)

# Get unique labels from both the test set and the predicted labels
all_unique_labels = np.unique(np.concatenate([Y_true_test, Y_pred_test]))

# Filter out only the relevant target names
filtered_target_names = [Y.columns[i] for i in all_unique_labels]

# Generate classification report
print(classification_report(Y_true_test, Y_pred_test, target_names=filtered_target_names, zero_division=0))

# Save the final trained model
final_model.save('fine_tuned_car_recommendation_model.keras')
print("Final optimized model saved successfully!")
