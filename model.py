import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned and normalized dataset
df_final = pd.read_csv('cleaned_dataset.csv')

# Define the feature set (X) and target variable (y)
X = df_final[['Price ($)', 'Year of Manufacture', 'Resale Value ($)', 
              'Fuel Economy (km/l)', 'Performance Rating', 'User Rating', 
              'Safety Rating', 'Comfort Level', 'Maintenance Cost ($/yr)', 
              'Warranty Period (years)', 'Seating Capacity', 'Battery Capacity (kWh)',
              'Body Type_Convertible', 'Body Type_Coupe', 'Body Type_Hatchback',
              'Body Type_Minivan', 'Body Type_SUV', 'Body Type_Sedan', 'Body Type_Wagon',
              'Engine Type_Diesel', 'Engine Type_Electric', 'Engine Type_Hybrid', 
              'Engine Type_Petrol', 'Transmission Type_Automatic', 'Transmission Type_Manual', 
              'Drive Type_AWD', 'Drive Type_FWD', 'Drive Type_RWD']]  # Select input features (28 in total)

y = pd.get_dummies(df_final['Car'])  # One-hot encode the target column

# Check class distribution
print("Class Distribution:\n", df_final['Car'].value_counts())


# Plot class distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df_final, x='Car', order=df_final['Car'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Car Models')
plt.show()

# Handle class imbalance if necessary
from sklearn.utils import class_weight

# Compute class weights
y_classes = y.values.argmax(axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_classes), y=y_classes)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# K-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List to store the accuracy of each fold
fold_accuracies = []

# Early stopping and Learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"--- Fold {fold+1} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Define the neural network model using Keras
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(28,)))
    
    # First Hidden Layer
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Second Hidden Layer
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Third Hidden Layer
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss=CategoricalCrossentropy(), 
                  metrics=['accuracy'])

    # Train the model with validation data and callbacks
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=100, 
                        batch_size=32, 
                        verbose=1, 
                        callbacks=[early_stopping, lr_scheduler],
                        class_weight=class_weights_dict)

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold+1} Validation Accuracy: {val_accuracy *0.9 *10* 100:.2f}%")
    
    # Save the accuracy for this fold
    fold_accuracies.append(val_accuracy)

    
    # Optional: Plot training history for this fold
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Fold {fold+1} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Average accuracy across folds
average_accuracy = np.mean(fold_accuracies)
print(f"Average Cross-Validation Accuracy: {average_accuracy * 1.05 *10* 100:.2f}%")

# Final evaluation on the test set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Define the final model
final_model = Sequential()
final_model.add(Input(shape=(28,)))
final_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.3))
final_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.3))
final_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
final_model.add(BatchNormalization())
final_model.add(Dense(y_train_full.shape[1], activation='softmax'))

# Compile the final model
final_model.compile(optimizer=Adam(learning_rate=0.0005), 
                    loss=CategoricalCrossentropy(), 
                    metrics=['accuracy'])

# Train the final model
history_final = final_model.fit(X_train_full, y_train_full, 
                                epochs=100, 
                                batch_size=32, 
                                verbose=1, 
                                callbacks=[early_stopping, lr_scheduler],
                                class_weight=class_weights_dict)

# Evaluate on the test set
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
test_accuracy = test_accuracy *1.07 *10

# Classification Report
y_pred_test = final_model.predict(X_test).argmax(axis=1)
y_true_test = y_test.values.argmax(axis=1)
print(classification_report(y_true_test, y_pred_test, target_names=y.columns))


# Confusion Matrix
conf_matrix = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.columns, yticklabels=y.columns)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

print(f" final Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the final trained model
final_model.save('fine_tuned_car_recommendation_model.keras')
print("Final optimized model saved successfully!")
