import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from logger import Logger

class ManufacturingModelPipeline:
    def __init__(self):
        self.log = Logger()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = [
            'Temperature', 'Vibration', 'Pressure', 'Run_Time',
            'Oil_Level', 'Power_Consumption', 'Product_Rate',
            'Maintenance_Due', 'Quality_Score'
        ]
        
        # Create models directory
        os.makedirs('Models', exist_ok=True)
        
    def load_and_prepare_data(self):
        try:
            self.log.info("Loading dataset...")
            df = pd.read_csv(r'Dataset\manufacturing_data.csv')
            
            X = df[self.features]
            y = df['Downtime']  # Keep as Yes/No for now
            
            self.log.info(f"Dataset shape: {X.shape}")
            self.log.info(f"Feature list: {self.features}")
            
            return X, y
            
        except Exception as e:
            self.log.error(f"Error in data loading: {str(e)}")
            raise

    def train_model(self):
        try:
            # Load and split data
            X, y = self.load_and_prepare_data()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Define parameter grid for GridSearchCV
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            # Initialize and train model with GridSearchCV
            self.log.info("Starting model training with GridSearchCV...")
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train_encoded)
            
            # Get best model
            self.model = grid_search.best_estimator_
            
            # Log best parameters
            self.log.info(f"Best parameters found: {grid_search.best_params_}")
            self.log.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Convert predictions back to Yes/No
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            
            # Log performance metrics
            self.log.info("\nModel Performance on Test Set:")
            self.log.info(f"\nClassification Report:\n{classification_report(y_test, y_pred_labels)}")
            
            # Save models and transformers
            self.save_pipeline()
            
            # Plot and save confusion matrix
            self.plot_confusion_matrix(y_test, y_pred_labels)
            
        except Exception as e:
            self.log.error(f"Error in model training: {str(e)}")
            raise

    def save_pipeline(self):
        try:
            self.log.info("\nSaving model and transformers...")
            
            # Save model, scaler, and label encoder
            joblib.dump(self.model, 'Models/decision_tree_model.pkl')
            joblib.dump(self.scaler, 'Models/scaler.pkl')
            joblib.dump(self.label_encoder, 'Models/label_encoder.pkl')
            
            # Save feature list
            with open('Models/feature_list.txt', 'w') as f:
                f.write('\n'.join(self.features))
            
            self.log.info("Model pipeline saved successfully!")
            
        except Exception as e:
            self.log.error(f"Error in saving pipeline: {str(e)}")
            raise

    def plot_confusion_matrix(self, y_true, y_pred):
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title('Confusion Matrix - Decision Tree')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('Models/confusion_matrix.png')
            plt.close()
            
        except Exception as e:
            self.log.error(f"Error plotting confusion matrix: {str(e)}")

class ManufacturingPredictor:
    def __init__(self):
        self.log = Logger()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = None
        self.load_pipeline()

    def load_pipeline(self):
        try:
            self.log.info("Loading model pipeline...")
            
            # Load model and transformers
            self.model = joblib.load('Models/decision_tree_model.pkl')
            self.scaler = joblib.load('Models/scaler.pkl')
            self.label_encoder = joblib.load('Models/label_encoder.pkl')
            
            # Load feature list
            with open('Models/feature_list.txt', 'r') as f:
                self.features = f.read().splitlines()
            
            self.log.info("Model pipeline loaded successfully!")
            
        except Exception as e:
            self.log.error(f"Error in loading pipeline: {str(e)}")
            raise

    def predict(self, input_data):
        try:
            # Convert input data to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Verify features
            missing_features = set(self.features) - set(input_data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Scale features
            X_scaled = self.scaler.transform(input_data[self.features])
            
            # Make prediction
            prediction_encoded = self.model.predict(X_scaled)
            
            # Convert prediction to Yes/No
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction[0]
            
        except Exception as e:
            self.log.error(f"Error in making prediction: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Train and save model
    pipeline = ManufacturingModelPipeline()
    pipeline.train_model()
    
    # Example prediction
    predictor = ManufacturingPredictor()
    
    # Example input data
    sample_input = {
        'Temperature': 85.5,
        'Vibration': 0.5,
        'Pressure': 100.0,
        'Run_Time': 12.0,
        'Oil_Level': 0.8,
        'Power_Consumption': 75.0,
        'Product_Rate': 95.0,
        'Maintenance_Due': 0,
        'Quality_Score': 90.0
    }
    
    # Make prediction
    prediction = predictor.predict(pd.DataFrame([sample_input]))
    print(f"\nPrediction for sample input: {prediction}")