import pandas as pd
import joblib
from logger import Logger
import os

class DowntimePredictor:
    def __init__(self):
        self.log = Logger()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = None
        self.load_model()

    def load_model(self):
        """Load the saved model and preprocessors"""
        try:
            model_path = 'Models'
            self.log.info("Loading model and preprocessors...")
            
            # Check if model directory exists
            if not os.path.exists(model_path):
                raise FileNotFoundError("Models directory not found!")
            
            # Load model components
            self.model = joblib.load(os.path.join(model_path, 'decision_tree_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
            
            # Load feature list
            with open(os.path.join(model_path, 'feature_list.txt'), 'r') as f:
                self.features = f.read().splitlines()
            
            self.log.info("Model loaded successfully!")
            self.log.info(f"Required features: {', '.join(self.features)}")
            
        except Exception as e:
            self.log.error(f"Error loading model: {str(e)}")
            raise

    def validate_input(self, input_data):
        """Validate input data"""
        try:
            missing_features = set(self.features) - set(input_data.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Check data types
            for feature in self.features:
                if not isinstance(input_data[feature], (int, float)):
                    raise ValueError(f"Feature '{feature}' must be numeric")
                    
            return True
            
        except Exception as e:
            self.log.error(f"Input validation error: {str(e)}")
            raise

    def predict(self, input_data):
        """Make prediction for input data"""
        try:
            # Validate input
            self.validate_input(input_data)
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[self.features]
            
            # Scale features
            scaled_input = self.scaler.transform(input_df)
            
            # Make prediction
            prediction_encoded = self.model.predict(scaled_input)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            self.log.info("Prediction made successfully!")
            return prediction
            
        except Exception as e:
            self.log.error(f"Error making prediction: {str(e)}")
            raise

def get_user_input():
    """Get input values from user"""
    input_data = {}
    try:
        print("\nPlease enter the following values:")
        input_data['Temperature'] = float(input("Temperature (°C): "))
        input_data['Vibration'] = float(input("Vibration level (0-1): "))
        input_data['Pressure'] = float(input("Pressure: "))
        input_data['Run_Time'] = float(input("Run Time (hours): "))
        input_data['Oil_Level'] = float(input("Oil Level (0-1): "))
        input_data['Power_Consumption'] = float(input("Power Consumption (%): "))
        input_data['Product_Rate'] = float(input("Product Rate (units/hour): "))
        input_data['Maintenance_Due'] = int(input("Maintenance Due (0 or 1): "))
        input_data['Quality_Score'] = float(input("Quality Score (0-100): "))
        return input_data
    except ValueError as e:
        print(f"Invalid input: {str(e)}")
        return None

def main():
    # Initialize predictor
    predictor = DowntimePredictor()
    
    while True:
        print("\n1. Enter new data for prediction")
        print("2. Use sample data")
        print("3. Exit")
        choice = input("Choose an option (1-3): ")
        
        if choice == '1':
            input_data = get_user_input()
            if input_data:
                try:
                    result = predictor.predict(input_data)
                    print(f"\nPredicted Downtime: {result}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
        elif choice == '2':
            # Sample data
            sample_data = {
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
            try:
                result = predictor.predict(sample_data)
                print("\nUsing sample data:")
                for key, value in sample_data.items():
                    print(f"{key}: {value}")
                print(f"\nPredicted Downtime: {result}")
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == '3':
            print("Exiting program...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()