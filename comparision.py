import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from logger import Logger
import warnings
warnings.filterwarnings('ignore')

class ManufacturingMLAnalysis:
    def __init__(self):
        self.log = Logger()
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier()
        }
        self.results = {}
        self.best_model = None
        self.best_accuracy = 0
        
    def load_data(self):
        try:
            self.log.info("Loading dataset...")
            df = pd.read_csv(r'Dataset\manufacturing_data.csv')
            
            # Select relevant features
            features = [
                'Temperature', 'Vibration', 'Pressure', 'Run_Time',
                'Oil_Level', 'Power_Consumption', 'Product_Rate',
                'Maintenance_Due', 'Quality_Score'
            ]
            
            X = df[features]
            
            # Convert Yes/No to 1/0
            le = LabelEncoder()
            y = le.fit_transform(df['Downtime'])
            
            self.log.info(f"Selected features: {features}")
            self.log.info(f"Dataset shape: {X.shape}")
            return X, y
            
        except Exception as e:
            self.log.error(f"Error in data loading: {str(e)}")
            raise

    def prepare_data(self, X, y):
        try:
            self.log.info("Preparing and splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            self.log.error(f"Error in data preparation: {str(e)}")
            raise

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        try:
            plt.figure(figsize=(6, 4))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.log.error(f"Error plotting confusion matrix: {str(e)}")

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        try:
            self.log.info(f"\nEvaluating {model_name}...")
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            self.results[model_name] = {
                'Accuracy': accuracy,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std()
            }
            
            # Update best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model_name
            
            # Log results
            self.log.info(f"Accuracy: {accuracy:.4f}")
            self.log.info(f"Cross-validation mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            self.log.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            
        except Exception as e:
            self.log.error(f"Error in model evaluation for {model_name}: {str(e)}")

    def plot_model_comparison(self):
        try:
            # Create comparison dataframe
            results_df = pd.DataFrame(self.results).T
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            results_df[['Accuracy', 'CV Mean']].plot(kind='bar', yerr=results_df['CV Std'])
            plt.title('Model Comparison')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Print detailed comparison
            self.log.info("\nModel Comparison:")
            self.log.info("\n" + str(results_df.round(4)))
            self.log.info(f"\nBest performing model: {self.best_model} with accuracy: {self.best_accuracy:.4f}")
            
        except Exception as e:
            self.log.error(f"Error in plotting model comparison: {str(e)}")

    def run_analysis(self):
        try:
            # Load and prepare data
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            
            # Evaluate each model
            for name, model in self.models.items():
                self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            
            # Plot final comparison
            self.plot_model_comparison()
            
        except Exception as e:
            self.log.error(f"Error in analysis execution: {str(e)}")
            raise

if __name__ == "__main__":
    analysis = ManufacturingMLAnalysis()
    analysis.run_analysis()