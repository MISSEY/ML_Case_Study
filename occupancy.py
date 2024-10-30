import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import os 
import joblib
import argparse
import json


class Occupancy:
    def __init__(self,model_path):
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = model_path

    def load_data(self, data_path):
        """Load and preprocess the data"""
        # Read data
        df = pd.read_csv(data_path)
        
        # Convert date string to datetime temporarily for feature extraction
        df['date'] = pd.to_datetime(df['date'])

        print("\nDataset Date Range:")
        print(f"Start Date: {df['date'].min()}")
        print(f"End Date: {df['date'].max()}")
        print(f"Number of unique days: {df['date'].dt.date.nunique()}")
        
        return df
    
    def analyze_correlations(self, df, target=None):
        """Analyze and visualize correlations"""
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Print correlations with target if specified
        if target is not None and target in df.columns:
            target_corr = corr_matrix[target].sort_values(ascending=False)
            print("\nCorrelations with Occupancy:")
            for feature, corr in target_corr.items():
                if feature != target:
                    if feature.startswith('day_of_week'):
                        day_num = int(feature.split('_')[-1])
                        day_name = pd.Timestamp(2024, 1, 1 + day_num).strftime('%A')
                        print(f"{day_name}: {corr:.4f}")
                    else:
                        print(f"{feature}: {corr:.4f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(24, 20))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('correlation_heatmap.png')
        plt.close()

    def feature_engineering(self, df):
        """Extract datetime features and create one-hot encodings"""
        # Extract datetime features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Drop the original date column
        df = df.drop('date', axis=1)

        print("\nFeature Ranges:")
        print(f"Hours (0-23): {sorted(df['hour'].unique())}")
        print(f"Days of Week (0-6): {sorted(df['day_of_week'].unique())}")

        
        
        # One-hot encode categorical features
        categorical_features = ['hour', 'day_of_week']
        df_encoded = pd.get_dummies(df, columns=categorical_features)

        # Ensure all possible values are present
        # For hours (0-23)
        for hour in range(24):
            if f'hour_{hour}' not in df_encoded.columns:
                df_encoded[f'hour_{hour}'] = bool(0)
                
        # For weekdays (0-6)
        for day in range(7):
            if f'day_of_week_{day}' not in df_encoded.columns:
                df_encoded[f'day_of_week_{day}'] = bool(0)

        print("\nOne-hot encoded features:")
        for prefix in ['hour_', 'day_']:
            cols = [col for col in df_encoded.columns if col.startswith(prefix)]
            print(f"{prefix[:-1]}: {len(cols)} features")
        
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Select numerical features
        numerical_features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        
        # Get one-hot encoded columns
        categorical_features = [col for col in df.columns if col.startswith(('hour_', 'day_'))]

        print("\nFeature Summary:")
        print(f"Number of numerical features: {len(numerical_features)}")
        print(f"Number of encoded features: {len(categorical_features)}")
        print(f"Total features: {len(numerical_features) + len(categorical_features)}")
        
        # Combine features
        feature_columns = numerical_features + categorical_features
        
        # Analyze correlations before splitting
        self.analyze_correlations(df, target='Occupancy')
        
        # Split into features and target
        X = df[feature_columns]
        y = df['Occupancy']
        
        return X, y
    
    def train(self, X_train, y_train, random_state=42):
        """Train the XGBoost model with random search"""
        # Scale numerical features
        numerical_features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        # Define parameter grid for random search
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        best_score = 0
        best_params = None
        
        n_iterations = 10
        for i in range(n_iterations):
            # Randomly sample parameters
            params = {
                key: np.random.choice(values) 
                for key, values in param_grid.items()
            }
            
            # Train model with current parameters
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=random_state,
                **params
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Calculate validation score
            y_pred = model.predict_proba(X_train_scaled)[:, 1]
            score = roc_auc_score(y_train, y_pred)
            
            # Print progress
            print(f"Iteration {i+1}/{n_iterations}, Score: {score:.4f}")
            
            # Update best parameters if score improves
            if score > best_score:
                best_score = score
                best_params = params
                self.model = model
        
        print(f"\nBest ROC AUC Score: {best_score:.4f}")
        print("Best Parameters:", best_params)
        
        # Print feature importance
        self.print_feature_importance(X_train.columns)
        
        return self.model
    
    def save_model(self):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(self.model_path, 'xgboost_model.joblib')
        scaler_path = os.path.join(self.model_path, 'scaler.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self,):
        """Load saved model and scaler"""
        model_path = os.path.join(self.model_path, 'xgboost_model.joblib')
        scaler_path = os.path.join(self.model_path, 'scaler.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler file not found")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        print("Model and scaler loaded successfully")

    def print_feature_importance(self, feature_names):
        """Print feature importance from the trained model"""
        if self.model is not None:
            importance = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            for idx, row in feature_imp.iterrows():
                if row['Feature'].startswith('day_'):
                    day_num = int(row['Feature'].split('_')[-1])
                    day_name = pd.Timestamp(2024, 1, 1 + day_num).strftime('%A')
                    print(f"{day_name}: {row['Importance']:.4f}")
                else:
                    print(f"{row['Feature']}: {row['Importance']:.4f}")

    def predict_single(self, data):
        """Make prediction for single data point using one-hot encoding"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Extract time features
            if isinstance(data['date'], str):
                timestamp = pd.to_datetime(data['date'])
            else:
                timestamp = data['date']
                
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            data.pop('date','None')
            # Create empty dataframe with numerical features
            features = pd.DataFrame(data, index=[0])
            
            # Create one-hot encoded columns for hour (0-23)
            for h in range(24):
                features[f'hour_{h}'] = bool(1) if hour == h else bool(0)
                
            # Create one-hot encoded columns for day_of_week (0-6)
            for d in range(7):
                features[f'day_of_week_{d}'] = bool(1) if day_of_week == d else bool(0)
            
            # Ensure columns are in the correct order
            if hasattr(self.model, 'feature_names_'):
                features = features.reindex(columns=self.model.feature_names_)
            
            # # Print feature details for debugging
            # print("\nFeature details:")
            # print(f"Hour: {hour} (one-hot column: hour_{hour} = 1)")
            # print(f"Day of week: {day_of_week} (one-hot column: day_of_week_{day_of_week} = 1)")
            # print("\nFeature columns:", features.columns.tolist())
            
            # Scale numerical features
            numerical_features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
            features[numerical_features] = self.scaler.transform(features[numerical_features])
            
            # Make prediction
            prob = self.model.predict_proba(features)[0, 1]
            prediction = int(prob >= 0.5)
            
            return {
                'timestamp': timestamp,
                'occupied': prediction,
                'probability': float(prob),
                'hour': hour,
                'day_of_week': day_of_week,
                'day_name': timestamp.strftime('%A')
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Scale test features
        numerical_features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        X_test_scaled = X_test.copy()
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Get predictions
        y_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"Test ROC AUC Score: {roc_auc:.4f}")
        
        return roc_auc

def train(data_path,model_path):

    
    if data_path is None:
        data_path = 'occupancy_data'
        model_path = 'models'

    # Initialize pipeline
    pipeline = Occupancy(model_path=model_path)
    
    # Load and preprocess data
    df_train = pipeline.load_data(f'{data_path}/datatraining.txt')
    df_train = pipeline.feature_engineering(df_train)
    df_test = pipeline.load_data(f'{data_path}/datatest.txt')
    df_test = pipeline.feature_engineering(df_test)
    # Prepare features
    X_train, y_train = pipeline.prepare_features(df_train)
    X_test, y_test = pipeline.prepare_features(df_test)
    
    
    # Train model
    model = pipeline.train(X_train, y_train)
    
    # Evaluate model
    roc_auc = pipeline.evaluate(X_test, y_test)
    print("\nSaving model...")
    pipeline.save_model()
    xcom_value = {
        'roc_auc': float(roc_auc),  # Convert numpy float to Python float
    }
    return xcom_value

def evaluate(data_path,model_path):
    # Initialize pipeline
    if data_path is None:
        data_path = 'occupancy_data'
        model_path = 'models'

    # Initialize pipeline
    pipeline = Occupancy(model_path=model_path)

    df_test = pipeline.load_data(f'{data_path}/datatest.txt')
    # Load saved model
    print("Loading model...")
    pipeline.load_model()
    df_test = pipeline.feature_engineering(df_test)
    X_test, y_test = pipeline.prepare_features(df_test)
    roc_auc = pipeline.evaluate(X_test, y_test)
    xcom_value = {
        'roc_auc': float(roc_auc),  # Convert numpy float to Python float
    }
    return xcom_value
    # data = df_test.iloc[1550].to_dict()
    
    # # Make prediction
    # print("\nMaking prediction...")

    # print('GT:','Yes' if data['Occupancy'] else 'No')

    # data.pop('Occupancy',None)
    # result = pipeline.predict_single(data)
    # print("\nPrediction result:")
    # print(f"Timestamp: {result['timestamp']}")
    # print(f"Occupied: {'Yes' if result['occupied'] else 'No'}")
    # print(f"Probability: {result['probability']:.4f}")

def prepare_args():
    """
    prepare arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluate', action='store_true',required=False,help='evaluate')
    parser.add_argument('-d', '--data_path',default=None,required=False,help='data path')
    parser.add_argument('-m', '--model_path',default=None,required=False,help='model path')

    return parser

if __name__ == "__main__":
    parser =prepare_args()
    args = parser.parse_args()
    if args.evaluate:
        roc_auc = evaluate(data_path=args.data_path,model_path=args.model_path)
    else:
        roc_auc = train(data_path=args.data_path,model_path=args.model_path)
    
    return_json = {"roc_auc": f"{roc_auc}"}

    f = open("/airflow/xcom/return.json", "w")
    f.write(f"{return_json}")
    f.close()
