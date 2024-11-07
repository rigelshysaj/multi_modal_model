# -*- coding: utf-8 -*-
"""Multimodal Regression Model"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


# Set constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Tabular data preprocessor
class TabularPreprocessor:
    """Preprocessor for tabular data (Step 1)"""
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_cols = None
        self.categorical_cols = None
        self.target_transform = 'log'  # Options: 'log', 'normalize', None

    def fit_transform(self, data):
        df = data.copy()

        # Remove 'description' column
        df = df.drop('description', axis=1, errors='ignore')

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Apply target transformation if specified
        if self.target_transform == 'log':
            df['target'] = np.log1p(df['target'])
        elif self.target_transform == 'normalize':
            self.target_mean = df['target'].mean()
            self.target_std = df['target'].std()
            df['target'] = (df['target'] - self.target_mean) / self.target_std

        # Identify categorical and numerical columns
        self.categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.drop('target', errors='ignore').tolist()
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('target', errors='ignore').tolist()

        # Identify columns with all NaNs and remove them
        cols_with_all_nan = [col for col in self.numerical_cols if df[col].isnull().all()]
        if cols_with_all_nan:
            print(f"Dropping columns with all NaNs: {cols_with_all_nan}")
            df.drop(columns=cols_with_all_nan, inplace=True)
            self.numerical_cols = [col for col in self.numerical_cols if col not in cols_with_all_nan]

        # Ensure numerical columns are numeric
        for col in self.numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        df[self.numerical_cols] = self.numerical_imputer.fit_transform(df[self.numerical_cols])
        df[self.categorical_cols] = self.categorical_imputer.fit_transform(df[self.categorical_cols])

        # Encode categorical features
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Scale numerical features
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])

        return df

    def transform(self, data):
        df = data.copy()

        # Remove 'description' column if present
        df = df.drop('description', axis=1, errors='ignore')

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Apply the same transformation to 'target' if present
        if 'target' in df.columns and self.target_transform == 'log':
            df['target'] = np.log1p(df['target'])
        elif 'target' in df.columns and self.target_transform == 'normalize':
            df['target'] = (df['target'] - self.target_mean) / self.target_std

        # Use the columns from fit_transform
        numerical_cols = [col for col in self.numerical_cols if col in df.columns]
        categorical_cols = [col for col in self.categorical_cols if col in df.columns]

        # Ensure numerical columns are numeric
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        df[numerical_cols] = self.numerical_imputer.transform(df[numerical_cols])
        df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])

        # Encode categorical features
        for col in categorical_cols:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])

        # Scale numerical features
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        return df

    def inverse_transform_target(self, y):
        if self.target_transform == 'log':
            return np.expm1(y)
        elif self.target_transform == 'normalize':
            return y * self.target_std + self.target_mean
        else:
            return y

# Step 1 Pipeline
class Step1Pipeline:
    """Complete pipeline for Step 1 (tabular data only)"""
    def __init__(self):
        self.preprocessor = TabularPreprocessor()
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_SEED
        )

    def train(self, data):
        # Preprocess the data
        processed_data = self.preprocessor.fit_transform(data)

        # Split the data
        X = processed_data.drop('target', axis=1)
        y = processed_data['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Apply inverse transformation to predictions
        y_train_pred_inv = self.preprocessor.inverse_transform_target(y_train_pred)
        y_test_pred_inv = self.preprocessor.inverse_transform_target(y_test_pred)
        y_train_inv = self.preprocessor.inverse_transform_target(y_train)
        y_test_inv = self.preprocessor.inverse_transform_target(y_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_inv, y_train_pred_inv)
        test_mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
        train_r2 = r2_score(y_train_inv, y_train_pred_inv)
        test_r2 = r2_score(y_test_inv, y_test_pred_inv)

        print(f"Step 1 - Train MAE: {train_mae:.4f}")
        print(f"Step 1 - Test MAE: {test_mae:.4f}")
        print(f"Step 1 - Train R2 Score: {train_r2:.4f}")
        print(f"Step 1 - Test R2 Score: {test_r2:.4f}")

        return X_train, X_test, y_train, y_test

    def predict(self, data):
        processed_data = self.preprocessor.transform(data)
        predictions = self.model.predict(processed_data)
        return self.preprocessor.inverse_transform_target(predictions)
