import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataPreprocessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def preprocess_data(self):
        df = pd.read_csv(self.dataset_path)
        df_encoded = self.OneHotEncoder_features(df)
        df_features = self.feature_importance(df_encoded)
        df_refined = self.select_features(df_encoded)
        df_scaled = self.scaling(df_refined) 
        features = df_scaled.columns.tolist()
        return(features,df_scaled)

    #Let's perform one-hot encoding on the categorical features
    def OneHotEncoder_features(self, df): 
        # Selecting categorical columns for one-hot encoding
        categorical_columns = ['Industry', 'Ethnicity', 'Citizen']

        # Applying OneHotEncoder
        encoder = OneHotEncoder()
        encoded_features_sparse = encoder.fit_transform(df[categorical_columns])

        # Converting to dense array manually
        encoded_features = encoded_features_sparse.toarray()
        
        # Correctly generating column names
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
        
        # Dropping the original categorical columns and 'ZipCode' from the dataset
        df_reduced = df.drop(categorical_columns + ['ZipCode'], axis=1)
        
        # Ensuring indices align for concatenation
        df_reduced.reset_index(drop=True, inplace=True)
        encoded_df.reset_index(drop=True, inplace=True)
        
        # Concatenating the encoded features with the rest of the dataset
        df_encoded = pd.concat([df_reduced, encoded_df], axis=1)

        return(df_encoded)
    

    def feature_importance(self, df_encoded):
        # Separating the features and the target variable
        X = df_encoded.drop('Approved', axis=1)
        y = df_encoded['Approved']
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fitting the model to the training data
        rf.fit(X_train, y_train) 

        # Getting feature importances
        feature_importances = rf.feature_importances_
        # Creating a DataFrame to visualize the feature importances
        features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

        return(features_df)
    
    def select_features(self, df_encoded):  
        # Refining the dataset to include only the selected top features and the target variable
        selected_features = ['PriorDefault', 'CreditScore', 'YearsEmployed', 'Income', 'Debt', 'Age', 'Employed', 'Approved']
        df_refined = df_encoded[selected_features]
        return(df_refined)

    

    def scaling(self, df_refined): 
        # Applying log transformation with a small constant to handle zero values
        df_transformed = df_refined.copy()
        df_transformed['CreditScore'] = np.log(df_transformed['CreditScore'] + 1)
        df_transformed['YearsEmployed'] = np.log(df_transformed['YearsEmployed'] + 1)
        df_transformed['Income'] = np.log(df_transformed['Income'] + 1)
        df_transformed['Debt'] = np.log(df_transformed['Debt'] + 1)
        df_transformed['Age'] = np.log(df_transformed['Age'] + 1)


        # Initializing the MinMaxScaler
        scaler = MinMaxScaler()

        # Selecting the numerical features for scaling
        features_to_scale = ['CreditScore', 'YearsEmployed', 'Income', 'Debt', 'Age' ]
        
        # Applying Min-Max scaling
        df_transformed[features_to_scale] = scaler.fit_transform(df_transformed[features_to_scale])
        return(df_transformed)
