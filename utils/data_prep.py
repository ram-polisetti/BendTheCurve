"""
Helper functions for data preparation and analysis in BendTheCurve blog posts.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def load_and_clean_data(filepath: str,
                       date_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[List[str]] = None,
                       numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and perform basic cleaning on a dataset.
    
    Args:
        filepath: Path to the data file (csv, excel, etc.)
        date_columns: List of column names to parse as dates
        categorical_columns: List of column names to treat as categorical
        numerical_columns: List of column names to treat as numerical
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Determine file type and read accordingly
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")
    
    # Convert date columns
    if date_columns:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
    
    # Convert categorical columns
    if categorical_columns:
        for col in categorical_columns:
            df[col] = df[col].astype('category')
    
    # Convert numerical columns
    if numerical_columns:
        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def handle_missing_values(data: pd.DataFrame,
                         strategy: Dict[str, str] = None,
                         drop_thresh: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        data: Input DataFrame
        strategy: Dictionary mapping column names to imputation strategy
                 ('mean', 'median', 'most_frequent', or 'constant')
        drop_thresh: Drop columns with more missing values than this threshold
    
    Returns:
        DataFrame with handled missing values
    """
    df = data.copy()
    
    # Drop columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    df = df.drop(columns=missing_ratio[missing_ratio > drop_thresh].index)
    
    if strategy:
        for col, method in strategy.items():
            if col in df.columns:
                imputer = SimpleImputer(strategy=method)
                df[col] = imputer.fit_transform(df[[col]])
    
    return df

def create_features(data: pd.DataFrame,
                   date_column: str = None,
                   cyclical_features: bool = True,
                   lag_features: List[int] = None) -> pd.DataFrame:
    """
    Create common features from existing data.
    
    Args:
        data: Input DataFrame
        date_column: Name of the date column to extract features from
        cyclical_features: Whether to create cyclical features from dates
        lag_features: List of lag periods to create
    
    Returns:
        DataFrame with additional features
    """
    df = data.copy()
    
    if date_column and date_column in df.columns:
        # Extract basic date features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        
        if cyclical_features:
            # Create cyclical features for month and day of week
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    if lag_features:
        for lag in lag_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def scale_features(data: pd.DataFrame,
                  columns: List[str] = None,
                  scaler_type: str = 'standard') -> pd.DataFrame:
    """
    Scale numerical features in the dataset.
    
    Args:
        data: Input DataFrame
        columns: List of columns to scale (if None, scales all numeric columns)
        scaler_type: Type of scaling ('standard' or 'minmax')
    
    Returns:
        DataFrame with scaled features
    """
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler type")
    
    df[columns] = scaler.fit_transform(df[columns])
    
    return df

def encode_categorical(data: pd.DataFrame,
                      columns: List[str] = None,
                      method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        data: Input DataFrame
        columns: List of categorical columns to encode
        method: Encoding method ('onehot' or 'label')
    
    Returns:
        DataFrame with encoded categorical variables
    """
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['category', 'object']).columns
    
    for col in columns:
        if method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        elif method == 'label':
            df[col] = pd.factorize(df[col])[0]
        else:
            raise ValueError("Unsupported encoding method")
    
    return df
