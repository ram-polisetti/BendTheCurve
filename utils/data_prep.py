"""
Helper functions for data preparation and analysis in BendTheCurve blog posts.

This module provides a collection of utility functions for common data preprocessing tasks:
1. Loading and cleaning data from various file formats
2. Feature engineering and transformation
3. Scaling and normalization
4. Categorical encoding

Example usage:
    ```python
    # Load and prepare data
    df = load_and_clean_data(
        'data.csv',
        date_columns=['date'],
        categorical_columns=['category'],
        numerical_columns=['value']
    )
    
    # Create time-based features
    df_features = create_features(
        df,
        date_column='date',
        cyclical_features=True,
        lag_features=[1, 7, 30]
    )
    
    # Scale numerical features
    df_scaled = scale_features(
        df_features,
        columns=['value'],
        scaler_type='standard'
    )
    ```
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_and_clean_data(filepath: str,
                       date_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[List[str]] = None,
                       numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and perform basic cleaning on a dataset.
    
    This function handles:
    - Loading data from various file formats (csv, excel)
    - Converting columns to appropriate data types
    - Basic data type validation
    
    Args:
        filepath: Path to the data file (csv, excel, etc.)
        date_columns: List of column names to parse as dates
        categorical_columns: List of column names to treat as categorical
        numerical_columns: List of column names to treat as numerical
    
    Returns:
        Cleaned pandas DataFrame
    
    Example:
        >>> df = load_and_clean_data(
        ...     'sales.csv',
        ...     date_columns=['order_date'],
        ...     categorical_columns=['product_category'],
        ...     numerical_columns=['price', 'quantity']
        ... )
    """
    # Determine file type and read accordingly
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.split('.')[-1]}")
    
    # Convert date columns
    if date_columns:
        for col in date_columns:
            if col not in df.columns:
                raise ValueError(f"Date column '{col}' not found in data")
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col not in df.columns:
                raise ValueError(f"Categorical column '{col}' not found in data")
            df[col] = df[col].astype('category')
    
    # Convert numerical columns
    if numerical_columns:
        for col in numerical_columns:
            if col not in df.columns:
                raise ValueError(f"Numerical column '{col}' not found in data")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def handle_missing_values(data: pd.DataFrame,
                         strategy: Dict[str, str] = None,
                         drop_thresh: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    This function handles:
    - Dropping columns with high missing value rates
    - Imputing missing values using various strategies
    
    Args:
        data: Input DataFrame
        strategy: Dictionary mapping column names to imputation strategy
                 ('mean', 'median', 'most_frequent', or 'constant')
        drop_thresh: Drop columns with more missing values than this threshold
    
    Returns:
        DataFrame with handled missing values
    
    Example:
        >>> df = handle_missing_values(
        ...     df,
        ...     strategy={'age': 'mean', 'country': 'most_frequent'}
        ... )
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
    
    This function handles:
    - Extracting basic date features (year, month, day, day of week)
    - Creating cyclical features for dates (month, day of week)
    - Creating lag features
    
    Args:
        data: Input DataFrame
        date_column: Name of the date column to extract features from
        cyclical_features: Whether to create cyclical features from dates
        lag_features: List of lag periods to create
    
    Returns:
        DataFrame with additional features
    
    Example:
        >>> df_features = create_features(
        ...     df,
        ...     date_column='date',
        ...     cyclical_features=True,
        ...     lag_features=[1, 7, 30]
        ... )
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
    
    This function handles:
    - Scaling numerical features using StandardScaler or MinMaxScaler
    
    Args:
        data: Input DataFrame
        columns: List of columns to scale (if None, scales all numeric columns)
        scaler_type: Type of scaling ('standard' or 'minmax')
    
    Returns:
        DataFrame with scaled features
    
    Example:
        >>> df_scaled = scale_features(
        ...     df,
        ...     columns=['price', 'quantity'],
        ...     scaler_type='standard'
        ... )
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
    
    This function handles:
    - One-hot encoding
    - Label encoding
    
    Args:
        data: Input DataFrame
        columns: List of categorical columns to encode
        method: Encoding method ('onehot' or 'label')
    
    Returns:
        DataFrame with encoded categorical variables
    
    Example:
        >>> df_encoded = encode_categorical(
        ...     df,
        ...     columns=['category'],
        ...     method='onehot'
        ... )
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
