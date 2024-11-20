# BendTheCurve Utilities

This directory contains helper functions for data analysis and visualization in your blog posts.

## Modules

### 1. `plotting.py`
Contains functions for creating various types of plots:
- Distribution plots
- Correlation heatmaps
- Scatter plots with trend lines
- Time series plots
- Interactive plots using Plotly

### 2. `data_prep.py`
Contains functions for data preparation and preprocessing:
- Loading and cleaning data
- Handling missing values
- Feature engineering
- Scaling features
- Encoding categorical variables

## Usage Example

```python
from utils.plotting import plot_distribution, correlation_heatmap
from utils.data_prep import load_and_clean_data, handle_missing_values

# Load and prepare data
df = load_and_clean_data('data.csv', 
                        date_columns=['date'],
                        numerical_columns=['value'])

# Clean missing values
df_clean = handle_missing_values(df, 
                               strategy={'value': 'mean'})

# Create a distribution plot
fig = plot_distribution(df_clean['value'],
                       title='Distribution of Values',
                       xlabel='Value')
```

## Dependencies
Make sure you have the following packages installed:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

You can install them using:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```
