"""
Helper functions for data visualization in BendTheCurve blog posts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

def set_style(style: str = "whitegrid") -> None:
    """
    Set the default style for all plots.
    
    Args:
        style: The seaborn style to use. Default is "whitegrid".
    """
    sns.set_style(style)
    plt.style.use('seaborn')

def plot_distribution(data: Union[pd.Series, np.ndarray],
                     title: str = "",
                     xlabel: str = "",
                     ylabel: str = "Count",
                     figsize: Tuple[int, int] = (10, 6),
                     bins: int = 30,
                     kde: bool = True) -> plt.Figure:
    """
    Plot the distribution of a numerical variable.
    
    Args:
        data: The data to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        bins: Number of histogram bins
        kde: Whether to show the kernel density estimate
    
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=data, bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt.gcf()

def correlation_heatmap(data: pd.DataFrame,
                       figsize: Tuple[int, int] = (10, 8),
                       cmap: str = "coolwarm",
                       annot: bool = True) -> plt.Figure:
    """
    Create a correlation heatmap for numerical columns in a DataFrame.
    
    Args:
        data: pandas DataFrame containing numerical columns
        figsize: Figure size as (width, height)
        cmap: Color map for the heatmap
        annot: Whether to annotate cells with numerical value
    
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    corr = data.corr()
    sns.heatmap(corr, cmap=cmap, annot=annot, center=0)
    plt.title("Correlation Heatmap")
    return plt.gcf()

def scatter_plot(x: Union[pd.Series, np.ndarray],
                y: Union[pd.Series, np.ndarray],
                title: str = "",
                xlabel: str = "",
                ylabel: str = "",
                figsize: Tuple[int, int] = (10, 6),
                add_trend: bool = True) -> plt.Figure:
    """
    Create a scatter plot with optional trend line.
    
    Args:
        x: Data for x-axis
        y: Data for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        add_trend: Whether to add a trend line
    
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y)
    
    if add_trend:
        sns.regplot(x=x, y=y, scatter=False, color='red')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt.gcf()

def time_series_plot(data: pd.Series,
                    title: str = "",
                    xlabel: str = "Date",
                    ylabel: str = "Value",
                    figsize: Tuple[int, int] = (12, 6),
                    rolling_window: Optional[int] = None) -> plt.Figure:
    """
    Create a time series plot with optional rolling average.
    
    Args:
        data: Time series data (pandas Series with datetime index)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        rolling_window: Size of rolling window for moving average
    
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    plt.plot(data.index, data.values, label='Original')
    
    if rolling_window:
        rolling_mean = data.rolling(window=rolling_window).mean()
        plt.plot(data.index, rolling_mean, 
                label=f'{rolling_window}-period Moving Average',
                color='red')
        plt.legend()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    return plt.gcf()

def interactive_scatter(data: pd.DataFrame,
                       x: str,
                       y: str,
                       color: Optional[str] = None,
                       size: Optional[str] = None,
                       title: str = "") -> go.Figure:
    """
    Create an interactive scatter plot using plotly.
    
    Args:
        data: pandas DataFrame
        x: Column name for x-axis
        y: Column name for y-axis
        color: Column name for color coding points
        size: Column name for sizing points
        title: Plot title
    
    Returns:
        plotly Figure object
    """
    fig = px.scatter(data, x=x, y=y, color=color, size=size,
                    title=title, template="simple_white")
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    return fig
