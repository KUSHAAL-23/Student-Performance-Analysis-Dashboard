import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Configuration and Data Loading ---

st.set_page_config(layout="wide", page_title="Student Performance Dashboard")
st.title("Student Performance Analysis Dashboard")

# The data loading uses caching for better performance.
# NOTE: This assumes the 'stud.csv' file is in the same directory as this script.
@st.cache_data
def load_data(file_path="stud.csv"):
    try:
        # Replaces the Colab-specific data loading
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()
    
    # Feature Engineering from the notebook
    data['Total_Score'] = data['math_score'] + data['reading_score'] + data['writing_score']
    data['average_score'] = data['Total_Score'] / 3
    return data

data = load_data()

# --- Model Training (Caching the model train/predict steps) ---

@st.cache_data
def train_and_evaluate_model(df):
    # Data Preparation as in the notebook
    df_model = df.drop(columns=['Total_Score', 'average_score'])
    df_model = pd.get_dummies(df_model, drop_first=True)
    
    # Target variable (Y) and features (X)
    X = df_model.drop(columns=['math_score']) # Predicting math_score
    Y = df_model['math_score']
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, Y_train)
    
    # Predictions
    Y_pred = lr_model.predict(X_test)
    
    # Evaluate Metrics
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame({'Actual Value': Y_test, 'Predicted Value': Y_pred.round(1)})
    pred_df['Difference'] = pred_df['Actual Value'] - pred_df['Predicted Value']
    
    return r2, mae, mse, rmse, pred_df, Y_test, Y_pred

r2, mae, mse, rmse, pred_df, Y_test, Y_pred = train_and_evaluate_model(data.copy())


# --- Dashboard Layout ---

# 1. Data Overview Section
st.header("1. Data Overview and Summary")
if st.checkbox('Show Raw Data and Summary', True):
    st.subheader("Raw Data (First 5 Rows)")
    st.dataframe(data.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Shape")
        st.write(f"The dataset has **{data.shape[0]} rows** and **{data.shape[1]} columns**.")
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe().T)
    with col2:
        st.subheader("Check for Missing Values and Duplicates")
        st.text(f"Missing Values:\n{data.isnull().sum().to_string()}")
        st.text(f"Duplicated Rows: {data.duplicated().sum()}")


# 2. Exploratory Data Analysis (EDA) Section
st.header("2. Exploratory Data Analysis (EDA)")
st.write("Visualizing the distribution of categorical and numerical features.")

# Categorical Feature Counts (Univariate Analysis)
with st.expander("Categorical Feature Distribution (Count Plots)"):
    
    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    
    # Sidebar selection for the count plot
    selected_cat_col = st.selectbox("Select a Categorical Column for Analysis:", categorical_cols, key='cat_select')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=selected_cat_col, data=data, palette='viridis', ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Count Plot of {selected_cat_col}')
    plt.ylabel('Count')
    st.pyplot(fig)


# Score Distributions and Correlation (Bivariate Analysis)
with st.expander("Score Distribution and Correlation"):
    col3, col4 = st.columns(2)
    
    # Score Distribution
    with col3:
        st.subheader("Score Distribution (Math, Reading, Writing)")
        # Plotting the histograms/KDEs from the notebook
        fig_dist, ax_dist = plt.subplots(3, 1, figsize=(10, 15))
        score_cols = ['math_score', 'reading_score', 'writing_score']
        
        for i, col in enumerate(score_cols):
            sns.histplot(data[col], kde=True, ax=ax_dist[i], bins=20, color=sns.color_palette("Set2")[i])
            ax_dist[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig_dist)

    # Correlation Heatmap
    with col4:
        st.subheader("Correlation Heatmap of Scores")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr_matrix = data[score_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        st.pyplot(fig_corr)

# Box Plots (Categorical vs Numerical)
with st.expander("Categorical vs. Score (Box Plots)"):
    col5, col6 = st.columns(2)

    with col5:
        # Sidebar selection for box plots (Categorical)
        selected_box_cat = st.selectbox("Select Categorical Feature:", categorical_cols, key='box_cat')
    with col6:
        # Sidebar selection for box plots (Numerical/Score)
        selected_box_num = st.selectbox("Select Score Feature:", score_cols + ['average_score'], key='box_num')

    fig_box, ax_box = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=selected_box_cat, y=selected_box_num, data=data, palette='Pastel1', ax=ax_box)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{selected_box_num} by {selected_box_cat}')
    st.pyplot(fig_box)

# 3. Model Evaluation Section
st.header("3. Linear Regression Model Evaluation")
st.markdown("Model trained to predict **Math Score** based on other features.")

col7, col8, col9, col10 = st.columns(4)

with col7:
    st.metric(label="R-Squared Score", value=f"{r2:.4f}")
with col8:
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
with col9:
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
with col10:
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")

# Actual vs Predicted Plot
st.subheader("Actual vs. Predicted Math Scores")

fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
# Scatter plot of actual vs predicted
ax_pred.scatter(Y_test, Y_pred)
# Plot the ideal line (y=x)
ax_pred.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', label='Ideal Prediction')
ax_pred.set_xlabel("Actual Math Score")
ax_pred.set_ylabel("Predicted Math Score")
ax_pred.set_title("Actual vs. Predicted Math Scores")
ax_pred.legend()
st.pyplot(fig_pred)

# Prediction Comparison Table
st.subheader("Prediction Comparison Table (Test Set)")
st.dataframe(pred_df)