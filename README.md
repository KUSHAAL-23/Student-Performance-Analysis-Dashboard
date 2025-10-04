# Student-Performance-Analysis-Dashboard

This project is a Streamlit Dashboard designed for Student Performance Analysis.

1. Objective
The primary goal is to provide an interactive web application to analyze student performance data, visualize key distributions and correlations, and evaluate a machine learning model.

2. Key Features and Sections
The dashboard is structured into three main sections:

Data Overview and Summary: Displays the raw dataset (first 5 rows), its shape, descriptive statistics, and checks for missing values and duplicates.

Exploratory Data Analysis (EDA): Includes interactive visualizations for in-depth data exploration.

Univariate Analysis: Count plots for selected categorical features (e.g., gender, race/ethnicity).

Bivariate Analysis: Histograms/KDE plots for the distribution of scores (Math, Reading, Writing), a correlation heatmap of the scores, and interactive box plots to compare scores against categorical features.

Linear Regression Model Evaluation: Focuses on the performance of a machine learning model trained to predict Math Score.

Displays key regression metrics: R-Squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

Presents a scatter plot comparing Actual vs. Predicted Math Scores.

Shows a prediction comparison table for the test set.

3. Technology Stack
The application uses Python and several popular libraries:

Streamlit: For building and deploying the interactive web dashboard.

Pandas & NumPy: For data manipulation, loading (stud.csv), and numerical processing.

Matplotlib & Seaborn: For all data visualization and plotting.

Scikit-learn: For machine learning tasks, including data splitting (train_test_split), model training (LinearRegression), and evaluation metrics.
