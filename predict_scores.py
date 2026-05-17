import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def main():
    print("=== Step 1: Load dataset (Synthetic) ===")
    np.random.seed(42)
    n_samples = 300
    
    # Generate synthetic data
    study_hours = np.random.uniform(1, 10, n_samples)
    sleep_hours = np.random.uniform(4, 10, n_samples)
    
    # Score depends on study hours, sleep hours, and some noise
    # Base score of 20, + 6 per study hour, + 3 per sleep hour
    noise = np.random.normal(0, 5, n_samples)
    scores = 20 + 6 * study_hours + 3 * sleep_hours + noise
    
    # Clip scores to be within 0-100
    scores = np.clip(scores, 0, 100)
    
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Sleep_Hours': sleep_hours,
        'Score': scores
    })
    
    # Save to CSV as an artifact for the user
    df.to_csv('student_data.csv', index=False)
    print("Dataset generated and saved to 'student_data.csv'.")
    print("Dataset head:")
    print(df.head())
    print()

    print("=== Step 2: Perform EDA ===")
    os.makedirs('visualizations', exist_ok=True)
    
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    print()
    
    # 1. Pairplot
    plt.figure(figsize=(8, 6))
    sns.pairplot(df)
    plt.suptitle("Pairplot of Study Hours, Sleep Hours, and Score", y=1.02)
    plt.savefig('visualizations/eda_pairplot.png')
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(6, 5))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('visualizations/eda_correlation.png')
    plt.close()
    
    print("EDA Visualizations saved to 'visualizations/' directory.")
    print()

    print("=== Step 3: Train a multiple linear regression model ===")
    X = df[['Study_Hours', 'Sleep_Hours']]
    y = df['Score']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    print()

    print("=== Step 4: Evaluate and visualize results ===")
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    # Visualizing Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs Predicted Scores')
    plt.tight_layout()
    plt.savefig('visualizations/actual_vs_predicted.png')
    plt.close()
    
    # Visualizing Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=20)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('visualizations/residuals_distribution.png')
    plt.close()
    
    print("Evaluation visualizations saved to 'visualizations/' directory.")
    print()

    print("=== Step 5: Interpret feature importance ===")
    # Feature importance based on coefficients
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    
    print("Model Coefficients:")
    print(importance_df.to_string(index=False))
    print(f"Intercept: {model.intercept_:.2f}")
    print("\nInterpretation:")
    
    study_coef = model.coef_[0]
    sleep_coef = model.coef_[1]
    
    if study_coef > sleep_coef:
        print(f"- Study Hours is the most important feature (Coefficient: {study_coef:.2f}).")
    else:
        print(f"- Sleep Hours is the most important feature (Coefficient: {sleep_coef:.2f}).")
        
    print(f"- For every 1 hour increase in Study Hours, the score increases by approx {study_coef:.2f} points (holding sleep constant).")
    print(f"- For every 1 hour increase in Sleep Hours, the score increases by approx {sleep_coef:.2f} points (holding study constant).")

if __name__ == "__main__":
    main()
