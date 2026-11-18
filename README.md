# AI-Powered Student Score Predictor

> **Superseded by [Student-Performance-Prediction](https://github.com/sahithi-manda/Student-Performance-Prediction)**, which covers the same problem with more features, a simple-vs-multiple regression comparison and feature-importance analysis. This repo is kept for history.

Predicts a student's exam score from study hours and sleep hours with linear regression.

## Data and results

`student_data.csv` has 300 synthetic students with `Study_Hours`, `Sleep_Hours` and `Score`. Verified by running the script:

| Metric | Value |
|---|---|
| R² | 0.90 |
| RMSE | 5.59 points |

Study hours dominate: each extra hour adds about 5.75 points, holding sleep constant.

## Run it

```bash
pip install -r requirements.txt
python predict_scores.py
```

Plots (EDA, actual vs predicted, residuals) are saved to `visualizations/`.
