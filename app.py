
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.base import clone

app = Flask(__name__)

# Choices displayed on the website
job_titles = [
    "Data Analyst",
    "Data Scientist",
    "ML Engineer",
    "Software Engineer",
    "Business Analyst",
    "AI Engineer"
]

education_levels = ["Bachelor's", "Master's", "PhD"]


# Generate a reproducible demo dataset
def create_dataset():
    rng = np.random.default_rng(42)
    rows = 1200

    ages = rng.integers(21, 61, rows)
    experience = np.array([
        rng.integers(0, max(1, min(int(age) - 20, 36)))
        for age in ages
    ])

    education = rng.choice(
        education_levels,
        size=rows,
        p=[0.55, 0.35, 0.10]
    )
    jobs = rng.choice(job_titles, size=rows)

    education_bonus = {
        "Bachelor's": 0,
        "Master's": 180000,
        "PhD": 350000
    }

    job_bonus = {
        "Data Analyst": 100000,
        "Data Scientist": 350000,
        "ML Engineer": 400000,
        "Software Engineer": 250000,
        "Business Analyst": 150000,
        "AI Engineer": 420000
    }

    salaries = []

    for age, exp, edu, job in zip(
        ages, experience, education, jobs
    ):
        salary = (
            300000
            + int(exp) * 65000
            + education_bonus[edu]
            + job_bonus[job]
            + max(0, int(age) - 22) * 5000
            + rng.normal(0, 90000)
        )
        salaries.append(max(250000, salary))

    return pd.DataFrame({
        "Age": ages,
        "Experience": experience,
        "Education": education,
        "Job": jobs,
        "Salary": np.round(salaries).astype(int)
    })


df = create_dataset()

feature_columns = ["Age", "Experience", "Education", "Job"]
X = df[feature_columns]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                ["Education", "Job"]
            )
        ],
        remainder="passthrough"
    )


# Main salary prediction model
model = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Evaluate the main model on unseen test data
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


# Compare multiple regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

model_comparison = []

for model_name, regressor in regression_models.items():
    evaluation_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", clone(regressor))
    ])

    evaluation_pipeline.fit(X_train, y_train)
    evaluation_predictions = evaluation_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, evaluation_predictions)

    model_comparison.append({
        "name": model_name,
        "r2": round(float(
            r2_score(y_test, evaluation_predictions)
        ), 3),
        "mae": round(float(
            mean_absolute_error(y_test, evaluation_predictions)
        ), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(np.sqrt(mse)), 2)
    })


@app.route("/")
def home():
    job_averages = [
        int(round(df.loc[df["Job"] == job, "Salary"].mean()))
        for job in job_titles
    ]

    salary_data = {
        "jobLabels": job_titles,
        "jobAverages": job_averages,
        "salaries": [
            int(value) for value in df["Salary"].iloc[:100]
        ],
        "actualPredicted": [
            {
                "x": int(actual),
                "y": int(round(float(predicted)))
            }
            for actual, predicted in zip(
                y_test.iloc[:50], predictions[:50]
            )
        ],
        "residuals": [
            int(round(float(actual - predicted)))
            for actual, predicted in zip(
                y_test.iloc[:50], predictions[:50]
            )
        ]
    }

    return render_template(
        "index.html",
        employee_count=len(df),
        avg_salary=int(df["Salary"].mean()),
        mae=int(mae),
        r2=round(float(r2), 3),
        jobs=job_titles,
        education_levels=education_levels,
        salary_dashboard_data=salary_data,
        model_comparison=model_comparison
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}

        age = int(data.get("age", 0))
        experience = int(data.get("experience", -1))
        education = data.get("education", "")
        job = data.get("job", "")

        if not 18 <= age <= 65:
            return jsonify({
                "error": "Age must be between 18 and 65."
            }), 400

        if not 0 <= experience <= age - 18:
            return jsonify({
                "error": "Please enter valid work experience."
            }), 400

        if education not in education_levels:
            return jsonify({
                "error": "Please select a valid education level."
            }), 400

        if job not in job_titles:
            return jsonify({
                "error": "Please select a valid job role."
            }), 400

        employee = pd.DataFrame([{
            "Age": age,
            "Experience": experience,
            "Education": education,
            "Job": job
        }])

        salary = float(model.predict(employee)[0])

        return jsonify({
            "predicted_salary": round(salary),
            "model": "Random Forest Regressor"
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Please enter valid information."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
