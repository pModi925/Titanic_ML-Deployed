"""
Titanic Survival Prediction — Flask Web Application
====================================================
Loads three pre-trained sklearn pipeline models (Logistic Regression,
Decision Tree, Random Forest) at startup and serves a form where users
can enter passenger details and receive a survival prediction along
with a probability score.
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ──────────────────────────── App Setup ────────────────────────────
app = Flask(__name__)

# ──────────────────────────── Load Models ──────────────────────────
# All three models are full sklearn Pipeline objects that handle
# imputation → one-hot encoding → scaling → classification internally.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Logistic Regression": os.path.join(BASE_DIR, "model_log_reg.pkl"),
    "Decision Tree": os.path.join(BASE_DIR, "model_tree.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "model_forest.pkl"),
}

models = {}
for name, path in MODEL_PATHS.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)
    print(f"✔ Loaded model: {name}")


# ──────────────────────────── Helper ───────────────────────────────
def build_input_df(form: dict) -> pd.DataFrame:
    """
    Parse and validate form data, returning a one-row DataFrame whose
    columns match what the pipeline expects:
        Pclass | Sex | Age | SibSp | Parch | Fare | Embarked
    Raises ValueError with a user-friendly message on bad input.
    """
    try:
        pclass = int(form["pclass"])
        if pclass not in (1, 2, 3):
            raise ValueError("Pclass must be 1, 2, or 3.")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid Pclass: {e}")

    sex = form.get("sex", "").strip().lower()
    if sex not in ("male", "female"):
        raise ValueError("Sex must be 'male' or 'female'.")

    try:
        age = float(form["age"])
        if age < 0 or age > 120:
            raise ValueError("Age must be between 0 and 120.")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid Age: {e}")

    try:
        sibsp = int(form["sibsp"])
        if sibsp < 0:
            raise ValueError("SibSp cannot be negative.")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid SibSp: {e}")

    try:
        parch = int(form["parch"])
        if parch < 0:
            raise ValueError("Parch cannot be negative.")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid Parch: {e}")

    try:
        fare = float(form["fare"])
        if fare < 0:
            raise ValueError("Fare cannot be negative.")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid Fare: {e}")

    embarked = form.get("embarked", "").strip().upper()
    if embarked not in ("C", "Q", "S"):
        raise ValueError("Embarked must be C, Q, or S.")

    # Build a DataFrame that mirrors the training data's column order
    data = pd.DataFrame(
        [[pclass, sex, age, sibsp, parch, fare, embarked]],
        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    )
    return data


# ──────────────────────────── Routes ───────────────────────────────
@app.route("/")
def home():
    """Render the input form page."""
    return render_template("index.html", model_names=list(models.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle the prediction POST request:
      1. Validate & parse input
      2. Select the chosen model
      3. Run prediction + probability
      4. Return result to the template
    """
    try:
        # --- Parse user input ---
        input_df = build_input_df(request.form)

        # --- Select model ---
        model_name = request.form.get("model", "Logistic Regression")
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        model = models[model_name]

        # --- Predict ---
        prediction = model.predict(input_df)[0]
        result = "🎉 Survived" if prediction == 1 else "💀 Did Not Survive"

        # --- Probability (if available) ---
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            # proba[1] = probability of survival
            probability = round(float(proba[1]) * 100, 2)

        return render_template(
            "index.html",
            model_names=list(models.keys()),
            result=result,
            probability=probability,
            survived=prediction == 1,
            selected_model=model_name,
            # Echo back old values so the form stays filled
            form=request.form,
        )

    except ValueError as ve:
        return render_template(
            "index.html",
            model_names=list(models.keys()),
            error=str(ve),
            form=request.form,
        )
    except Exception as ex:
        return render_template(
            "index.html",
            model_names=list(models.keys()),
            error=f"Unexpected error: {ex}",
            form=request.form,
        )


# ──────────────────────────── Run ──────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
