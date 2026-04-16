# 🚢 Titanic Survival Predictor

A **Flask web application** that predicts whether a Titanic passenger would survive or not, using pre-trained **Machine Learning models** built with scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Screenshots

| Input Form | Prediction Result |
|:---:|:---:|
| ![Form](screenshots/form.png) | ![Result](screenshots/result.png) |

---

## ✨ Features

- **3 ML Models** — Choose between Logistic Regression, Decision Tree, or Random Forest
- **Full sklearn Pipeline** — Each model includes imputation, one-hot encoding, scaling, and classification
- **Probability Score** — Displays survival probability (%) with an animated progress bar
- **Input Validation** — Graceful error handling for invalid or missing inputs
- **Modern Dark UI** — Glassmorphism card, gradient header, smooth animations, fully responsive
- **Form Persistence** — Input values are retained after prediction

---

## 🗂️ Project Structure

```
titanic-survival-predictor/
├── app.py                  # Flask backend (routes, model loading, prediction logic)
├── templates/
│   └── index.html          # Jinja2 template with embedded CSS
├── model_log_reg.pkl       # Trained Logistic Regression pipeline
├── model_tree.pkl          # Trained Decision Tree pipeline
├── model_forest.pkl        # Trained Random Forest pipeline
├── ML_1.ipynb              # Jupyter notebook (model training & EDA)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<pModi925>/titanic-survival-predictor.git
   cd titanic-survival-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| **Pclass** | int (1, 2, 3) | Passenger class |
| **Sex** | categorical | male / female |
| **Age** | float | Age in years |
| **SibSp** | int | Number of siblings/spouses aboard |
| **Parch** | int | Number of parents/children aboard |
| **Fare** | float | Ticket fare paid (£) |
| **Embarked** | categorical | Port of embarkation — C (Cherbourg), Q (Queenstown), S (Southampton) |

---

## ⚙️ How It Works

1. **Model Loading** — At startup, three pre-trained sklearn `Pipeline` objects are loaded from `.pkl` files.
2. **User Input** — The user fills in passenger details and selects a model via the web form.
3. **Preprocessing** — The pipeline automatically handles:
   - Missing value imputation (mean for Age, most frequent for Embarked)
   - One-hot encoding (Sex, Embarked)
   - Standard scaling (all features)
4. **Prediction** — The selected model outputs:
   - **Class** — Survived (1) or Did Not Survive (0)
   - **Probability** — via `predict_proba()` (percentage of survival)

---

## 📊 Models & Probability Logic

| Model | How Probability Is Computed |
|-------|----------------------------|
| **Logistic Regression** | Sigmoid function: `P = 1 / (1 + e^(-z))` — a true probability |
| **Decision Tree** | Fraction of training samples in the leaf node that survived |
| **Random Forest** | Average vote across all trees in the ensemble |

---

## 🌐 Deployment (Production)

For production deployment (Linux), use **Gunicorn**:

```bash
gunicorn app:app --bind 0.0.0.0:8000
```

> **Note:** Gunicorn does not run on Windows. Use `python app.py` for local development on Windows.

---

## 🛠️ Tech Stack

- **Backend** — Flask (Python)
- **ML** — scikit-learn (Pipeline, LogisticRegression, DecisionTree, RandomForest)
- **Data** — pandas, NumPy
- **Frontend** — HTML5, CSS3 (custom dark theme, no external CSS frameworks)
- **Server** — Gunicorn (production)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ using Flask & scikit-learn
</p>
