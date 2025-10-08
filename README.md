# PS-IV-Mini-Project-Book-Recommendation-System

A Flask-based book recommendation web app that suggests similar books using a trained machine learning model.
Built for reproducibility with the **uv package manager** and **Python 3.13**.

---

## 🧩 Project Overview

This project provides:

* A **Flask frontend** for users to browse and search books.
* A **Python backend (Jupyter notebook)** for model training and data analysis.
* A **machine learning model** (KNN-based recommender) stored as a `.pkl` file and used in the Flask app for predictions.

---

## 🏗️ Folder Structure

```
PS-MINIPROJECT/
│
├── .venv/               # uv virtual environment (ignored in git)
├── app/                 # Flask frontend and API endpoints
├── data/                # Raw datasets (e.g., books.csv)
├── model/               # Trained model files (.pkl)
├── notebook/            # Jupyter notebooks for training and analysis
├── requirements.txt      # Optional fallback for pip installs
├── .gitignore
├── README.md
└── .lock                # uv dependency lock (keep in git for reproducibility)
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender
```

### 2️⃣ Create and sync environment (with `uv`)

Make sure you have **uv** installed:

```bash
pip install uv
```

Then sync the environment (creates `.venv/` automatically):

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate  # Linux / macOS
# or
.venv\Scripts\activate     # Windows
```

---

## 🧠 Model Training (Jupyter Notebook)

Train and store the recommendation model using your dataset.

1. Launch Jupyter Notebook:

   ```bash
   uv run jupyter notebook
   ```
2. Open the notebook inside `notebook/`
3. Run all cells — this will:

   * Preprocess the data
   * Train a KNN-based recommender
   * Save the model as `model/recommender.pkl`

---

## 🌐 Running the Flask App

1. Move to the `app` directory:

   ```bash
   cd app
   ```
2. Run the Flask server:

   ```bash
   uv run flask run
   ```
3. Open in browser:
   👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

You’ll see the **Book Store UI** where users can:

* Search for books
* View book details
* Get personalized recommendations

---

## 🧩 Technologies Used

* **Python 3.13**
* **Flask** (frontend + API)
* **scikit-learn** (KNN recommender)
* **pandas**, **numpy**, **matplotlib**, **seaborn**
* **uv** (modern dependency & environment manager)
* **Jupyter Notebook** (model training)

---

## 💾 Reproducibility

All dependencies are **locked in `.lock`** via `uv`.
To exactly reproduce the environment:

```bash
uv sync --frozen
```

---

## 📸 Future Improvements

* Add a collaborative filtering recommender
* Integrate user authentication (Flask-Login)
* Deploy on Render / Railway
* Add a modern UI using Bootstrap or Tailwind

