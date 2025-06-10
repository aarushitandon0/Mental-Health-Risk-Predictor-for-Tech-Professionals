# Mental Health Risk Predictor for Tech Professionals

A Streamlit-based web app that predicts the likelihood of someone needing mental health support based on their workplace environment, personal experiences, and demographics — trained on the OSMI Mental Health in Tech dataset.

## Demo

🚀 Try the live app here: [https://your-username-mindscope.streamlit.app](https://your-username-mindscope.streamlit.app)

## 📌 Features

- Simple UI built with Streamlit
- One-hot encoded survey inputs from user
- Neural Network model (trained using TensorFlow/Scikit-learn)
- Predicts risk level for needing mental health treatment
- 25+ factors from workplace culture, country, benefits, history, etc.

## 📊 Dataset

- Dataset: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
- Preprocessed: cleaned missing values, one-hot encoded categorical features
- Model trained on 112 features

## 📁 Files in this repo

| File | Description |
|------|-------------|
| `app.py` | Streamlit app code |
| `mental_health_nn_model.pkl` | Trained neural network model |
| `model_columns.pkl` | List of one-hot encoded column names (used to align user input) |
| `survey_template.csv` | One-row template with all encoded columns |
| `MentalHealthRiskPredictor.ipynb` | Model training |


