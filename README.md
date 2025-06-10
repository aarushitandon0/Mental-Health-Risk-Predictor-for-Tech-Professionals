# Mental Health Risk Predictor for Tech Professionals

A Streamlit-based web app that predicts the likelihood of someone needing mental health support based on their workplace environment, personal experiences, and demographics — trained on the OSMI Mental Health in Tech dataset.

## Demo

Try the live app here: [https://your-username-mindscope.streamlit.app](https://your-username-mindscope.streamlit.app)

## Features

- Simple UI built with Streamlit
- One-hot encoded survey inputs from user
- Neural Network model (trained using TensorFlow/Scikit-learn)
- Predicts risk level for needing mental health treatment
- 25+ factors from workplace culture, country, benefits, history, etc.

## 📊 Dataset

- Dataset: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
- Preprocessed: cleaned missing values, one-hot encoded categorical features
- Model trained on 112 features

##  Files in this repo

| File | Description |
|------|-------------|
| `app.py` | Streamlit app code |
| `mental_health_nn_model.pkl` | Trained neural network model |
| `model_columns.pkl` | List of one-hot encoded column names (used to align user input) |
| `survey_template.csv` | One-row template with all encoded columns |
| `MentalHealthRiskPredictor.ipynb` | Model training |

## Dataset & Feature Description

**Source**: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)

We used 112 one-hot encoded features derived from the following columns:

| Column | Description |
|--------|-------------|
| `Age` | Respondent's age |
| `Gender` | Self-identified gender (Male, Female, Other) |
| `Country` | Country of residence |
| `state` | U.S. state (if applicable) |
| `self_employed` | Whether the respondent is self-employed |
| `family_history` | Family history of mental illness |
| `treatment` | Has the respondent sought mental health treatment before |
| `work_interfere` | Frequency of mental health interference with work |
| `no_employees` | Number of employees at the company |
| `remote_work` | Whether they work remotely |
| `tech_company` | Whether the company is a tech company |
| `benefits` | Availability of mental health benefits from employer |
| `care_options` | Awareness of mental health care options provided |
| `wellness_program` | Inclusion of mental health in wellness programs |
| `seek_help` | Availability of resources for seeking help |
| `anonymity` | Anonymity protection regarding mental health info |
| `leave` | Ease of taking medical leave for mental health reasons |
| `mental_health_consequence` | Perceived consequences for discussing mental health |
| `phys_health_consequence` | Perceived consequences for physical health discussion |
| `coworkers` | Comfort level discussing mental health with coworkers |
| `supervisor` | Comfort level discussing with supervisor |
| `mental_health_interview` | Willingness to bring up mental health in interviews |
| `phys_health_interview` | Willingness to bring up physical health in interviews |
| `mental_vs_physical` | Whether employer treats mental and physical health equally |
| `obs_consequence` | Observed consequences of others discussing mental health |
| `comments` | Free-form comment field (excluded from prediction) |

> After preprocessing, all categorical variables were **one-hot encoded**, resulting in a feature vector of length 112.

---

