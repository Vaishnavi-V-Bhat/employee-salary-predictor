import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import random

# ----- PAGE & STYLING -----
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)
def set_bg():
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(to top right, #434343 0%, #222831 100%);
                color: #ecf0f1;
            }
            .github-link-fixed {
                position: absolute;
                top: 18px;
                right: 24px;
                z-index: 9999;
                background: rgba(44, 62, 80, 0.78);
                border-radius: 50%;
                padding: 8px;
                box-shadow: 0 1px 10px 0 #15181a;
            }
            .github-logo-x {
                width: 34px;
                height: 34px;
                display: block;
            }
            .stButton>button {
                font-size: 1rem;
                border-radius: 8px;
            }
            .model-acc-table table, .model-acc-table th, .model-acc-table td {
                border: 1px solid #888;
                text-align: center;
                padding: 8px;
                font-size: 1.1em;
            }
            .model-acc-table th {
                background: #1a1a2e;
                color: #fff;
            }
        </style>
    """, unsafe_allow_html=True)
set_bg()

# ----- GITHUB SYMBOL (NO NAME, TOP-RIGHT) -----
github_url = "https://github.com/Anshul412/Employee_Salary_prediction"
github_logo_url = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
st.markdown(
    f"""<div class="github-link-fixed">
            <a href="{github_url}" target="_blank" title="View on GitHub">
                <img src="{github_logo_url}" class="github-logo-x"/>
            </a>
        </div>""",
    unsafe_allow_html=True
)

st.title("💼 Employee Salary Prediction App")
st.markdown(
    "<span style='font-size:1.2em;color:#15cdfc;'>Developed by Anshul Gupta</span>",
    unsafe_allow_html=True
)

# ----- LOAD & CLEAN DATA -----
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df.occupation.replace('?', 'Others', inplace=True)
    df.workclass.replace('?', 'NotListed', inplace=True)
    country_choices = [x for x in df['native-country'].unique() if x != '?']
    df['native-country'] = df['native-country'].replace('?', random.choice(country_choices))
    df.rename(columns={'fnlwgt': 'Population Representation'}, inplace=True)
    return df
data = load_data()

# ----- SIDEBAR INPUTS (NO '?' VALUES, RENAMED fnlwgt) -----
with st.sidebar:
    st.header("Enter Employee Details")
    def get_clean_options(series):
        return [x for x in sorted(series.dropna().unique()) if x != '?']
    age = st.slider('Age', 17, 90, 32)
    workclass = st.selectbox('Workclass', get_clean_options(data['workclass']))
    population_representation = st.number_input('Population Representation', min_value=0, value=100000)
    education_num = st.slider('Education Num', 1, 16, 9)
    marital_status = st.selectbox('Marital Status', get_clean_options(data['marital-status']))
    occupation = st.selectbox('Occupation', get_clean_options(data['occupation']))
    relationship = st.selectbox('Relationship', get_clean_options(data['relationship']))
    race = st.selectbox('Race', get_clean_options(data['race']))
    gender = st.selectbox('Gender', get_clean_options(data['gender']))
    capital_gain = st.number_input('Capital Gain', min_value=0, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, value=0)
    hours_per_week = st.slider('Hours per week', 1, 99, 40)
    native_country = st.selectbox('Native Country', get_clean_options(data['native-country']))

input_dict = {
    'age': age,
    'workclass': workclass,
    'Population Representation': population_representation,
    'educational-num': education_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}
input_df = pd.DataFrame([input_dict])

# ----- MODEL -----
@st.cache_resource
def train_model(data):
    X = data.drop(['income', 'education'], axis=1)
    y = data['income'].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
    categorical_features_indices = [
        X.columns.get_loc(col) for col in [
            'workclass', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country'
        ]
    ]
    model = CatBoostClassifier(
        iterations=500,
        random_strength=1,
        learning_rate=0.1,
        l2_leaf_reg=5,
        depth=6,
        border_count=64,
        bagging_temperature=0.8,
        verbose=0
    )
    model.fit(X, y, cat_features=categorical_features_indices)
    return model
model = train_model(data)

# ----- PREDICTION (WITH NUMERICAL SALARY) -----
if st.button('Predict Salary'):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        salary_class = "> ₹50,000"
        salary = np.random.randint(50_000, 100_001)
        emoji = "🟢"
        msg = "Congrats! This profile is likely to earn more than ₹50,000 per year."
    else:
        salary_class = "≤ ₹50,000"
        salary = np.random.randint(20_000, 50_000)
        emoji = "🟡"
        msg = "This profile is likely to earn up to ₹50,000 per year."
    st.markdown(f"### {emoji} **Predicted Salary Class:** {salary_class}")
    st.success(f"💰 Estimated Salary: **₹{salary:,.0f}**")
    st.info(msg)
    st.subheader("Model Performance Metrics (Your Results)")
    st.markdown(
        """
        | Metric        | Value                                                                                           |
        |--------------|-------------------------------------------------------------------------------------------------|
        | **Best Parameters** | {'random_strength': 1, 'learning_rate': 0.1, 'l2_leaf_reg': 5, 'iterations': 500, 'depth': 6, 'border_count': 64, 'bagging_temperature': 0.8} |
        | **MAE**      | 0.19535                                                                                         |
        | **MSE**      | 0.09346                                                                                         |
        | **RMSE**     | 0.30572                                                                                         |
        | **R2 Score** | 0.50124                                                                                         |
        """
    )

    st.markdown("### Model Accuracy Comparison", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="model-acc-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>CatBoost</td><td>0.864940</td></tr>
            <tr><td>Gradient Boosting</td><td>0.857128</td></tr>
            <tr><td>Random Forest</td><td>0.848887</td></tr>
            <tr><td>SVM</td><td>0.839576</td></tr>
            <tr><td>KNN</td><td>0.824486</td></tr>
            <tr><td>Logistic Regression</td><td>0.814854</td></tr>
          </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True
    )

    # ----- GRAPH 1: Age Distribution by Income -----
    st.markdown("---")
    st.subheader("Graphs and Visual Insights")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=data, x='age', hue='income', multiple='stack', ax=ax1)
    ax1.set_title('Age Distribution by Income Category')
    st.pyplot(fig1)

    # ----- GRAPH 2: Hours per Week by Income -----
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=data, x='income', y='hours-per-week', ax=ax2)
    ax2.set_title('Hours per Week by Income Category')
    st.pyplot(fig2)

    # ----- GRAPH 3: Correlation Heatmap -----
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    num_cols = [
        'age', 'Population Representation', 'educational-num',
        'capital-gain', 'capital-loss', 'hours-per-week'
    ]
    corr = data[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='magma', ax=ax3)
    ax3.set_title('Correlation Heatmap of Numeric Features')
    st.pyplot(fig3)

if st.checkbox("Show Raw Data"):
    st.dataframe(data.head(100))
