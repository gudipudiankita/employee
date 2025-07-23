import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Employee Salary Dashboard", layout="wide")

# ---- SESSION STATE ----
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "chart" not in st.session_state:
    st.session_state.chart = "Income"

# ---- LOAD + CLEAN DATA ----
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("P:/projects/app/adult 3.csv")
    df.columns = [col.strip().lower().replace("-", "_") for col in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(df):
    encoders = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le
    return df_enc, encoders

df = load_and_clean_data()
df_encoded, encoders = preprocess(df)

# ---- METRICS ----
def get_metrics(df):
    total = len(df)
    high_income = len(df[df['income'] == '>50K'])
    low_income = len(df[df['income'] == '<=50K'])
    avg_age = int(df['age'].mean())
    avg_hours = round(df['hours_per_week'].mean(), 1)
    return total, high_income, low_income, avg_age, avg_hours

# ---- SIDEBAR ----
with st.sidebar:
    st.title("📂 Navigation")
    st.session_state.page = st.radio("Select Page", ["Dashboard", "Predict"])
    if st.session_state.page == "Dashboard":
        st.markdown("### 📈 Select Chart")
        st.session_state.chart = st.radio("Chart Type", ["Income", "Education", "Age Distribution"])

    with st.expander("📊  Summary Metrics", expanded=False):
        total, high, low, avg_age, avg_hours = get_metrics(df)
        st.metric("Total Employees", total)
        st.metric("Income >50K", high)
        st.metric("Income <=50K", low)
        st.metric("Average Age", avg_age)
        st.metric("Avg Hours/Week", avg_hours)

# ---- DASHBOARD ----
if st.session_state.page == "Dashboard":
    st.title("📊 Employee Salary Dashboard")

    st.write("Use the sidebar to explore different visualizations of employee data related to income, education, and age.")

    if st.session_state.chart == "Income":
        st.subheader("📌 Income Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="income", data=df, palette="Set2", ax=ax)
        ax.set_title("Income Category Distribution")
        st.pyplot(fig)

    elif st.session_state.chart == "Education":
        st.subheader("🎓 Education Category Insights")

        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.countplot(x="education", hue="income", data=df, palette="Set2", ax=ax1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_title("Education vs Income")
        st.pyplot(fig1)

    elif st.session_state.chart == "Age Distribution":
        st.subheader("📈 Age Distribution Insights")

        
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.histplot(df['age'], bins=20, kde=True, ax=ax4, color='orange')
        ax4.set_title("Age Histogram with KDE")
        st.pyplot(fig4)

# ---- PREDICTION ----
elif st.session_state.page == "Predict":
    st.title("🧠 Predict Employee Income Category")

    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]
    model = RandomForestClassifier()
    model.fit(X, y)

    st.markdown("Please enter employee details to estimate income category:")

    user_input = {}
    for col in X.columns:
        label = col.replace("_", " ").capitalize()
        if col in encoders:
            user_input[col] = st.selectbox(label, encoders[col].classes_)
        else:
            user_input[col] = st.number_input(label, value=float(df[col].mean()))

    if st.button("🔍 Predict Income"):
        processed_input = {
            col: encoders[col].transform([user_input[col]])[0] if col in encoders else user_input[col]
            for col in X.columns
        }
        input_df = pd.DataFrame([processed_input])
        prediction = model.predict(input_df)[0]
        result = encoders["income"].inverse_transform([prediction])[0]
        st.success(f"📌 Predicted Income Category: **{result}**")
