
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the company.")

st.divider()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "🔮 Predict Churn",
    "ℹ️ About Project"
])


# ==================================
# TAB 1: CUSTOMER CHURN DASHBOARD
# ==================================

with tab1:

    st.header("Customer Churn Analysis")

    # Load dataset
    df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

    # Calculate statistics
    total_customers = len(df)

    churned_customers = len(df[df["Churn"] == "Yes"])

    retained_customers = len(df[df["Churn"] == "No"])

    churn_rate = (churned_customers / total_customers) * 100

    # Display metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", churned_customers)
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    # 1. Churn Distribution
    st.subheader("Customer Churn Distribution")

    fig, ax = plt.subplots()

    sns.countplot(data=df, x="Churn", ax=ax)

    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    plt.close(fig)

    # 2. Contract-wise Churn
    st.subheader("Churn Based on Contract Type")

    fig, ax = plt.subplots()

    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)

    ax.set_title("Contract Type vs Churn")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    plt.close(fig)

    # 3. Tenure vs Churn
    st.subheader("Customer Tenure vs Churn")

    fig, ax = plt.subplots()

    sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)

    ax.set_title("Customer Tenure vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Tenure (Months)")

    st.pyplot(fig)
    plt.close(fig)

    # 4. Monthly Charges vs Churn
    st.subheader("Monthly Charges vs Churn")

    fig, ax = plt.subplots()

    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)

    ax.set_title("Monthly Charges vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")

    st.pyplot(fig)
    plt.close(fig)


# ==================================
# TAB 2: CUSTOMER CHURN PREDICTION
# ==================================

with tab2:

    st.header("Predict Customer Churn")

    st.write("Enter customer details to predict the possibility of churn.")

    st.subheader("Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])

    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Has Partner?", ["Yes", "No"])

    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (Months)", min_value=0, max_value=100, value=12
    )

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    # Predict button
    if st.button("Predict Churn"):

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        prediction = model.predict(customer)

        probability = model.predict_proba(customer)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Churn Probability",
            f"{probability * 100:.2f}%"
        )

        if prediction[0] == 1:
            st.error("Customer is likely to churn!")
        else:
            st.success("Customer is likely to stay!")


# ==================================
# TAB 3: ABOUT PROJECT
# ==================================

with tab3:

    st.header("About This Project")

    st.write("""
    **Customer Churn Prediction and Retention Analysis Using Machine Learning**

    This project predicts whether a telecom customer is likely to leave
    the company based on their service usage, contract details and
    billing information.

    The objective is to help telecom companies identify customers
    who may leave and take appropriate retention measures.
    """)

    st.subheader("Dataset")

    st.write("""
    - Dataset: IBM Telco Customer Churn
    - Total Records: 7,043
    - Original Features: 20 input features and 1 target column
    - Target Variable: Churn (Yes / No)
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Matplotlib and Seaborn
    - Scikit-learn
    - Streamlit
    - Joblib
    """)

    st.subheader("Machine Learning Algorithms")

    st.write("""
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    """)

    st.subheader("Project Workflow")

    st.write("""
    1. Data Collection
    2. Data Cleaning
    3. Exploratory Data Analysis (EDA)
    4. Data Preprocessing
    5. Train-Test Split
    6. Model Training
    7. Model Evaluation
    8. Model Saving
    9. Customer Churn Prediction
    """)

    st.subheader("Final Model")

    st.write("""
    Logistic Regression was selected as the initial final model
    based on its performance in the model comparison.
    """)


# ==================================
# FOOTER
# ==================================

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project | Python, Scikit-learn & Streamlit"
)
