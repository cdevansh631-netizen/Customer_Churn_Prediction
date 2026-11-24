from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_DIR = Path(__file__).resolve().parent / "model"


@st.cache_resource
def load_artifacts():
    required_files = [
        MODEL_DIR / "Logestic.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "encoder.pkl",
        MODEL_DIR / "preprocessing_columns.pkl",
    ]
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(", ".join(missing_files))

    model = joblib.load(required_files[0])
    scaler = joblib.load(required_files[1])
    encoder = joblib.load(required_files[2])
    columns = joblib.load(required_files[3])
    return model, scaler, encoder, columns


st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 10%, rgba(96, 165, 250, 0.20), transparent 28%),
            radial-gradient(circle at 92% 18%, rgba(167, 139, 250, 0.18), transparent 25%),
            linear-gradient(135deg, #f7fbff 0%, #eef7f5 55%, #f5f1ff 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .stApp h1 {
        color: #17324d;
        font-size: 2.6rem;
        letter-spacing: -0.04em;
    }

    .stApp h2,
    .stApp h3 {
        color: #142b43 !important;
        font-weight: 700;
    }

    .stApp p {
        color: #526579;
    }

    [data-testid="stForm"] {
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 16px 45px rgba(35, 57, 86, 0.09);
    }

    [data-testid="stWidgetLabel"] p {
        color: #263b50;
        font-weight: 600;
    }

    [data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 700;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        border: 0;
        background: linear-gradient(90deg, #1d4ed8, #6d28d9);
        color: white;
    }

    [data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Customer Churn Predictor")
st.write(
    "Enter the customer details below. The app will estimate whether the "
    "customer may leave the service."
)

try:
    model, scaler, encoder, columns = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Some model files are missing: {error}. "
        "Run the final cells in Train.ipynb to create them in the model folder."
    )
    st.stop()

numeric_columns = columns["numeric_columns"]
categorical_columns = columns["categorical_columns"]
encoder_categories = dict(zip(categorical_columns, encoder.categories_))

friendly_names = {
    "Partner": "Does the customer have a partner?",
    "Dependents": "Does the customer have dependents?",
    "PhoneService": "Does the customer have phone service?",
    "MultipleLines": "Does the customer have multiple phone lines?",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract length",
    "PaperlessBilling": "Paperless billing?",
    "PaymentMethod": "Payment method",
    "gender": "Gender",
}

with st.form("customer_details"):
    st.subheader("Basic details")
    left, right = st.columns(2)
    with left:
        senior_citizen = st.selectbox(
            "Is the customer a senior citizen?", ["No", "Yes"]
        )
        tenure = st.number_input(
            "How many months has the customer stayed?",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
        )
    with right:
        monthly_charges = st.number_input(
            "Monthly bill amount",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total amount billed so far",
            min_value=0.0,
            value=840.0,
            step=10.0,
        )

    st.subheader("Services and account")
    customer = {
        "SeniorCitizen": int(senior_citizen == "Yes"),
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    for column in categorical_columns:
        label = friendly_names.get(column, column)
        options = [str(value) for value in encoder_categories[column]]
        customer[column] = st.selectbox(label, options)

    submitted = st.form_submit_button("Check churn estimate")

if submitted:
    customer_data = pd.DataFrame([customer])
    scaled_numbers = pd.DataFrame(
        scaler.transform(customer_data[numeric_columns]),
        columns=numeric_columns,
        index=customer_data.index,
    )
    encoded_categories = pd.DataFrame(
        encoder.transform(customer_data[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
        index=customer_data.index,
    )
    prepared_customer = pd.concat(
        [scaled_numbers, encoded_categories], axis=1
    )

    prediction = model.predict(prepared_customer)[0]
    churn_class_index = list(model.classes_).index(1)
    churn_probability = model.predict_proba(prepared_customer)[0][
        churn_class_index
    ]

    st.subheader("Estimate")
    st.metric("Estimated chance of leaving", f"{churn_probability:.0%}")
    if prediction == 1:
        st.warning("The model thinks this customer may leave.")
    else:
        st.success("The model thinks this customer is likely to stay.")
    st.caption("This is an estimate from the model, not a guarantee.")