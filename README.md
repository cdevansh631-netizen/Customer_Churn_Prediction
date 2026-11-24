# Customer Churn Prediction

A machine learning project that estimates whether a customer may stop using a service. The project includes model exploration and training notebooks, saved preprocessing/model files, and a Streamlit app.

## Live app

[Open the Customer Churn Predictor](https://chunkpredictor.streamlit.app/?)

Enter the customer and service details in the form to see the model's estimated chance of churn and its prediction.

## Project features

- Predicts customer churn using a trained Logistic Regression model.
- Scales numeric inputs with a fitted `StandardScaler`.
- Converts service and account choices with a fitted `OneHotEncoder`.
- Shows an estimated churn probability and a simple prediction in Streamlit.
- Keeps the fitted preprocessing and model files together in `model/`.

## Technology stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Web app | Streamlit |
| Data handling | pandas |
| Machine learning | scikit-learn |
| Saving/loading artifacts | joblib |
| Model used by the app | Logistic Regression |
| Other model experiments | Random Forest, XGBoost, and a Keras neural network |

Dependency versions are listed in [`requirements.txt`](./requirements.txt). The saved scikit-learn artifacts were created with scikit-learn 1.9.1, so keep that version when loading them.

## How it works

1. The training notebook reads `Cleaned.csv`.
2. It separates the `Churn` target from the customer features and excludes `customerID`.
3. It splits the data into training and test sets, keeping the churn proportions similar in both sets.
4. It fits the scaler and one-hot encoder using the training data, then uses those fitted transformers for the test data.
5. It trains and evaluates several models. The Streamlit app uses the Logistic Regression model.
6. The final notebook cell saves the model and preprocessing files in `model/`.
7. The app loads those saved files, applies the same transformations to form inputs, and predicts churn.

The app's input fields cover customer tenure, charges, senior-citizen status, services, contract, and payment details. The prediction is an estimate, not a guarantee.

## Model result

The Logistic Regression model's reported results on the notebook's 20% stratified test split are:

| Measure | Result |
| --- | ---: |
| Accuracy | 80.70% |
| Precision | 66.04% |
| Recall | 56.15% |
| F1 score | 60.69% |

These are test results from this dataset; performance may differ on new customer data.

## Project files

```text
.
├── app.py
├── requirements.txt
├── EDA.ipynb
├── Train.ipynb
└── model/
    ├── Logestic.pkl
    ├── scaler.pkl
    ├── encoder.pkl
    ├── preprocessing_columns.pkl
    └── ANN.keras
```

- [`EDA.ipynb`](./EDA.ipynb): data exploration.
- [`Train.ipynb`](./Train.ipynb): preprocessing, model experiments, evaluation, and artifact export.
- [`app.py`](./app.py): Streamlit user interface and prediction logic.
- `model/Logestic.pkl`: Logistic Regression model used by the app. The filename retains its spelling from the project.
- `model/scaler.pkl`: fitted numeric feature scaler.
- `model/encoder.pkl`: fitted categorical one-hot encoder.
- `model/preprocessing_columns.pkl`: numeric and categorical column names used during training.
- `model/ANN.keras`: neural network experiment; the current Streamlit app does not load it.

`Cleaned.csv` is needed to rerun training, but not to run the deployed app. Do not publish customer data or other private CSV files unless you have permission and have checked that they are safe to share.

## Run locally on Windows

1. Install Python. Python 3.12 is recommended for consistency with the deployment environment.
2. Open PowerShell in this project folder.
3. Install the project packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

5. Open the local address printed in the terminal, usually `http://localhost:8501`.

Press **Ctrl+C** in the terminal to stop the app.

## Retrain the model

1. Place `Cleaned.csv` in the project folder.
2. Open `Train.ipynb` and run the notebook cells in order.
3. Run the final export cell. It saves `Logestic.pkl`, `scaler.pkl`, `encoder.pkl`, and `preprocessing_columns.pkl` into `model/`.
4. Restart the Streamlit app so it loads the newly saved artifacts.

Keep the model and preprocessing files from the same training run together. The app expects the exact filenames shown above.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and the required files in `model/` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Create an app and select the repository and branch.
4. Set the app's main file to `app.py`, then deploy.

The deployed app needs the model and preprocessing files, but it does not need `Cleaned.csv` to make predictions. Avoid adding private or unnecessary CSV files to a public repository.
