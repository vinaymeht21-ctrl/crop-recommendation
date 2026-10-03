# Crop Recommendation System

A machine learning web app that recommends the best crop to plant from soil nutrients (N, P, K), pH, temperature, humidity and rainfall.

**Live demo:** https://YOUR-APP-NAME.streamlit.app

## How it works

- **Data:** 2,200 records, 22 crops, 100 samples each (`data/Crop_recommendation.csv`)
- **Model:** Random Forest, chosen over Logistic Regression and KNN using 5-fold cross-validation
- **Results:** about 99% cross-validated accuracy, 99.5% on a held-out test set
- **App:** Streamlit UI that shows the top 5 crops with confidence scores

All preprocessing sits inside a scikit-learn `Pipeline`, so new inputs are treated exactly like the training data.

## Project structure

```
app.py                     Streamlit web app
notebooks/Crop_Prediction.ipynb   EDA, model comparison, evaluation, export
models/crop_model.joblib   Trained model used by the app
data/Crop_recommendation.csv
requirements.txt           App dependencies
requirements-dev.txt       Adds notebook dependencies
```

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/crop-recommendation.git
cd crop-recommendation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To re-train the model, install `requirements-dev.txt`, open the notebook in `notebooks/` and run all cells.

## Deploy (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Pick the repo, branch `main` and main file `app.py`, then deploy.

## Tech stack

Python, pandas, scikit-learn, Streamlit, Matplotlib, Seaborn

## Disclaimer

Predictions come from a small public dataset and are meant for learning and demonstration, not agronomic advice.

## License

MIT
