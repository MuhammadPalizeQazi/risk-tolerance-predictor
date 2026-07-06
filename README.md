# True Risk Tolerance Predictor

A Flask web app that predicts an individual's financial risk tolerance from a handful of demographic and financial inputs, serving a trained XGBoost regression model behind a simple form.

## What it does

A user fills in age, education level, marital status, occupation category, number of kids, net worth, income, debt, and whether they self-identify as someone who takes financial risks. The app feeds these into a trained model and returns a predicted **True Risk Tolerance** score — the idea being that people's self-reported risk appetite often doesn't match what their actual financial situation would support, and a model trained on real survey data can estimate the latter.

## Model

A tuned `XGBRegressor` (`n_estimators=194`, `max_depth=9`, `learning_rate≈0.172`, `colsample_bytree≈0.746`), taking 9 features:

| Feature | Relative importance |
|---|---|
| Net worth | 30.3% |
| Debt | 12.1% |
| Self-reported risk appetite | 10.7% |
| Income | 9.5% |
| Occupation category | 8.5% |
| Marital status | 7.4% |
| Number of kids | 7.3% |
| Education level | 7.2% |
| Age | 6.9% |

Net worth dominates the prediction by a wide margin, roughly 2.5x the next most important feature, which lines up with intuition: how much of a financial cushion someone actually has predicts real risk capacity better than self-reported attitude or income alone.

**Note on scope:** this repo covers the deployed inference app, the trained model artifact (`model.pkl`) and the hyperparameters/feature importances it exposes. The original training notebook and dataset weren't preserved alongside this deployment, so the app is documented here from the model's own attributes rather than from training-time analysis.

## Running it locally

```
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── app.py                 # Flask routes + model inference
├── model.pkl              # trained XGBRegressor
├── templates/
│   └── index.html         # input form + result display
└── static/
    ├── style.css
    ├── script.js
    └── background.jpg
```

## Stack

Python · Flask · XGBoost · Bootstrap
