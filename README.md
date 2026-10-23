# 💳 LoanIQ — Loan Approval Predictor

A Flask web app that serves a pre-trained **Logistic Regression** model to predict whether a loan application is likely to be approved. It comes with a modern, responsive UI (dark and light mode), an explanation of each prediction, an EMI calculator, and a JSON API. It is ready to deploy on AWS.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![AWS](https://img.shields.io/badge/Deploy-AWS-FF9900?logo=amazonaws&logoColor=white)

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Demo](#-demo)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [The Model](#-the-model)
6. [Getting Started](#-getting-started)
7. [Usage](#-usage)
8. [API Reference](#-api-reference)
9. [Deploying on AWS](#-deploying-on-aws)
10. [Troubleshooting](#-troubleshooting)
11. [Roadmap](#-roadmap)
12. [Contributing](#-contributing)
13. [License](#-license)
14. [Disclaimer](#-disclaimer)

---

## ✨ Features

- 🎯 **Instant predictions**: approval or rejection with a probability score.
- 📊 **Explainable results**: a bar chart of the inputs that pushed the decision up or down.
- 🌗 **Dark and light mode**: remembers your choice and follows the system theme on first visit.
- 🧮 **EMI calculator**: adjustable interest rate, total interest, and EMI as a share of income.
- 🕘 **Prediction history**: the last 8 predictions, stored in the browser.
- 🎲 **Sample and reset**: fill the form with random data or return to defaults in one click.
- 🔌 **REST API**: JSON endpoint with input validation.
- ❤️ **Health check**: `/health` endpoint for AWS load balancers and monitoring.
- 📱 **Responsive design**: works on desktop, tablet, and mobile.

---

## 🖼 Demo

> Add screenshots here after you run the app.

| Dark mode | Light mode |
|-----------|------------|
| `docs/dark.png` | `docs/light.png` |

---

## 🛠 Tech Stack

| Layer      | Technology                                  |
|------------|---------------------------------------------|
| Backend    | Python, Flask                               |
| ML         | scikit-learn (LogisticRegression), pandas   |
| Server     | Gunicorn                                    |
| Frontend   | HTML, CSS, vanilla JavaScript (single file) |
| Deployment | AWS Elastic Beanstalk or EC2                |

---

## 📂 Project Structure

```
.
├── app.py              # Flask app: routes, validation, and the full UI
├── logistic.pkl        # Trained Logistic Regression model
├── requirements.txt    # Python dependencies
├── Procfile            # Gunicorn start command (Elastic Beanstalk)
├── .gitignore
└── README.md
```

The UI is embedded in `app.py`, so there is no `templates/` or `static/` folder to manage.

---

## 🧠 The Model

| Property        | Value                                           |
|-----------------|-------------------------------------------------|
| Algorithm       | Logistic Regression (scikit-learn defaults)     |
| Trained with    | scikit-learn **1.8.0**                          |
| Output          | `1` = Approved, `0` = Rejected                  |
| Input features  | 11                                              |

### Input features

| Feature             | Type        | Accepted values / range                   | Encoding used by the app                          |
|---------------------|-------------|-------------------------------------------|---------------------------------------------------|
| `Gender`            | Categorical | Male, Female                              | Female=0, Male=1                                  |
| `Married`           | Categorical | Yes, No                                   | No=0, Yes=1                                       |
| `Dependents`        | Categorical | 0, 1, 2, 3+                               | 0, 1, 2, 3                                        |
| `Education`         | Categorical | Graduate, Not Graduate                    | Graduate=0, Not Graduate=1                        |
| `Self_Employed`     | Categorical | Yes, No                                   | No=0, Yes=1                                       |
| `ApplicantIncome`   | Numeric     | 0 – 1,000,000                             | as is                                             |
| `CoapplicantIncome` | Numeric     | 0 – 1,000,000                             | as is                                             |
| `LoanAmount`        | Numeric     | 1 – 10,000 (**in thousands**)             | as is                                             |
| `Loan_Amount_Term`  | Numeric     | 6 – 480 (months)                          | as is                                             |
| `Credit_History`    | Binary      | 1 = meets guidelines, 0 = does not        | as is                                             |
| `Property_Area`     | Categorical | Rural, Semiurban, Urban                   | Rural=0, Semiurban=1, Urban=2                     |

> ⚠️ **Important:** the pickle contains only the classifier, not the encoders. The mappings above (alphabetical label encoding) are an assumption. If you encoded your data differently during training, update the `MAPS` dictionary at the top of `app.py`.

### How explanations work

For each prediction the app multiplies every feature value by its model coefficient (`coef × value`) and shows the six largest contributions. Positive bars push towards approval and negative bars push towards rejection. This is a simple, transparent view of a linear model. It is not a full SHAP analysis.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11 or newer** (required by scikit-learn 1.8)
- `pip`
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run locally

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

### Run with Gunicorn (production-style)

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

### Configuration

| Variable     | Default                   | Description                        |
|--------------|---------------------------|------------------------------------|
| `PORT`       | `8000`                    | Port used by `python app.py`       |
| `MODEL_PATH` | `./logistic.pkl`          | Path to the pickled model          |

---

## 🧭 Usage

1. Fill in the applicant details, or click **✨ Sample** to generate random data.
2. Click **Predict approval**.
3. Review the approval probability, risk level, and influencing factors.
4. Use the **EMI calculator** to check whether the monthly payment is affordable.
5. Switch between dark and light mode with the toggle in the header.

---

## 🔌 API Reference

### `POST /api/predict`

**Request**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

**Response `200 OK`**

```json
{
  "approved": true,
  "probability": 83.3,
  "risk": "Low",
  "factors": [
    { "name": "Credit history", "impact": 3.483 },
    { "name": "Loan term", "impact": -2.391 }
  ],
  "time": "14:32:10"
}
```

| Field         | Description                                                      |
|---------------|------------------------------------------------------------------|
| `approved`    | `true` if the model predicts approval                            |
| `probability` | Approval probability in percent                                  |
| `risk`        | `Low` (≥ 75%), `Medium` (50–75%), `High` (< 50%)                 |
| `factors`     | Top six contributing features with their signed impact           |

**Response `400 Bad Request`**: invalid or missing input.

```json
{ "error": "Invalid value for Gender: ''" }
```

### `GET /health`

Returns the app status and the features the model expects. Use it as the health check URL on AWS.

```json
{ "status": "ok", "model": "LogisticRegression", "features": ["Gender", "Married", "..."] }
```

---

## ☁️ Deploying on AWS

### Option A: Elastic Beanstalk (recommended)

1. Add a `Procfile` to the project root:
   ```
   web: gunicorn -w 2 -b 0.0.0.0:8000 app:app
   ```
2. Install the EB CLI and deploy:
   ```bash
   pip install awsebcli
   eb init -p python-3.11 loaniq --region <your-region>
   eb create loaniq-env
   eb open
   ```
3. Set the health check path to `/health` in the environment configuration.
4. To redeploy after changes: `eb deploy`.

> Pick a Python 3.11 or newer platform. Run `eb platform list` to see what is available in your region.

### Option B: EC2

```bash
# On the instance (Amazon Linux / Ubuntu)
sudo apt update && sudo apt install -y python3-pip python3-venv git   # Ubuntu
git clone https://github.com/<your-username>/<your-repo>.git && cd <your-repo>
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

- Open **port 8000** (or 80/443 behind nginx) in the instance's security group.
- To keep the app running after you log out, create a `systemd` service or use `nohup`.

### Security tips

- Put the app behind HTTPS (an Application Load Balancer with an ACM certificate, or nginx with Let's Encrypt).
- Only load pickle files you created yourself. Unpickling an untrusted file can execute arbitrary code.
- Restrict the security group to the ports you need.

---

## 🩺 Troubleshooting

| Problem | Fix |
|---------|-----|
| `InconsistentVersionWarning` or the model fails to load | Install the exact scikit-learn version the model was trained with: `pip install scikit-learn==1.8.0` |
| `No matching distribution found for scikit-learn==1.8.0` | Your Python is older than 3.11. Upgrade Python |
| Predictions look wrong | Check the encoding assumptions in `MAPS` against your training code |
| `ModuleNotFoundError` on Elastic Beanstalk | Make sure `requirements.txt` is in the project root before running `eb deploy` |
| 502 error on Elastic Beanstalk | Check that the `Procfile` points to `app:app` and uses port `8000` |
| Dark/light preference not saved | The browser is blocking `localStorage` (private mode). Everything else still works |

---

## 🗺 Roadmap

- [ ] Add screenshots and a live demo link
- [ ] Bundle the preprocessing pipeline with the model in one file
- [ ] Add SHAP-based explanations
- [ ] Dockerfile and docker-compose
- [ ] Unit tests with `pytest`
- [ ] CI/CD with GitHub Actions

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. Add a `LICENSE` file to the repository to make this official.

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. Predictions are produced by a simple statistical model and must not be used for real lending decisions or treated as financial advice.

---

<p align="center">Made with ❤️ using Flask and scikit-learn</p>
