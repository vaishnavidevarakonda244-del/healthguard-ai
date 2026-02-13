from flask import Flask, request
import numpy as np
from sklearn.linear_model import LogisticRegression
import datetime

app = Flask(__name__)

# -----------------------------
# TRAIN MODEL (AUTO AT START)
# -----------------------------

# fever, cough, fatigue
X = np.array([
    [1,1,1],
    [1,1,0],
    [1,0,1],
    [0,1,1],
    [0,0,1],
    [0,0,0],
    [1,0,0],
    [0,1,0]
])

# 0 = Low, 1 = Medium, 2 = High
y = np.array([2,2,2,1,1,0,0,0])

model = LogisticRegression(multi_class="multinomial", max_iter=200)
model.fit(X, y)

# -----------------------------
# HTML UI
# -----------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>HealthGuard AI</title>
<style>
body { font-family: Arial; text-align: center; margin-top: 40px; }
input { margin: 10px; padding: 8px; width: 60px; }
button { padding: 10px 20px; background: green; color: white; border: none; }
.result { margin-top: 20px; font-size: 18px; font-weight: bold; }
.low { color: green; }
.medium { color: orange; }
.high { color: red; }
.alert { margin-top: 15px; color: red; font-weight: bold; }
</style>
</head>
<body>

<h2>HealthGuard AI - Real-Time Risk Prediction</h2>

<form method="POST">
Fever (1=Yes, 0=No):<br>
<input type="number" name="fever" min="0" max="1" required><br>

Cough (1=Yes, 0=No):<br>
<input type="number" name="cough" min="0" max="1" required><br>

Fatigue (1=Yes, 0=No):<br>
<input type="number" name="fatigue" min="0" max="1" required><br>

<button type="submit">Predict</button>
</form>

<div class="result">{result}</div>
<div class="alert">{alert}</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    alert = ""

    if request.method == "POST":
        fever = int(request.form["fever"])
        cough = int(request.form["cough"])
        fatigue = int(request.form["fatigue"])

        features = np.array([[fever, cough, fatigue]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = round(max(probabilities) * 100, 2)

        if prediction == 0:
            result = f"<span class='low'>Low Risk ({confidence}% confidence) - Monitor symptoms.</span>"
        elif prediction == 1:
            result = f"<span class='medium'>Medium Risk ({confidence}% confidence) - Consider doctor consultation.</span>"
        else:
            result = f"<span class='high'>High Risk ({confidence}% confidence) - Immediate medical attention required.</span>"

            # Simulated Alert System
            time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert = f"""
            🚨 Emergency Alert Triggered! <br>
            ✔ Family Member Notified <br>
            ✔ Nearby Hospital Alert Sent <br>
            Time: {time_now}
            """

    return HTML_PAGE.format(result=result, alert=alert)

if __name__ == "__main__":
    app.run(debug=True)
