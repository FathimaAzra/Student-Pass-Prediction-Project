
from flask import Flask, render_template, request, jsonify
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -------------------------------------------
# 1. LOAD HISTORICAL DATA
# -------------------------------------------

DATA_PATH = "student_data.csv"

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments"
]

df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates().copy()
df["Final_Result"] = (
    df["Final_Result"].astype(str).str.strip().str.title()
)

if df[FEATURES + ["Final_Result"]].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

# if not set(df["Final_Result"].unique()).issubset({"Pass", "Fail"}):
#     raise ValueError("Final_Result must contain Pass or Fail only.")

# if df["Final_Result"].nunique() != 2:
#     raise ValueError("Dataset must contain both Pass and Fail.")

# -------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# -------------------------------------------

X = df[FEATURES]
y = df["Final_Result"]

# -------------------------------------------
# 3. SPLIT DATA
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------
# 4. CREATE AND TRAIN MODEL
# -------------------------------------------

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# -------------------------------------------
# 5. EVALUATE MODEL
# -------------------------------------------

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

print("Student Pass Prediction System")
print("Model trained successfully!")
print(f"Test accuracy: {accuracy * 100:.2f}%")

# -------------------------------------------
# 6. HOME PAGE
# -------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------------------
# 7. PREDICTION API
# -------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON input."}), 400

        # Convert values and validate on the server too
        values = {}

        for feature in FEATURES:
            value = data.get(feature)

            if value is None or isinstance(value, bool):
                return jsonify({
                    "error": f"Missing or invalid field: {feature}"
                }), 400

            try:
                values[feature] = float(value)
            except (TypeError, ValueError):
                return jsonify({
                    "error": f"{feature} must be a number."
                }), 400

            if not pd.notna(values[feature]):
                return jsonify({
                    "error": f"{feature} must be a finite number."
                }), 400

        student = pd.DataFrame([values], columns=FEATURES)

        # Check realistic input ranges
        limits = {
            "Study_Hours": (0, 24),
            "Attendance": (0, 100),
            "Previous_Marks": (0, 100),
            "Assignments": (0, 10)
        }

        for feature, (minimum, maximum) in limits.items():
            value = values[feature]

            if not minimum <= value <= maximum:
                return jsonify({
                    "error": f"{feature} must be between "
                             f"{minimum} and {maximum}."
                }), 400

        if not values["Assignments"].is_integer():
            return jsonify({
                "error": "Assignments must be a whole number."
            }), 400

        # Generate prediction
        prediction = model.predict(student)[0]
        probabilities = model.predict_proba(student)[0]

        # Probability corresponding to the predicted class
        predicted_index = list(model.classes_).index(prediction)
        probability = float(probabilities[predicted_index] * 100)

        return jsonify({
            "prediction": str(prediction),
            "probability": round(probability, 2),
            "test_accuracy": round(float(accuracy * 100), 2)
        })

    except Exception as error:
        app.logger.exception("Prediction error")
        return jsonify({
            "error": "An internal error occurred while predicting."
        }), 500

# -------------------------------------------
# 8. RUN SERVER
# -------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)