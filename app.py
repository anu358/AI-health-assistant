from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Basic advice dictionary
advice = {
    "Common Cold": "Rest, drink fluids, and use warm water gargle.",
    "COVID-19": "Isolate, wear a mask, consult a doctor if symptoms worsen.",
    "Heart Disease": "Seek immediate medical consultation.",
    "Allergy": "Avoid allergens, use antihistamines if prescribed.",
    "Food Poisoning": "Stay hydrated, avoid outside food, consult if persistent.",
    "skin allergy":"GO to checked up by skin doctor and always clean your body,bath daily"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    suggestion = None

    if request.method == "POST":
        symptoms = request.form["symptoms"]
        user_vec = vectorizer.transform([symptoms])
        prediction = model.predict(user_vec)[0]
        suggestion = advice.get(prediction, "Consult a doctor for proper guidance.")

    return render_template("index.html", prediction=prediction, suggestion=suggestion)

if __name__ == "__main__":
    app.run(debug=True)
