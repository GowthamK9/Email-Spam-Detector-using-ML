from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# ======================== Loading the saved models ==============================
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl file not found!")
    model = None

try:
    with open("feature_extraction.pkl", "rb") as f:
        feature_extraction = pickle.load(f)
except FileNotFoundError:
    print("Error: feature_extraction.pkl file not found!")
    feature_extraction = None

# ======================== Prediction Function ===================================
def predict_mail(input_text):
    if model is None or feature_extraction is None:
        return "Error: Model or Feature Extraction not loaded!"
    
    input_user_mail = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model.predict(input_data_features)

    # Return "Spam" or "Ham" based on prediction
    return "Spam" if prediction[0] == 1 else "Ham"

# ======================== Flask Routes ==========================================
@app.route('/', methods=['GET', 'POST'])
def analyze_mail():
    if request.method == 'POST':
        mail = request.form.get('mail')
        if not mail:
            return render_template('index.html', classify="Please enter a mail to analyze.", color="black")
        
        predicted_mail = predict_mail(mail)
        # Set color for prediction output (red for Spam, green for Ham)
        color = "red" if predicted_mail == "Spam" else "green"
        return render_template('index.html', classify=predicted_mail, color=color)

    return render_template('index.html')

# ======================== Running the Flask App =================================
if __name__ == '__main__':
    app.run(debug=True)
