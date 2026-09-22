from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get values from HTML form
    cgpa = float(request.form['CGPA'])
    internships = float(request.form['Internships'])
    projects = float(request.form['Projects'])
    certifications = float(request.form['Certifications'])
    communication_skills = float(request.form['Communication_Skills'])
    backlogs = float(request.form['Backlogs'])

    features = [cgpa, internships, projects,
                certifications,communication_skills, backlogs]

    # Convert to NumPy array
    final_features = np.array([features])

    # Make prediction
    prediction = model.predict(final_features)

    # Convert prediction to text
    output= 'Placed' if prediction[0] == 1 else 'Not Placed'

    # Display result
    return render_template( 'index.html', prediction_text="Prediction: " + output )

if __name__ == "__main__":
    app.run(debug=True)
