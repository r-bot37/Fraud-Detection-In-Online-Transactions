import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
from scipy.stats import mode
import numpy as np


app = Flask(__name__)
#loading the model
lrmodel=pickle.load(open('lrmodel.pkl','rb'))
lirmodel=pickle.load(open('lirmodel.pkl','rb'))
dtmodel=pickle.load(open('dtmodel.pkl','rb'))
rfmodel=pickle.load(open('rfmodel.pkl','rb'))
gbmodel=pickle.load(open('gbmodel.pkl','rb'))
# Load the LabelEncoder
label_encoder=pickle.load(open('label_encoder.pkl','rb'))

# Ensure the CSV file is correctly loaded
data = pd.read_csv('Cleaned_data.csv')
# Print the columns to ensure all columns exists
print(data.columns)

@app.route('/')
def index():
   type = data['type'].unique()

# # Define a custom sorting function
#    def custom_sort(location):
#       # Convert float values to strings
#       location = str(location)
#       # Split the location name into text and numerical parts
#       parts = location.split()
#       text_part = ''.join(filter(str.isalpha, parts[0]))  # Extract alphabetic characters
#       num_part = ''.join(filter(str.isdigit, parts[0]))  # Extract numerical characters
#       return (text_part, int(num_part or 0))  # Convert numerical part to integer, handle empty strings

# Sort the locations using the custom sorting function
#    locations = sorted(locations, key=custom_sort)

   return render_template('index.html', **locals())


# def preprocess_input(location):
#     if location not in data['location'].unique():
#         location = 'other'
#     return location

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
       type = request.form['type']
       amount = float(request.form['amount'])
       oldbalanceOrg = float(request.form['oldbalanceOrg'])
       newbalanceOrig=float(request.form['newbalanceOrig'])
       oldbalanceDest = float(request.form['oldbalanceDest'])
       newbalanceDest = float(request.form['newbalanceDest'])
       # Dummy prediction logic, replace with actual model
        # Construct input data DataFrame with reshaped numerical features
       type = label_encoder.transform([type])
       data = pd.DataFrame([[type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest]], columns=["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"])
        # Encode the 'type' column using the saved LabelEncoder

       # type = label_encoder.transform([type])
       lrprediction = lrmodel.predict(data) # Replace with model prediction logic
       lirprediction = lirmodel.predict(data)
       dtprediction = dtmodel.predict(data)
       rfprediction = rfmodel.predict(data)
       gbprediction = gbmodel.predict(data)
       combined_predictions = np.array([lrprediction, gbprediction, rfprediction,dtprediction])
       prediction = mode(combined_predictions)[0][0]
       if prediction==1:
           return render_template('index.html',pred='The transaction is fraudulent')
       else:
           return render_template('index.html',pred='The transaction is not fraudulent')

if __name__ == "__main__":
    app.run(debug=True, port=5000)