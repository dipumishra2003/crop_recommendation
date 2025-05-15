from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and the encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get inputs from form
    features = [float(request.form[f]) for f in 
                ['N','P','K','temperature','humidity','ph','rainfall']]
    arr = np.array([features])

    # model prediction gives an array of encoded labels
    pred_enc = model.predict(arr)            # e.g. array([15])
    
    # use the encoder to reverse-map it back to crop name
    pred_crop = encoder.inverse_transform(pred_enc)[0]

    return render_template('index.html', prediction_text=f"Recommended Crop: {pred_crop}")

if __name__ == "__main__":
    app.run(debug=True)
