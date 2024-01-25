import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("Crop_recommender.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
	try:
		float_features = [float(x) for x in request.form.values()]
		features = [np.array(float_features)]
		prediction = model.predict(features)[0]
		return render_template("index.html", prediction_text = "The recommended crop is {}".format(prediction))
		#return jsonify({'prediction': prediction})

		
	except Exception as e:
		return jsonify({'error': str(e)})
   

if __name__ == "__main__":
    flask_app.run(debug=True)
