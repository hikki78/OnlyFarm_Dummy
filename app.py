from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

with open('num_of_crop.pkl', 'rb') as f:
    crop_dict = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def frontend():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract data from the request
        data = request.json

        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])
        temp = float(data['Temperature'])
        moisture = float(data['Moisture'])
        ph = float(data['pH'])
        rainfall = float(data['Rainfall'])

        feature_list = [N, P, K, temp, moisture, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        prediction = model.predict(final_features)

        if not np.issubdtype(prediction.dtype, np.integer):
            prediction = prediction.astype(int)

        # Use the trained label encoder to get the crop name
        if prediction[0] in crop_dict.values():
            crop_name = [crop for crop, label in crop_dict.items()
                         if label == prediction[0]][0]
            result = "{} is the best crop to be cultivated right there".format(
                crop_name)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        return jsonify({'result': result})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'result': 'Error processing the request'})


if __name__ == "__main__":
    app.run(debug=True)
