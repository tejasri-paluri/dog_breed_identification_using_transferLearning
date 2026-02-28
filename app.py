from flask import Flask, render_template, request, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# configure upload path inside static so files can be served
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("dogbreed.h5")


# Dog breed names (same order as training)
class_names = [
'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber',
'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont',
'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter',
'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
'french_bulldog', 'german_shepherd', 'german_short-haired_pointer',
'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane',
'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel',
'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor',
'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg',
'lhasa', 'malamute', 'malinois', 'maltese_dog',
'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
'otterhound', 'papillon', 'pekinese', 'pembroke',
'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback',
'rottweiler', 'saint_bernard', 'saluki', 'samoyed',
'schipperke', 'scotch_terrier', 'scottish_deerhound',
'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu',
'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier',
'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer',
'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier',
'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier',
'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier'
]


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # Get uploaded file
        f = request.files['file']
        filename = f.filename

        # Save file under static so it can be displayed
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        pred = model.predict(img_array)
        predicted_index = np.argmax(pred)

        # Convert to breed name
        breed_name = class_names[predicted_index]

        image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template("output.html", breed_name=breed_name, image_url=image_url)

    return render_template("predict.html")


# Run app
if __name__ == "__main__":
    app.run(debug=True)