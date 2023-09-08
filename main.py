from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

type(5)
with open("c_v_d.pkl", "rb") as file:
    model = pickle.load(file)

# Create Server
app = Flask(__name__,static_folder='static')



@app.route("/", methods=["GET"])
def root():
    return render_template("index.html")



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        
        # Handle the image upload
        uploaded_file = request.files['image']
        
        
        # Process the uploaded file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        # image = np.frombuffer(uploaded_file.read(), np.uint8)
        # image = np.fromstring(uploaded_file.read(), np.uint8)
        
        image = cv2.resize(image,(256,256))
        print(image.shape)
        image_input = image.reshape((1,256,256,3))
        print(image_input.shape)
        
        result = model.predict(image_input)
        # result = model.predict(image)

        return render_template('index.html', output=int(result))


# start the server
app.run(port=4001, host="0.0.0.0", debug=True)