import numpy as np
from keras.models import load_model
from scipy.misc import imread, imresize
from prepare_data import normalize
from flask import Flask,render_template, request
import base64
import json

mlp = load_model("D:\\Python Projects\\Drawing Predication\\mlp_94.h5")
conv = load_model("D:\\Python Projects\\Drawing Predication\\vehicle.h5")
FRUITS = {0: "Airplane", 1: "Bicycle", 2: "Bus", 3: "Helicopter", 4: "Truck"}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index2.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        net = request.form["net"]
    img = base64.b64decode(data)
    print(img)
    with open('D:\\Python Projects\\Drawing Predication\\temp.png', 'wb') as output:
        output.write(img)

    x = imread('D:\\Python Projects\\Drawing Predication\\temp.png', mode='L')

    x = imresize(x, (28, 28))

    model = conv
    x = np.expand_dims(x, axis=0)
    x = np.reshape(x, (28, 28, 1))
    # invert the colors
    x = np.invert(x)
    # brighten the image by 60%
    for i in range(len(x)):
        for j in range(len(x)):
            if x[i][j] > 50:
                x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

    x = normalize(x)
    val = model.predict(np.array([x]))
    pred = FRUITS[np.argmax(val)]
    classes = ["Airplane", "Bicycle","Bus", "Helicopter", "Truck"]
    print(pred)
    print(list(val[0]))
    return render_template("index2.html", preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"], net=net)
    
if __name__ == "__main__":
    app.run()
