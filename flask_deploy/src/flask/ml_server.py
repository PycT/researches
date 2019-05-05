import keras;
from PIL import Image;
from flask import Flask, request, jsonify, Session;
import base64, io;
from tensorflow import get_default_graph;


#model_path = "../../models/inceptionv3_retina_screening1.h5";
model_path = "../../models/inceptionv3_retina_screening600.h5";
input_shape = (600, 600, 3);

web_app = Flask(__name__);
web_app.config["SECRET_KEY"] = "mahkeh";


global model;
global graph;
global test;

model = None;
graph = None;
test = "pew-pew";

def init():

	global model;
	global graph;
	model =	keras.models.load_model(model_path);
	graph = get_default_graph();

@web_app.route("/analyze/", methods = ["POST"])
def prediction():

	global model;
	global graph;
	#init();

	if request.method == "POST":


		encoded_image = request.data;
		decoded_image = base64.b64decode(encoded_image);
		image = Image.open(io.BytesIO(decoded_image));


		resized_image = image.resize((600, 600));

		tensor = keras.preprocessing.image.img_to_array(resized_image);

		input_tensor = tensor.reshape((1,) + tensor.shape);

		with graph.as_default():
			score = model.predict(input_tensor);

		inference = score[0] * 1000 - 453;

		print("Inference = {}".format(inference));

		if (inference > 0.5):
			prediction = "Normal";
		else:
			prediction = "Abnormal";

		print(prediction);

		return jsonify(screening = prediction);

	return "boo";


if __name__ == "__main__":

	init();
	web_app.run(debug = True, host = "0.0.0.0", port = 3050);