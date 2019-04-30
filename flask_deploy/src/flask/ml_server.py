import tensorflow as tf;
from PIL import Image;
from flask import Flask, request, jsonify;
import base64, io;

web_app = Flask(__name__);

model_path = "../../models/inceptionv3_retina_screening1.h5";

input_shape = (600, 600, 3);

global graph;
graph = tf.get_default_graph();

model = tf.keras.models.load_model(model_path);
model.summary();


# def screen(model, image):

# 	resized_image = image.resize((600, 600));

# 	tensor = tf.keras.preprocessing.image.img_to_array(resized_image);

# 	input_tensor = tensor.reshape((1,) + tensor.shape);

# 	with graph.as_default():
# 		score = model.predict(input_tensor);

# 	inference = score * 1000 - 453;

# 	print("Inference = {}".format(inference));

# 	if (inference > 0.5):
# 		prediction = 8;#"Normal";
# 		print("Normal");
# 	else:
# 		prediction = 4;#"Abnormal";
# 		print("Abnormal");

# 	return prediction;

@web_app.route("/analyze/", methods = ["POST"])
def prediction():

	if request.method == "POST":

		encoded_image = request.data;
		decoded_image = base64.b64decode(encoded_image);
		image = Image.open(io.BytesIO(decoded_image));

#		prediction = screen(model, image);

		resized_image = image.resize((600, 600));

		tensor = tf.keras.preprocessing.image.img_to_array(resized_image);

		input_tensor = tensor.reshape((1,) + tensor.shape);

		with graph.as_default():
			score = model.predict(input_tensor);

		inference = score * 1000 - 453;

		print("Inference = {}".format(inference));

		if (inference > 0.5):
			prediction = 8;#"Normal";
			print("Normal");
		else:
			prediction = 4;#"Abnormal";
			print("Abnormal");


		return jsonify(screening = prediction);

	return "boo";


if __name__ == "__main__":

	web_app.run(debug = True, host = "0.0.0.0", port = 3050);