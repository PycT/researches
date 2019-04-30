import sys;

import tensorflow as tf;
from PIL import Image;
import json;

import grpc;
import hydro_serving_grpc as hs;


config_file = open("config.json", "r");
configuration = json.loads(config_file.read());
config_file.close();


if (len(sys.argv) < 2):
	path_to_the_image = "im0162.ppm";
	#print("Please specify the path to the image in command line, e.g. :\n python client.py image.png");
else:
	path_to_the_image = sys.argv[1];


creds = grpc.ssl_channel_credentials();
channel = grpc.secure_channel('dev.k8s.hydrosphere.io:443', creds);
stub = hs.PredictionServiceStub(channel);
model_spec = hs.ModelSpec(name="retina_screening");


image = Image.open(path_to_the_image);

resized_image = image.resize((600, 600));

image_array = tf.keras.preprocessing.image.img_to_array(resized_image);

image_shaped = image_array.reshape((1,) + image_array.shape);


image_tensor_shape = hs.TensorShapeProto\
(
	dim = \
	[
		hs.TensorShapeProto.Dim(size = dim)\
		for dim in image_shaped.shape
	]
);

image_tensor_proto = hs.TensorProto\
(
	dtype = hs.DT_DOUBLE,
	double_val = image_shaped.flatten(),
	tensor_shape = image_tensor_shape
);

request = hs.PredictRequest(model_spec = model_spec, inputs = {"shaped_image": image_tensor_proto})

result = stub.Predict(request);

print(result);