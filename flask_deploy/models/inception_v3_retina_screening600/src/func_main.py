import tensorflow as tf;
import hydro_serving_grpc as hs;

model_path = "/model/files/inceptionv3_retina_screening1.h5";

model = tf.keras.models.load_model(model_path);

def predict(shaped_image):

	result = model(shaped_image);

	inference = result[0][0] * 1000 - 453;

	if (inference > 0.5):
		prediction = "normal ({})".format(inference);
	else:
		prediction = "abnormal ({})".format(inference);

	prediction_tensor_shape = hs.TensorShapeProto(dim = hs.TensorShapeProto(size = 1));

	prediction_tensor_proto = hs.TensorProto\
	(
		dtype = hs.DT_STRING,
		string_val = [prediction],
		tensor_shape = prediction_tensor_shape
	);

	return hs.PredictResponse(outputs = {"result": prediction_tensor_proto});