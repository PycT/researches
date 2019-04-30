from torch import nn, load, tensor;
from torchvision import models, transforms;

import hydro_serving_grpc as hs;

from inceptionmodel import InceptionModel;

path_to_the_model = "/model/files/retina_screening1.pt";
#path_to_the_model = "../../retina_screening1.pt";

model = load(path_to_the_model);
model.eval();


def predict(sample):

	output = model(sample);

	if (output[0] > output[1]):
		prediction = [b"normal"];
	else:
		prediction = [b"abnormal"];

	response_tensor_shape = hs.TensorShapeProto(dim = hs.TensorShapeProto.Dim(size = 1));

	return hs.PredictResponse(outputs = {"result": hs.TensorProto(dtype = DT_STRING,\
	 string_val = prediction, tensor_shape = response_tensor_shape)});