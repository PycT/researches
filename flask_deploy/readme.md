
---
The serving example is in the `/src/flask` folder.
All the other files are the artifacts of research and are ok to get rid of.

`/src/flask/ui_server.py`:
	UI flask server running on port 5000 (`http://localhost:5000`)
	launched as follows: `python ui_server.py`

	It encodes the image chosen to BASE64 and sends it to server where the model is served.
	It receives a JSON with prediction from that server back.
	The appointment scheduling part is a mock up with no real scheduling mechanics behind.

`/src/flask/ml_server.py`:
	A server where the model is deployed and served, running on port 3050 (`http://localhost:3050`)
	Deploy happens on script launch: `python ml_server.py`

	Launch takes a time, wait till model is loaded before starting sending images from UI.

---

An application is based on model prepared for real customer and trained on an open retina dataset.
(http://cecas.clemson.edu/~ahoover/stare/)


DO NOT use tensorflow.keras to train your model unless you intending to use tensorflow-serving to serve it.
If you're ok with Flask than use pure keras.
(tf.get_default_graph is utilized anyways).

---

requirements:

keras
tensorflow
flask
pillow
