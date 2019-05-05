Application based on model prepared for real customer and trained on an open retina dataset.
(http://cecas.clemson.edu/~ahoover/stare/)


DO NOT use tensorflow.keras to train your model unless you intending to use tensorflow-serving to serve it.
If you're ok with Flask than use pure keras.
(tf.get_default_graph is utilized anyways).

