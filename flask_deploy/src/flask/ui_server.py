from flask import Flask, request, render_template, url_for;
from PIL import Image;
import base64;
import json;

import requests;


ml_server_url = "http://0.0.0.0:3050/analyze/";

web_app = Flask(__name__);

@web_app.route("/", methods = ["GET", "POST"])
def index_page():

	if request.method == "POST":
		
		patient = request.form["Patient"];

		print(patient);

		image = request.files["retina_image"];

		image.save("static/{}".format(image.filename));

		displayable = Image.open("static/{}".format(image.filename));

		temp_file = open("static/{}".format(image.filename), "rb");

		encoded_image = base64.encodestring(temp_file.read());
		temp_file.close();

		image_name = image.filename[:-3] + "png";

		displayable.save("static/{}".format(image_name));

		http_headers = {"content-type": "application/json", "accept": "application/json"};

		http_payload = encoded_image.decode();

		res = requests.post(url = ml_server_url, headers = http_headers, data = http_payload);

		print("Request sent ok.");
		screening = json.loads(res.text)["screening"];

		#return screening;
		return render_template("index.html", image_name = image_name, patient = patient, screening = screening);


	return render_template("index.html");


@web_app.route("/appointment/", methods = ["GET", "POST"])
def appointment_confirmation_page():

	if request.method == "POST":

		patient = request.form["patient"];
		scheduled_to = request.form["scheduled_to"];

		return render_template("appointment.html", scheduled_to = scheduled_to, patient = patient);

	return "You are not supposed to be here.";

if __name__ == "__main__":
	web_app.run(debug = True, host = "0.0.0.0", port = 5000);