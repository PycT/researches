import base64;

image_file = open("flash.jpg", "rb");

encoded_image = base64.encodestring(image_file.read());

print(encoded_image);