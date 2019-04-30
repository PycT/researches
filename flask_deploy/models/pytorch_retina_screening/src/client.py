from torch import tensor;
from torchvision import transforms;
from PIL import Image;

def load_image(path_to_image):

img_transform = transforms.Compose\
	(
	    [
	        transforms.Resize((299, 299)),
	        transforms.ToTensor(),
	    ]
	);
    
    image = Image.open(path_to_image);
    image = img_transform(image).float();
    image = torch.tensor(image);
    image = image.unsqueeze(0);
    
    return image;