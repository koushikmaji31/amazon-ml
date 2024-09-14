from warnings import filterwarnings
filterwarnings("ignore")

import easyocr
from PIL import Image

reader = easyocr.Reader(['en'], gpu=True)

def extract_text(image_path):
    result = reader.readtext(image_path, detail=0)
    result_string = ' '.join(result)
    return result_string

image_path = 'https://m.media-amazon.com/images/I/51WsuKKAVrL.jpg'
result_string = extract_text(image_path)
print(result_string)
