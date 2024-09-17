import os
import pytesseract
from PIL import Image
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

# Set Tesseract command path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

dataset_dir = "/raid/ai23resch11003/Adversarial/amazon-ml/train_images"
image_extensions = ['.png', '.jpg', '.jpeg']

output_dir = "/raid/ai23resch11003/Adversarial/amazon-ml/outputs"
output_filename = "output_new.csv"

os.makedirs(output_dir, exist_ok=True)

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return None

def process_dataset(dataset_dir):
    data = []  

    # Get list of image files
    image_files = [filename for filename in os.listdir(dataset_dir) if any(filename.lower().endswith(ext) for ext in image_extensions)]

    # Wrap the list with tqdm to show progress
    for filename in tqdm(image_files, desc='Processing Images', unit='image'):
        image_path = os.path.join(dataset_dir, filename)
        print(f"Processing {filename}...")

        text = extract_text_from_image(image_path)
        if text:
            print(f"Extracted Text from {filename}: {text[:100]}...")  

            # Append only the image name and the extracted text
            data.append({
                'Image Name': filename,
                'Extracted Text': text
            })

    return data

# Save data to CSV in the output directory
def save_to_csv(data, output_dir, output_filename):
    df = pd.DataFrame(data)
    
    csv_filepath = os.path.join(output_dir, output_filename)
    
    df.to_csv(csv_filepath, index=False)
    print(f"Data saved to {csv_filepath}")

data = process_dataset(dataset_dir)
save_to_csv(data, output_dir, output_filename)
