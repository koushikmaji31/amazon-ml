from warnings import filterwarnings
filterwarnings("ignore")

import easyocr
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# Define image directory and output file
dataset_dir = "train_images"
image_extensions = ['.png', '.jpg', '.jpeg']
output_dir = "outputs"
output_filename = "easyocr_output.csv"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Initialize EasyOCR reader (outside the loop)
reader = easyocr.Reader(['en'], gpu=True)

# CSV file path where the data will be written
csv_filepath = os.path.join(output_dir, output_filename)

# Function to extract text using EasyOCR
def extract_text_from_image(image_path):
    try:
        result = reader.readtext(image_path, detail=0)
        result_string = ' '.join(result)  # Join the OCR results into a single string
        return result_string
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return None

# Process dataset directory to extract text from images and write directly to CSV
def process_and_save_to_csv(dataset_dir, csv_filepath):
    # Open the CSV file in write mode and write headers
    with open(csv_filepath, mode='w') as csv_file:
        csv_file.write("Image Name,Extracted Text\n")  # CSV Header
        
        # Get list of image files
        image_files = [filename for filename in os.listdir(dataset_dir) if any(filename.lower().endswith(ext) for ext in image_extensions)]
        
        for filename in tqdm(image_files, desc='Processing Images', unit='image'):
            image_path = os.path.join(dataset_dir, filename)
            print(f"Processing {filename}...")

            # Extract text from the image
            text = extract_text_from_image(image_path)
            if text:
                # Write the filename and extracted text to the CSV file
                csv_file.write(f"{filename},{text.replace(',', ' ')}\n")  # Replace commas to avoid CSV issues
                print(f"Extracted Text from {filename}: {text[:100]}...")  # Print the first 100 characters of the extracted text

# Main function to process the dataset and save results to CSV
process_and_save_to_csv(dataset_dir, csv_filepath)
