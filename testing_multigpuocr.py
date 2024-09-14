from warnings import filterwarnings
filterwarnings("ignore")

import easyocr
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Define image directory and output file
dataset_dir = "train_images"
image_extensions = ['.png', '.jpg', '.jpeg']
output_dir = "outputs"
output_filename = "easyocr_output.csv"

os.makedirs(output_dir, exist_ok=True)

# Function to initialize the EasyOCR reader with a specific GPU
def init_reader(gpu_index):
    return easyocr.Reader(['en'], gpu=True, gpu_index=gpu_index)

# Function to extract text using EasyOCR
def extract_text_from_image(image_path, reader):
    try:
        result = reader.readtext(image_path, detail=0)
        result_string = ' '.join(result)  # Join the OCR results into a single string
        return result_string
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return None

# Process dataset directory to extract text from images and write directly to CSV
def process_and_save_to_csv(gpu_index, image_files):
    reader = init_reader(gpu_index)
    data = []

    for filename in tqdm(image_files, desc=f'GPU {gpu_index} Processing Images', unit='image'):
        image_path = os.path.join(dataset_dir, filename)
        print(f"Processing {filename}...")

        text = extract_text_from_image(image_path, reader)
        if text:
            data.append({'Image Name': filename, 'Extracted Text': text})

    return data

# Main function to distribute the images across GPUs
def run_parallel(dataset_dir, num_gpus):
    image_files = [filename for filename in os.listdir(dataset_dir) if any(filename.lower().endswith(ext) for ext in image_extensions)]
    
    # Split the image files between GPUs
    chunk_size = len(image_files) // num_gpus
    image_chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    # Set up multiprocessing with each GPU running its own batch
    with Pool(processes=num_gpus) as pool:
        results = pool.starmap(process_and_save_to_csv, [(i, image_chunks[i]) for i in range(num_gpus)])

    # Flatten the results from each GPU into a single list
    all_data = [item for sublist in results for item in sublist]

    # Save all data to a CSV
    df = pd.DataFrame(all_data)
    csv_filepath = os.path.join(output_dir, output_filename)
    df.to_csv(csv_filepath, index=False)
    print(f"Data saved to {csv_filepath}")

# Set the number of GPUs (adjust based on available GPUs)
num_gpus = 5

# Run the processing in parallel across GPUs
run_parallel(dataset_dir, num_gpus)
