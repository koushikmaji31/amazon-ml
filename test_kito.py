import os
import pytesseract
from PIL import Image
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel

# Set Tesseract command path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

dataset_dir = "/raid/ai23resch11003/Adversarial/amazon-ml/images"
image_extensions = ['.png', '.jpg', '.jpeg']

output_dir = "/raid/ai23resch11003/Adversarial/amazon-ml/outputs"
output_filename = "output.csv"

os.makedirs(output_dir, exist_ok=True)

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return None

def apply_bert_to_text(text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)  
    return embeddings

def process_dataset(dataset_dir):
    data = []  

    for filename in os.listdir(dataset_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions): 
            image_path = os.path.join(dataset_dir, filename)
            print(f"Processing {filename}...")

            text = extract_text_from_image(image_path)
            if text:
                print(f"Extracted Text from {filename}: {text[:100]}...")  

                embeddings = apply_bert_to_text(text)
                embeddings_flattened = embeddings.squeeze().tolist() 
                # Append the result as a dictionary
                data.append({
                    'image_name': filename,
                    'extracted_text': text,
                    'bert_embeddings': embeddings_flattened
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
