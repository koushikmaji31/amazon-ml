import pandas as pd

train_df = pd.read_csv('/DATA1/ai23mtech12001/Amazon/amazon-ml/dataset/test.csv')
easyocr_df = pd.read_csv('outputs/easyocr_test_output.csv')
tesseract_df = pd.read_csv('outputs/output_new_tes_test.csv')

train_df['image_name'] = train_df['image_link'].apply(lambda x: x.split('/')[-1])

easyocr_df.rename(columns={'Image Name': 'image_name', 'Extracted Text': 'easyocr_text'}, inplace=True)
tesseract_df.rename(columns={'image_name': 'image_name', 'extracted_text': 'tesseract_text'}, inplace=True)

merged_df = pd.merge(train_df, easyocr_df, on='image_name', how='left')
merged_df = pd.merge(merged_df, tesseract_df, on='image_name', how='left')

merged_df.to_csv('final_test.csv', index=False)

print("Merged CSV file has been created as 'final.csv'.")