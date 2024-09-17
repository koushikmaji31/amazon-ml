import pandas as pd
from unit_extractor import extract_value_unit

# Define the input and output CSV paths
input_csv_path = 'output.csv'
test_csv_path = 'dataset/test.csv'
output_csv_path = 'processed_output.csv'

# Load the test CSV to get the base structure
test_df = pd.read_csv(test_csv_path)

# Load the output CSV that contains the index and predictions
output_df = pd.read_csv(input_csv_path)

# Ensure the 'index' column exists in both DataFrames
if 'index' not in test_df.columns or 'index' not in output_df.columns:
    raise ValueError("'index' column must be present in both input files.")

# Convert 'prediction' column to string to avoid processing errors
output_df['prediction'] = output_df['prediction'].astype(str)

# Define a function to process each prediction
def process_prediction(prediction):
    try:
        # Use extract_value_unit to get the formatted value and unit
        result = extract_value_unit(prediction)
        return result
    except Exception as e:
        print(f"Error processing prediction: {e}")
        return ""

# Apply the processing function to the 'prediction' column in output_df
output_df['prediction'] = output_df['prediction'].apply(process_prediction)

# Create a DataFrame with only the indices present in test_df
output_df_filtered = output_df[output_df['index'].isin(test_df['index'])]

# Merge the processed predictions with the test DataFrame using the 'index' column
merged_df = pd.merge(test_df, output_df_filtered[['index', 'prediction']], on='index', how='left')

# Fill missing predictions with empty strings where there was no match in output_df
merged_df['prediction'].fillna("", inplace=True)

# Drop the columns 'image_link', 'group_id', and 'entity_name' as requested
merged_df.drop(columns=['image_link', 'group_id', 'entity_name'], inplace=True)

# Save the updated DataFrame to the output CSV, keeping only 'index' and 'prediction'
merged_df[['index', 'prediction']].to_csv(output_csv_path, index=False)

print(f"Processed predictions saved to {output_csv_path}")
