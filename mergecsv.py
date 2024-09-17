import pandas as pd

# List of new CSV file paths
csv_files = [
            'output30000_40000.csv',
            'output20000_30000.csv',
            'output15000_20000.csv',
            'output10000_15000.csv',
            'output.csv']


# Load each CSV into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Merge the new DataFrames (concatenate them)
new_merged_df = pd.concat(dfs, ignore_index=True)

# Load the existing merged_output.csv if it exists
existing_merged_df = pd.read_csv('output.csv')

# Concatenate the existing and new data
final_merged_df = pd.concat([existing_merged_df, new_merged_df], ignore_index=True)

# Drop duplicate rows based on the 'Index' column
final_merged_df.drop_duplicates(subset='index', keep='first', inplace=True)

# Save the final merged DataFrame to a new CSV file or overwrite the existing one
final_merged_df.to_csv('output.csv', index=False)

print("Merged CSV file saved as 'merged_output.csv', avoiding duplicate indices.")