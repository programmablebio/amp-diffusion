import pandas as pd
import os
from transformers import AutoModelForCausalLM
from evaluate import load
import numpy as np

# Load perplexity metric
perplexity = load("perplexity", module_type="metric")

# List of files to process
files = [
    "amp_apex.csv",
    "ampgan_results.csv",
    "hydramp_results.csv",
    "pepcvae_results.csv",
    "ampdiffusion.csv"
]

# Process each file
for file_name in files:
    print(f"Processing {file_name}...")
    
    # Construct full file path
    file_path = os.path.join('data', file_name)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Assume the sequence column is named 'sequence'
    sequence_column = 'sequence'
    
    # Check if the sequence column exists
    if sequence_column not in df.columns:
        print(f"Warning: '{sequence_column}' column not found in {file_name}. Skipping this file.")
        continue
    
    # Get sequences from the specified column
    sequences = df[sequence_column].tolist()
    
    # Calculate perplexity
    results = perplexity.compute(predictions=sequences, model_id='hugohrban/progen2-medium')
    
    # Add perplexity scores to the DataFrame
    df['ppl'] = results['perplexities']
    
    # Save the results
    output_file = os.path.join('data', f"{os.path.splitext(file_name)[0]}.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Calculate and report mean PPL and standard deviation
    mean_ppl = np.mean(df['ppl'])
    std_ppl = np.std(df['ppl'])
    print(f"Mean PPL for {file_name}: {mean_ppl:.2f} ± {std_ppl:.2f}")

print("All files processed and saved.")