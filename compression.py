import numpy as np
import os
from pathlib import Path

# Paths
old_path = "./results_old"
new_path = "./results"

print(f"Reading from: {old_path}")
print(f"Writing to:   {new_path}")

# Walk through the old results
for root, dirs, files in os.walk(old_path):
    for file in files:
        if file.endswith(".npz"):
            # 1. Construct the old file path
            old_file_path = Path(root) / file
            
            # 2. Determine the relative path to maintain folder structure
            relative_path = old_file_path.relative_to(old_path)
            new_file_path = Path(new_path) / relative_path
            
            # 3. Create the subdirectories in the new results folder
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # 4. Load from old
                with np.load(old_file_path) as data:
                    # Convert to float32
                    new_returns = np.array(data['returns'], dtype=np.float32)
                    new_ratios = np.array(data['ratios'], dtype=np.float32)
                
                # 5. Save compressed to new location
                np.savez_compressed(new_file_path, returns=new_returns, ratios=new_ratios)
                
            except Exception as e:
                print(f"Error processing {old_file_path}: {e}")

print("\n--- Process Complete ---")
print(f"Old size: ", end="")
os.system(f"du -sh {old_path}")
print(f"New size: ", end="")
os.system(f"du -sh {new_path}")