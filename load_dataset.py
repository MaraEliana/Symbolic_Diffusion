import json
import os

def load_diffusion_data(input_dir="diffusion_data", file_name="train.json"):
    """Load processed diffusion data from the specified JSON file."""
    file_path = os.path.join(input_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {str(e)}")
    
    print(f"Loaded {len(data)} records from {file_path}")
    return data