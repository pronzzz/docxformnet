import os
from datasets import load_dataset
import json
from tqdm import tqdm

def save_dataset(dataset_name, split, output_dir):
    """Downloads and saves a dataset to the specified directory in a standard JSON format."""
    print(f"Downloading {dataset_name} ({split})...")
    
    # Load dataset using Hugging Face datasets
    # CORD: receipts
    # FUNSD: forms
    if dataset_name == 'nielsr/cord-v2':
        try:
             ds = load_dataset(dataset_name, split=split)
        except Exception:
             print(f"Failed to load {dataset_name}, trying naver-clova-ix/cord-v2")
             ds = load_dataset('naver-clova-ix/cord-v2', split=split)
    elif dataset_name == 'nielsr/funsd':
        ds = load_dataset(dataset_name, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)
    
    data_list = []
    print(f"Processing {dataset_name} ({split})...")
    for item in tqdm(ds):
        # standardize format
        # image_path will be handled separately (we might need to download images if not included directly as PIL)
        # For now, HF datasets usually provide 'image' as a PIL object. 
        # We will save metadata and potentially the image if needed for visualization context.
        
        # Structure for CORD/FUNSD usually includes: words, bboxes, labels/ner_tags
        
        entry = {
            'id': item.get('id', str(len(data_list))),
            'words': item['words'],
            'bboxes': item['bboxes'],
            'ner_tags': item['ner_tags'],
            # 'image_path': ... # todo: handle image saving if we want local referencing
        }
        
        # Save image for visualization/debugging
        image = item['image']
        image_filename = f"{entry['id']}.png"
        image_path = os.path.join(output_dir, "images", image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)
        entry['image_path'] = image_path
        
        data_list.append(entry)

    json_path = os.path.join(output_dir, f"{split}.json")
    with open(json_path, 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved {len(data_list)} entries to {json_path}")

if __name__ == "__main__":
    base_data_dir = "data"
    
    # Download CORD
    save_dataset('nielsr/cord-v2', 'train', os.path.join(base_data_dir, 'CORD'))
    save_dataset('nielsr/cord-v2', 'test', os.path.join(base_data_dir, 'CORD')) # CORD uses 'test' or 'validation' depending on version, HF checks usually handle it
    save_dataset('nielsr/cord-v2', 'validation', os.path.join(base_data_dir, 'CORD'))

    # Download FUNSD
    save_dataset('nielsr/funsd', 'train', os.path.join(base_data_dir, 'FUNSD'))
    save_dataset('nielsr/funsd', 'test', os.path.join(base_data_dir, 'FUNSD'))
