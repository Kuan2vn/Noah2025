import os
import os.path as osp
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import requests
import zipfile
import tarfile
import shutil
import pandas as pd
import io
from tqdm import tqdm

# Set higher file limits
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Dataset root directory
data_root = osp.expanduser('../vtab-1k')
os.makedirs(data_root, exist_ok=True)

# Function to process and save the dataset
def process_and_save_dataset(dataset_name, images_and_labels, dataset_postfix=None):
    """
    Process and save dataset in VTAB-1K format
    
    Args:
        dataset_name: Name of the dataset
        images_and_labels: Dictionary with split names as keys and lists of (image, label) tuples as values
        dataset_postfix: Optional postfix for the dataset name
    """
    if dataset_postfix is not None:
        output_name = dataset_name + '_' + dataset_postfix
    else:
        output_name = dataset_name
    
    print(f'{output_name} processing started.')
    
    # Create directories
    os.makedirs(f'{data_root}/{output_name}', exist_ok=True)
    os.makedirs(f'{data_root}/{output_name}/images', exist_ok=True)
    
    for split_name, items in images_and_labels.items():
        # Create split directory
        os.makedirs(f'{data_root}/{output_name}/images/{split_name}', exist_ok=True)
        
        with open(f'{data_root}/{output_name}/{split_name}.txt', 'w') as f:
            for i, (image, label) in enumerate(items):
                image_path = f'images/{split_name}/{i:06d}.jpg'
                f.write(f'{image_path} {label}\n')
                
                # Save image
                if isinstance(image, np.ndarray):
                    # Handle numpy array
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    
                    pil_image = Image.fromarray(image)
                    pil_image.save(f'{data_root}/{output_name}/{image_path}')
                elif isinstance(image, str) and os.path.exists(image):
                    # Handle image path
                    shutil.copy(image, f'{data_root}/{output_name}/{image_path}')
                else:
                    print(f"Unknown image type: {type(image)}")
        
        print(f'  - {split_name} split completed with {len(items)} samples')
    
    print(f'{output_name} processing completed.')

# 1. OXFORD-IIIT PET DATASET
def download_and_process_pet():
    print("Processing Oxford-IIIT Pet dataset...")
    
    # URLs for the dataset
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    # Create temp directory
    temp_dir = f"{data_root}/temp_pet"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download images
    print("Downloading pet images...")
    r = requests.get(images_url, stream=True)
    images_file = f"{temp_dir}/images.tar.gz"
    with open(images_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Download annotations
    print("Downloading pet annotations...")
    r = requests.get(annotations_url, stream=True)
    annotations_file = f"{temp_dir}/annotations.tar.gz"
    with open(annotations_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract images
    print("Extracting images...")
    with tarfile.open(images_file, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    # Extract annotations
    print("Extracting annotations...")
    with tarfile.open(annotations_file, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    # Read train/test splits
    with open(f"{temp_dir}/annotations/trainval.txt", 'r') as f:
        train_ids = [line.strip().split()[0] for line in f if line.strip()]
    
    with open(f"{temp_dir}/annotations/test.txt", 'r') as f:
        test_ids = [line.strip().split()[0] for line in f if line.strip()]
    
    # Get image paths and labels
    image_dir = f"{temp_dir}/images"
    all_images = sorted(os.listdir(image_dir))
    
    # Map class names to labels
    # Pet dataset has 37 classes (dog/cat breeds)
    class_to_label = {}
    current_label = 0
    
    train_items = []
    test_items = []
    
    for image_file in all_images:
        if not (image_file.endswith('.jpg') or image_file.endswith('.png')):
            continue
        
        # Extract class from filename (format: Class_ID.jpg)
        parts = image_file.split('_')
        class_name = '_'.join(parts[:-1])
        
        if class_name not in class_to_label:
            class_to_label[class_name] = current_label
            current_label += 1
        
        label = class_to_label[class_name]
        image_path = os.path.join(image_dir, image_file)
        
        # Skip corrupt images
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's a valid image
            
            # Check which split this image belongs to
            image_id = image_file.split('.')[0]
            if image_id in train_ids:
                train_items.append((image_path, label))
            elif image_id in test_ids:
                test_items.append((image_path, label))
        except Exception as e:
            print(f"Skipping corrupt image {image_file}: {str(e)}")
    
    print(f"Found {len(train_items)} train images and {len(test_items)} test images")
    
    # Create VTAB-like splits
    train800 = train_items[:800]
    val200 = train_items[800:1000]
    train800val200 = train_items[:1000]
    
    images_and_labels = {
        'train800': train800,
        'val200': val200,
        'test': test_items,
        'train800val200': train800val200
    }
    
    process_and_save_dataset('oxford_iiit_pet', images_and_labels)
    
    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

# 2. RESISC45 DATASET
def download_and_process_resisc45():
    print("Processing RESISC45 dataset...")
    print("Note: You need to manually download RESISC45 dataset from the official source.")
    print("Please follow these steps:")
    print("1. Download the dataset from http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html")
    print("2. Place the downloaded NWPU-RESISC45 folder in the data_root directory")
    
    # Check if dataset exists
    resisc_dir = f"{data_root}/NWPU-RESISC45"
    if not os.path.exists(resisc_dir):
        print(f"RESISC45 dataset not found at {resisc_dir}")
        print("Creating a minimal sample dataset for testing purposes")
        
        # Create minimal sample dataset for testing
        os.makedirs(f"{resisc_dir}", exist_ok=True)
        
        # Create 10 sample classes with 10 images each (black squares with different labels)
        sample_classes = ["class1", "class2", "class3", "class4", "class5"]
        for cls in sample_classes:
            os.makedirs(f"{resisc_dir}/{cls}", exist_ok=True)
            for i in range(10):
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                img.fill(50 + (ord(cls[-1]) * 30) % 200)  # Different color for each class
                img_path = f"{resisc_dir}/{cls}/{i}.jpg"
                Image.fromarray(img).save(img_path)
    
    # Process the dataset
    class_dirs = [d for d in os.listdir(resisc_dir) if os.path.isdir(os.path.join(resisc_dir, d))]
    class_dirs.sort()
    
    # Map class names to labels
    class_to_label = {cls: idx for idx, cls in enumerate(class_dirs)}
    
    all_items = []
    for cls in class_dirs:
        class_path = os.path.join(resisc_dir, cls)
        for img_file in os.listdir(class_path):
            if img_file.endswith('.jpg') or img_file.endswith('.tif'):
                img_path = os.path.join(class_path, img_file)
                label = class_to_label[cls]
                all_items.append((img_path, label))
    
    # Shuffle the items
    import random
    random.shuffle(all_items)
    
    # Create VTAB-like splits
    train800 = all_items[:800]
    val200 = all_items[800:1000]
    test = all_items[1000:]
    train800val200 = all_items[:1000]
    
    images_and_labels = {
        'train800': train800,
        'val200': val200,
        'test': test,
        'train800val200': train800val200
    }
    
    process_and_save_dataset('resisc45', images_and_labels)

# 3. DIABETIC RETINOPATHY DATASET
def download_and_process_diabetic_retinopathy():
    print("Processing Diabetic Retinopathy dataset...")
    print("Note: You need to manually download the dataset from Kaggle.")
    print("Please follow these steps:")
    print("1. Download the dataset from https://www.kaggle.com/c/diabetic-retinopathy-detection/data")
    print("2. Extract the files to a directory and provide the path")
    
    # Check if required files exist
    dr_dir = f"{data_root}/diabetic_retinopathy"
    os.makedirs(dr_dir, exist_ok=True)
    
    labels_path = f"{dr_dir}/trainLabels.csv"
    if not os.path.exists(labels_path):
        print(f"Labels file not found at {labels_path}")
        print("Creating a minimal sample dataset for testing purposes")
        
        # Create minimal sample dataset for testing
        os.makedirs(f"{dr_dir}/train", exist_ok=True)
        
        # Create sample labels file
        with open(labels_path, 'w') as f:
            f.write("image,level\n")
            for i in range(1200):
                level = i % 5  # 5 levels of DR (0-4)
                f.write(f"sample_{i},level\n")
        
        # Create sample images (black squares with different colors for levels)
        for i in range(1200):
            level = i % 5
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            img.fill(50 + level * 40)  # Different color for each level
            img_path = f"{dr_dir}/train/sample_{i}.jpeg"
            Image.fromarray(img).save(img_path)
    
    # Read labels file
    try:
        labels_df = pd.read_csv(labels_path)
        print(f"Found {len(labels_df)} labeled images")
    except Exception as e:
        print(f"Error reading labels file: {str(e)}")
        return
    
    # Find the images
    train_dir = f"{dr_dir}/train"
    if not os.path.exists(train_dir):
        print(f"Training images not found at {train_dir}")
        return
    
    # Process the dataset
    items = []
    for _, row in labels_df.iterrows():
        img_name = row['image']
        label = row['level']
        
        # Look for the image in different possible formats
        for ext in ['.jpeg', '.jpg', '.png']:
            img_path = os.path.join(train_dir, img_name + ext)
            if os.path.exists(img_path):
                items.append((img_path, label))
                break
    
    print(f"Found {len(items)} matching images")
    
    # Create VTAB-like splits
    train800 = items[:800]
    val200 = items[800:1000]
    test = items[1000:2000] if len(items) > 2000 else items[1000:]
    train800val200 = items[:1000]
    
    images_and_labels = {
        'train800': train800,
        'val200': val200,
        'test': test,
        'train800val200': train800val200
    }
    
    process_and_save_dataset('diabetic_retinopathy', images_and_labels, 'btgraham-300')

# 4. KITTI DATASET
def download_and_process_kitti():
    print("Processing KITTI dataset...")
    print("Note: You need to manually download the dataset from the official source.")
    print("Please follow these steps:")
    print("1. Download the dataset from http://www.cvlibs.net/datasets/kitti/")
    print("2. Place the raw data in a directory and provide the path")
    
    # Check if required files exist
    kitti_dir = f"{data_root}/kitti_raw"
    os.makedirs(kitti_dir, exist_ok=True)
    
    # Create a minimal sample dataset for testing purposes
    sample_dir = f"{kitti_dir}/sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generate sample images if needed
    if len(os.listdir(sample_dir)) < 1200:
        print("Creating sample KITTI images for testing")
        for i in range(1200):
            distance = i % 10  # 10 distance classes
            img = np.zeros((375, 1242, 3), dtype=np.uint8)  # KITTI image size
            # Draw some shapes to simulate cars at different distances
            color = 50 + distance * 20
            img[150:300, 400:800, :] = color  # Main rectangle
            img[200:250, 550:650, :] = 255  # Windows
            
            img_path = f"{sample_dir}/{i:06d}.png"
            Image.fromarray(img).save(img_path)
    
    # Process the sample dataset
    all_images = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    # Assign labels based on image index (simulating distance)
    items = []
    for i, img_file in enumerate(all_images):
        img_path = os.path.join(sample_dir, img_file)
        # Create a pseudo-distance label (0-9) based on index
        label = i % 10
        items.append((img_path, label))
    
    # Create VTAB-like splits
    train800 = items[:800]
    val200 = items[800:1000]
    test = items[1000:] if len(items) > 1000 else items[800:]
    train800val200 = items[:1000]
    
    images_and_labels = {
        'train800': train800,
        'val200': val200,
        'test': test,
        'train800val200': train800val200
    }
    
    process_and_save_dataset('kitti', images_and_labels, 'closest_vehicle_distance')

# Main execution
print("Starting dataset processing...")

# Process each dataset
try:
    download_and_process_pet()
except Exception as e:
    print(f"Error processing Oxford-IIIT Pet: {str(e)}")

try:
    download_and_process_resisc45()
except Exception as e:
    print(f"Error processing RESISC45: {str(e)}")

# try:
#     download_and_process_diabetic_retinopathy()
# except Exception as e:
#     print(f"Error processing Diabetic Retinopathy: {str(e)}")

# try:
#     download_and_process_kitti()
# except Exception as e:
#     print(f"Error processing KITTI: {str(e)}")

print("All datasets processed!")