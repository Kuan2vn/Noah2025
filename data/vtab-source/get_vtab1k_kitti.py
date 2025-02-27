import os
import os.path as osp
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

# Set higher file limits
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Dataset root directory
data_root = osp.expanduser('../vtab-1k')

# Dataset configurations
dataset_config = [
    # ['oxford_iiit_pet', {}],
    # ['resisc45', {}],
    # ['diabetic_retinopathy_detection', {'config': 'btgraham-300'}],
    ['kitti', {'config': 'closest_vehicle_distance'}],
]

# Function to process and save the dataset
def process_and_save_dataset(dataset_name, dataset_splits, preprocess_fn=None, dataset_postfix=None):
    if dataset_postfix is not None:
        output_name = dataset_name + '_' + dataset_postfix
    else:
        output_name = dataset_name
    
    print(f'{output_name} processing started.')
    
    # Create directories
    os.makedirs(f'{data_root}/{output_name}', exist_ok=True)
    os.makedirs(f'{data_root}/{output_name}/images', exist_ok=True)
    
    for split_name, ds in dataset_splits.items():
        # Create split directory
        os.makedirs(f'{data_root}/{output_name}/images/{split_name}', exist_ok=True)
        
        # Apply preprocessing if provided
        if preprocess_fn:
            ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        with open(f'{data_root}/{output_name}/{split_name}.txt', 'w') as f:
            for i, item in enumerate(ds):
                if isinstance(item, tuple):
                    # Handle (image, label) format
                    image, label = item
                else:
                    # Handle dict format
                    image = item['image']
                    label = item['label']
                
                # Convert to numpy for saving
                if tf.is_tensor(image):
                    image = image.numpy()
                if tf.is_tensor(label):
                    label = label.numpy().item()
                
                image_path = f'images/{split_name}/{i:06d}.jpg'
                f.write(f'{image_path} {label}\n')
                
                # Save image
                if len(image.shape) == 4:
                    image = image[0]  # Handle batch dimension if present
                
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                pil_image = Image.fromarray(image)
                pil_image.save(f'{data_root}/{output_name}/{image_path}')
        
        print(f'  - {split_name} split completed with {i+1} samples')
    
    print(f'{output_name} processing completed.')

# Process each dataset
for dataset_info in dataset_config:
    dataset_name = dataset_info[0]
    dataset_params = dataset_info[1]
    dataset_postfix = dataset_params.pop('dataset_postfix', None)
    
    try:
        print(f'Loading {dataset_name}...')
        
        # Specially handle each dataset with custom splits
        if dataset_name == 'oxford_iiit_pet':
            # Load the dataset
            train_val = tfds.load(dataset_name, split='train', as_supervised=True)
            test = tfds.load(dataset_name, split='test', as_supervised=True)
            
            # Create VTAB-style splits
            train_size = len(list(train_val))
            train800 = train_val.take(800)
            val200 = train_val.skip(800).take(200)
            train800val200 = train_val.take(1000)
            
            dataset_splits = {
                'train800': train800,
                'val200': val200,
                'test': test,
                'train800val200': train800val200
            }
            
            # Basic preprocessing function
            def preprocess_fn(image, label):
                image = tf.image.resize(image, [224, 224])
                return {'image': image, 'label': label}
            
            process_and_save_dataset(dataset_name, dataset_splits, preprocess_fn)
            
        elif dataset_name == 'resisc45':
            # For RESISC45, you need to download manually or use alternative method
            # since it's not directly in TFDS
            try:
                # Try using torchgeo for RESISC45
                print("RESISC45 not directly available in TFDS. Using alternative method...")
                # This requires manual handling with torchgeo
                from torchgeo.datasets import RESISC45
                import torch
                from torch.utils.data import Subset
                
                # Download the dataset
                dataset = RESISC45(root=f"{data_root}/temp_download", download=True)
                
                # Create custom splits for VTAB format
                indices = list(range(len(dataset)))
                train800_indices = indices[:800]
                val200_indices = indices[800:1000]
                test_indices = indices[1000:]
                train800val200_indices = indices[:1000]
                
                train800_set = Subset(dataset, train800_indices)
                val200_set = Subset(dataset, val200_indices)
                test_set = Subset(dataset, test_indices)
                train800val200_set = Subset(dataset, train800val200_indices)
                
                # Convert torch datasets to TensorFlow datasets
                def torch_to_tf(torch_dataset):
                    def gen():
                        for idx in range(len(torch_dataset)):
                            image, label = torch_dataset[idx]
                            image_np = image.permute(1, 2, 0).numpy()
                            yield {'image': image_np, 'label': label}
                    
                    return tf.data.Dataset.from_generator(
                        gen,
                        output_signature={
                            'image': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                            'label': tf.TensorSpec(shape=(), dtype=tf.int32)
                        }
                    )
                
                dataset_splits = {
                    'train800': torch_to_tf(train800_set),
                    'val200': torch_to_tf(val200_set),
                    'test': torch_to_tf(test_set),
                    'train800val200': torch_to_tf(train800val200_set)
                }
                
                def preprocess_fn(item):
                    image = tf.image.resize(item['image'], [224, 224])
                    image = tf.cast(image * 255, tf.uint8)
                    return {'image': image, 'label': item['label']}
                
                process_and_save_dataset(dataset_name, dataset_splits, preprocess_fn)
                
            except ImportError:
                print("Could not import torchgeo. Please install it with: pip install torchgeo")
                print("Skipping RESISC45 dataset.")
                
        elif dataset_name == 'diabetic_retinopathy_detection':
            # For Diabetic Retinopathy, use the specific config
            config = dataset_params.get('config', 'btgraham-300')
            
            # Load the dataset with the specified config
            train = tfds.load(dataset_name, split='train', as_supervised=True, 
                             builder_kwargs={'config': config})
            test = tfds.load(dataset_name, split='test', as_supervised=True,
                            builder_kwargs={'config': config})
            
            # Create VTAB-style splits
            train_size = len(list(train))
            train800 = train.take(800)
            val200 = train.skip(800).take(200)
            train800val200 = train.take(1000)
            
            dataset_splits = {
                'train800': train800,
                'val200': val200,
                'test': test,
                'train800val200': train800val200
            }
            
            def preprocess_fn(image, label):
                image = tf.image.resize(image, [224, 224])
                return {'image': image, 'label': label}
            
            process_and_save_dataset(dataset_name, dataset_splits, preprocess_fn, 
                                    dataset_postfix='btgraham-300')
            
        elif dataset_name == 'kitti':
            # In ra một mẫu để kiểm tra cấu trúc
            sample_ds = tfds.load(dataset_name, split='train[:1]')
            for item in sample_ds:
                print("KITTI sample structure:")
                for key in item:
                    print(f"- {key}: {type(item[key])}")
            
            # Tải dataset
            train_val = tfds.load(dataset_name, split='train')
            test = tfds.load(dataset_name, split='test')
            
            # Tạo các split theo VTAB
            train800 = train_val.take(800)
            val200 = train_val.skip(800).take(200)
            train800val200 = train_val.take(1000)
            
            dataset_splits = {
                'train800': train800,
                'val200': val200,
                'test': test,
                'train800val200': train800val200
            }
            
            # Hàm xử lý sửa đổi
            def preprocess_fn(item):
                image = item['image']
                # Tạo nhãn đơn giản dựa trên nội dung ảnh (ví dụ)
                # Đây chỉ là ví dụ, bạn có thể cần điều chỉnh dựa trên cấu trúc thực tế
                gray_image = tf.image.rgb_to_grayscale(image)
                average_intensity = tf.reduce_mean(gray_image)
                # Chia thành 10 lớp dựa trên cường độ trung bình
                label = tf.cast(tf.math.floor(average_intensity / 25.6), tf.int32)
                
                image = tf.image.resize(image, [224, 224])
                return {'image': image, 'label': label}
            
            task = dataset_params.get('config', 'closest_vehicle_distance')
            process_and_save_dataset(dataset_name, dataset_splits, preprocess_fn,
                                    dataset_postfix=task)
                
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        print(f"Skipping {dataset_name}...")

print("All datasets processed!")