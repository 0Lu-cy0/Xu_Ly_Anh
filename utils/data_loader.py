"""
Module quản lý dữ liệu
Chứa các hàm để tải, xử lý và chuẩn bị dữ liệu huấn luyện
"""

import os
import cv2
import numpy as np
from utils.image_processing import extract_all_features

# Định nghĩa các loại trái cây
FRUIT_CLASSES = {
    0: 'Táo (Apple)',
    1: 'Chuối (Banana)', 
    2: 'Cam (Orange)',
    3: 'Nho (Grape)',
    4: 'Dưa hấu (Watermelon)'
}


def load_dataset(data_dir, label):
    """
    Tải tất cả ảnh từ một thư mục và gán nhãn
    
    Args:
        data_dir (str): Đường dẫn thư mục chứa ảnh
        label (int): Nhãn của loại trái cây
    
    Returns:
        tuple: (features, labels) - danh sách đặc trưng và nhãn
    """
    features = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Cảnh báo: Thư mục {data_dir} không tồn tại")
        return np.array([]), np.array([])
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Đang xử lý {len(image_files)} ảnh từ {data_dir}...")
    
    for filename in image_files:
        try:
            filepath = os.path.join(data_dir, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                print(f"Không thể đọc {filename}")
                continue
            
            # Trích xuất đặc trưng
            feat = extract_all_features(image)
            features.append(feat)
            labels.append(label)
            
        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {str(e)}")
            continue
    
    return np.array(features), np.array(labels)


def load_all_datasets(base_dir):
    """
    Tải tất cả dữ liệu từ các thư mục con
    
    Args:
        base_dir (str): Thư mục gốc chứa các thư mục con theo loại trái cây
    
    Returns:
        tuple: (X, y) - ma trận đặc trưng và vector nhãn
    """
    all_features = []
    all_labels = []
    
    # Duyệt qua các thư mục con
    for label, fruit_name in FRUIT_CLASSES.items():
        # Tạo tên thư mục (bỏ phần tiếng Việt và dấu ngoặc)
        folder_name = fruit_name.split('(')[0].strip().lower()
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.exists(folder_path):
            features, labels = load_dataset(folder_path, label)
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                print(f"Đã tải {len(features)} mẫu cho {fruit_name}")
        else:
            print(f"Thư mục {folder_path} không tồn tại")
    
    if len(all_features) == 0:
        print("Không có dữ liệu để tải!")
        return np.array([]), np.array([])
    
    # Kết hợp tất cả dữ liệu
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y


def create_sample_data_structure():
    """
    Tạo cấu trúc thư mục mẫu cho dữ liệu
    """
    base_dirs = ['data/train', 'data/test']
    
    for base_dir in base_dirs:
        for fruit_name in FRUIT_CLASSES.values():
            folder_name = fruit_name.split('(')[0].strip().lower()
            folder_path = os.path.join(base_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
    
    print("Đã tạo cấu trúc thư mục:")
    print("data/")
    print("├── train/")
    print("│   ├── táo/")
    print("│   ├── chuối/")
    print("│   ├── cam/")
    print("│   ├── nho/")
    print("│   └── dưa hấu/")
    print("└── test/")
    print("    ├── táo/")
    print("    ├── chuối/")
    print("    ├── cam/")
    print("    ├── nho/")
    print("    └── dưa hấu/")
