"""
Script tự động tổ chức ảnh từ Fruits 360 Dataset
Chạy sau khi đã tải và giải nén Fruits 360
"""

import os
import shutil
from pathlib import Path


def organize_fruits360_dataset():
    """
    Tổ chức lại ảnh từ Fruits 360 vào cấu trúc dự án
    """
    
    print("\n" + "="*70)
    print("SCRIPT TỔ CHỨC FRUITS 360 DATASET")
    print("="*70 + "\n")
    
    # Mapping từ tên trong Fruits360 sang tên trong dự án
    fruit_mapping = {
        'táo': [
            'Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 
            'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith',
            'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
            'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2'
        ],
        'chuối': [
            'Banana', 'Banana Lady Finger', 'Banana Red'
        ],
        'cam': [
            'Orange'
        ],
        'nho': [
            'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2',
            'Grape White 3', 'Grape White 4'
        ],
        'dưa hấu': [
            'Watermelon'
        ]
    }
    
    # Đường dẫn (có thể cần điều chỉnh)
    possible_source_dirs = [
        'fruits-360_dataset/fruits-360/Training',
        'fruits-360/Training',
        'Training',
        'temp_fruits360/Training',
        'fruits/Training'
    ]
    
    # Tìm thư mục nguồn
    source_dir = None
    for possible_dir in possible_source_dirs:
        if os.path.exists(possible_dir):
            source_dir = possible_dir
            break
    
    if source_dir is None:
        print("❌ Không tìm thấy thư mục Fruits 360!")
        print("\nVui lòng:")
        print("1. Tải Fruits 360 từ: https://www.kaggle.com/datasets/moltean/fruits")
        print("2. Giải nén file zip")
        print("3. Đảm bảo có thư mục 'Training' trong thư mục giải nén")
        print("\nHoặc chỉnh sửa đường dẫn trong script này.")
        return
    
    print(f"✅ Đã tìm thấy thư mục nguồn: {source_dir}\n")
    
    # Tạo thư mục đích
    target_base_train = 'data/train'
    target_base_test = 'data/test'
    
    os.makedirs(target_base_train, exist_ok=True)
    os.makedirs(target_base_test, exist_ok=True)
    
    # Thống kê
    total_copied = 0
    stats = {}
    
    # Xử lý mỗi loại trái cây
    for target_fruit, source_fruits in fruit_mapping.items():
        target_path_train = os.path.join(target_base_train, target_fruit)
        target_path_test = os.path.join(target_base_test, target_fruit)
        
        os.makedirs(target_path_train, exist_ok=True)
        os.makedirs(target_path_test, exist_ok=True)
        
        fruit_count = 0
        
        print(f"📂 Đang xử lý: {target_fruit}")
        
        for source_fruit in source_fruits:
            source_path = os.path.join(source_dir, source_fruit)
            
            if not os.path.exists(source_path):
                print(f"   ⚠️  Không tìm thấy: {source_fruit}")
                continue
            
            # Lấy danh sách ảnh
            image_files = [f for f in os.listdir(source_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) == 0:
                print(f"   ⚠️  Không có ảnh trong: {source_fruit}")
                continue
            
            # Chia 80% train, 20% test
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # Copy ảnh train
            for img_file in train_files:
                src = os.path.join(source_path, img_file)
                # Đổi tên để không trùng
                new_name = f"{source_fruit.replace(' ', '_')}_{img_file}"
                dst = os.path.join(target_path_train, new_name)
                shutil.copy2(src, dst)
                fruit_count += 1
            
            # Copy ảnh test
            for img_file in test_files:
                src = os.path.join(source_path, img_file)
                new_name = f"{source_fruit.replace(' ', '_')}_{img_file}"
                dst = os.path.join(target_path_test, new_name)
                shutil.copy2(src, dst)
            
            print(f"   ✅ {source_fruit}: {len(train_files)} train, {len(test_files)} test")
        
        stats[target_fruit] = fruit_count
        total_copied += fruit_count
        print(f"   📊 Tổng {target_fruit}: {fruit_count} ảnh\n")
    
    # In thống kê
    print("="*70)
    print("📊 THỐNG KÊ DATASET")
    print("="*70)
    print(f"\n{'Loại trái cây':<20} {'Số ảnh train':<20}")
    print("-"*70)
    for fruit, count in stats.items():
        print(f"{fruit:<20} {count:<20}")
    print("-"*70)
    print(f"{'TỔNG CỘNG':<20} {total_copied:<20}")
    print("="*70 + "\n")
    
    print("✅ HOÀN THÀNH!\n")
    print("Bước tiếp theo:")
    print("   python main_train.py  # Để huấn luyện model\n")
    
    # Kiểm tra xem có thư mục Test trong Fruits360 không
    test_source_dir = source_dir.replace('Training', 'Test')
    if os.path.exists(test_source_dir):
        print("💡 Mẹo: Bạn cũng có thể copy ảnh từ thư mục 'Test' của Fruits360")
        print(f"   Đường dẫn: {test_source_dir}\n")


def check_dataset_structure():
    """
    Kiểm tra cấu trúc dataset sau khi tổ chức
    """
    print("\n" + "="*70)
    print("KIỂM TRA CẤU TRÚC DATASET")
    print("="*70 + "\n")
    
    for split in ['train', 'test']:
        base_dir = f'data/{split}'
        if not os.path.exists(base_dir):
            print(f"❌ Thư mục {base_dir} không tồn tại!")
            continue
        
        print(f"📁 {split.upper()}:")
        for fruit_folder in os.listdir(base_dir):
            fruit_path = os.path.join(base_dir, fruit_folder)
            if os.path.isdir(fruit_path):
                num_images = len([f for f in os.listdir(fruit_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   {fruit_folder:<15} : {num_images:>4} ảnh")
        print()


if __name__ == '__main__':
    try:
        # Chạy script tổ chức
        organize_fruits360_dataset()
        
        # Kiểm tra kết quả
        check_dataset_structure()
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Gợi ý:")
        print("1. Đảm bảo đã tải và giải nén Fruits 360 Dataset")
        print("2. Kiểm tra đường dẫn trong script")
        print("3. Đảm bảo có quyền ghi vào thư mục data/")
