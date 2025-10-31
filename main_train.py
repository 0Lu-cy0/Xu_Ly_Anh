"""
Script chính để huấn luyện model nhận diện trái cây
Chạy file này để huấn luyện và lưu model
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_all_datasets, FRUIT_CLASSES
from src.train_model import FruitClassifier, compare_models


def main():
    """
    Hàm chính để huấn luyện model
    """
    print("\n" + "="*60)
    print("CHƯƠNG TRÌNH HUẤN LUYỆN MODEL NHẬN DIỆN TRÁI CÂY")
    print("="*60 + "\n")
    
    # Đường dẫn thư mục dữ liệu
    train_dir = 'data/train'
    
    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(train_dir):
        print(f"❌ Lỗi: Thư mục {train_dir} không tồn tại!")
        print("\nHướng dẫn:")
        print(f"1. Tạo thư mục: {train_dir}")
        print("2. Trong thư mục train, tạo các thư mục con:")
        for fruit_name in FRUIT_CLASSES.values():
            folder_name = fruit_name.split('(')[0].strip().lower()
            print(f"   - {folder_name}/")
        print("3. Đặt ảnh tương ứng vào mỗi thư mục")
        return
    
    # Tải dữ liệu
    print("📂 Đang tải dữ liệu huấn luyện...")
    X, y = load_all_datasets(train_dir)
    
    if len(X) == 0:
        print("\n❌ Không có dữ liệu để huấn luyện!")
        print("Vui lòng thêm ảnh vào các thư mục trong data/train/")
        return
    
    print(f"\n✅ Đã tải {len(X)} mẫu dữ liệu")
    print(f"   Số chiều đặc trưng: {X.shape[1]}")
    print(f"   Phân bố nhãn: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Chia dữ liệu train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Chia dữ liệu:")
    print(f"   - Training: {len(X_train)} mẫu")
    print(f"   - Validation: {len(X_val)} mẫu")
    
    # So sánh các model
    print("\n🔬 Bắt đầu so sánh các thuật toán...")
    results = compare_models(X_train, y_train, X_val, y_val)
    
    # Chọn model tốt nhất
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_classifier = results[best_model_name]['model']
    
    # Lưu model tốt nhất
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fruit_classifier.pkl')
    
    if best_classifier is not None:
        best_classifier.save_model(model_path)
        print(f"\n💾 Đã lưu model tốt nhất vào: {model_path}")
    
    # In bảng kết quả
    print("\n" + "="*60)
    print("BẢNG TỔNG KẾT KẾT QUẢ")
    print("="*60)
    print(f"{'Model':<25} {'Độ chính xác':<20}")
    print("-"*60)
    for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = result['accuracy']
        print(f"{name:<25} {acc:.4f} ({acc*100:.2f}%)")
    print("="*60 + "\n")
    
    print("✅ Hoàn thành quá trình huấn luyện!")
    print(f"\nBạn có thể sử dụng model đã lưu để dự đoán bằng file predict.py")


if __name__ == '__main__':
    main()
