"""
Script tự động thiết lập Kaggle API và tải dataset
"""

import os
import json
import subprocess
import sys


def setup_kaggle_credentials():
    """
    Hướng dẫn cài đặt Kaggle API credentials
    """
    print("\n" + "="*70)
    print("THIẾT LẬP KAGGLE API")
    print("="*70 + "\n")
    
    # Kiểm tra thư mục .kaggle
    home_dir = os.path.expanduser("~")
    kaggle_dir = os.path.join(home_dir, ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("✅ Đã tìm thấy kaggle.json")
        return True
    
    print("⚠️  Chưa tìm thấy file kaggle.json")
    print("\n📝 HƯỚNG DẪN LẤY API TOKEN:")
    print("   1. Truy cập: https://www.kaggle.com/settings")
    print("   2. Scroll xuống mục 'API'")
    print("   3. Click 'Create New Token'")
    print("   4. File kaggle.json sẽ được tải về")
    print("\n📂 Di chuyển file kaggle.json đến:")
    print(f"   {kaggle_dir}/")
    
    print("\n" + "="*70)
    choice = input("\nBạn đã có file kaggle.json chưa? (y/n): ")
    
    if choice.lower() == 'y':
        print("\n📝 Nhập thông tin Kaggle API:")
        username = input("   Username: ").strip()
        key = input("   Key: ").strip()
        
        # Tạo thư mục
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Tạo file kaggle.json
        credentials = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Chmod 600 (chỉ owner đọc/ghi)
        if os.name != 'nt':  # Không phải Windows
            os.chmod(kaggle_json, 0o600)
        
        print(f"\n✅ Đã tạo {kaggle_json}")
        return True
    
    return False


def download_dataset():
    """
    Tải dataset từ Kaggle
    """
    print("\n" + "="*70)
    print("TẢI DATASET: Fruits Fresh and Rotten")
    print("="*70 + "\n")
    
    dataset_name = "sriramr/fruits-fresh-and-rotten-for-classification"
    output_dir = "data/fruits_fresh_rotten"
    
    print(f"📦 Dataset: {dataset_name}")
    print(f"📂 Thư mục đích: {output_dir}/")
    print("\n⏳ Đang tải... (Có thể mất vài phút)")
    
    try:
        # Tạo thư mục
        os.makedirs(output_dir, exist_ok=True)
        
        # Tải dataset
        cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Tải thành công!")
            return True
        else:
            print(f"\n❌ Lỗi: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        return False


def organize_dataset():
    """
    Tổ chức lại cấu trúc dataset
    """
    print("\n" + "="*70)
    print("TỔ CHỨC DATASET")
    print("="*70 + "\n")
    
    base_dir = "data/fruits_fresh_rotten"
    
    # Kiểm tra cấu trúc
    if not os.path.exists(base_dir):
        print(f"❌ Không tìm thấy thư mục {base_dir}/")
        return False
    
    # Liệt kê nội dung
    print("📂 Cấu trúc dataset:")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # Chỉ hiển thị 3 file đầu mỗi thư mục
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files[:3]):
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... và {len(files) - 3} files khác')
    
    # Đếm số ảnh
    fresh_count = 0
    rotten_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        if 'fresh' in root.lower():
            fresh_count += len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
        elif 'rotten' in root.lower() or 'rot' in root.lower():
            rotten_count += len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n📊 Thống kê:")
    print(f"   🍎 Fresh: {fresh_count} ảnh")
    print(f"   🍂 Rotten: {rotten_count} ảnh")
    print(f"   📦 Tổng: {fresh_count + rotten_count} ảnh")
    
    return True


def main():
    """
    Quy trình hoàn chỉnh
    """
    print("\n" + "="*70)
    print("SETUP DATASET - PHÂN LOẠI TRÁI CÂY TƯƠI/HỎNG")
    print("="*70)
    
    # Bước 1: Setup Kaggle
    if not setup_kaggle_credentials():
        print("\n❌ Cần có Kaggle API token để tiếp tục!")
        print("   Tham khảo: DATASET_FRESH_ROTTEN.md")
        return
    
    # Bước 2: Tải dataset
    if not download_dataset():
        print("\n❌ Không thể tải dataset!")
        return
    
    # Bước 3: Tổ chức dataset
    if not organize_dataset():
        print("\n❌ Không thể tổ chức dataset!")
        return
    
    # Hoàn thành
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    
    print("\n🚀 BÂY GIỜ BẠN CÓ THỂ:")
    print("\n   # Test với 1 ảnh")
    print("   python classify_fresh_rotten.py data/fruits_fresh_rotten/train/fresh_apple/apple_0.jpg")
    print("\n   # Test hàng loạt (fresh)")
    print("   python classify_fresh_rotten.py data/fruits_fresh_rotten/train/fresh_apple/ --batch")
    print("\n   # Test hàng loạt (rotten)")
    print("   python classify_fresh_rotten.py data/fruits_fresh_rotten/train/rotten_apple/ --batch")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
