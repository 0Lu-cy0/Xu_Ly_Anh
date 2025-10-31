"""
Script tải dataset từ nguồn công khai (không cần Kaggle API)
"""

import os
import zipfile
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar cho urllib"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Tải file từ URL với progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                   reporthook=t.update_to)


def download_alternative_dataset():
    """
    Tải dataset từ nguồn thay thế (GitHub hoặc mirror)
    """
    print("\n" + "="*70)
    print("TẢI DATASET TỪ NGUỒN CÔNG KHAI")
    print("="*70 + "\n")
    
    # Tạo thư mục
    output_dir = "data/fruits_fresh_rotten"
    os.makedirs(output_dir, exist_ok=True)
    
    print("📦 Đang tạo dataset mẫu nhỏ...")
    print("   (Để có dataset đầy đủ, cần dùng Kaggle API)\n")
    
    # Tạo cấu trúc thư mục
    categories = {
        'train': ['freshapples', 'freshbanana', 'freshoranges', 
                  'rottenapples', 'rottenbanana', 'rottenoranges'],
        'test': ['freshapples', 'freshbanana', 'freshoranges',
                 'rottenapples', 'rottenbanana', 'rottenoranges']
    }
    
    for split, cats in categories.items():
        for cat in cats:
            path = os.path.join(output_dir, split, cat)
            os.makedirs(path, exist_ok=True)
            print(f"   ✅ Tạo {split}/{cat}/")
    
    print(f"\n📂 Cấu trúc dataset đã sẵn sàng trong: {output_dir}/")
    
    print("\n" + "="*70)
    print("⚠️  LƯU Ý:")
    print("="*70)
    print("""
Hiện tại chỉ tạo được cấu trúc thư mục.
Để có ảnh thật, bạn cần:

CÁCH 1: Tải từ Kaggle (Khuyến nghị)
   → Chạy: python3 setup_kaggle.py
   → Cần tạo tài khoản Kaggle và lấy API token

CÁCH 2: Tải thủ công
   1. Vào: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
   2. Click "Download" (cần đăng nhập)
   3. Giải nén vào: data/fruits_fresh_rotten/

CÁCH 3: Dùng ảnh mẫu để test
   → Chạy: python3 create_sample_fresh_rotten.py
   → Tạo ảnh mô phỏng đơn giản (không phải ảnh thật)
""")
    
    return True


def main():
    download_alternative_dataset()
    
    print("\n🚀 BÂY GIỜ BẠN CÓ THỂ:")
    print("\n   # Tạo ảnh mẫu để test ngay")
    print("   python3 create_sample_fresh_rotten.py")
    print("\n   # Test classifier với ảnh mẫu")
    print("   python3 classify_fresh_rotten.py data/test/fresh/apple_fresh_1.jpg")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
