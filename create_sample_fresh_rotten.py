"""
Script tạo ảnh mẫu đơn giản để demo
CHỈ DÙNG ĐỂ KIỂM TRA CODE - KHÔNG PHẢI ẢNH THẬT!
Trong thực tế cần dùng ảnh thật từ dataset
"""

import cv2
import numpy as np
import os


def create_sample_fresh_fruit():
    """
    Tạo ảnh mô phỏng trái cây tươi
    - Màu sắc tươi sáng
    - Bề mặt bóng (có highlights)
    - Không có vết thâm
    """
    print("🍎 Tạo ảnh mô phỏng trái cây TƯƠI...")
    
    fruits = {
        'apple': (50, 50, 220),      # Đỏ tươi
        'banana': (0, 220, 255),     # Vàng tươi
        'orange': (0, 165, 255)      # Cam tươi
    }
    
    output_dir = 'data/test/fresh'
    os.makedirs(output_dir, exist_ok=True)
    
    for fruit_name, base_color in fruits.items():
        for i in range(5):
            # Tạo ảnh nền trắng
            img = np.ones((300, 300, 3), dtype=np.uint8) * 255
            
            # Vẽ hình tròn (trái cây)
            center = (150, 150)
            radius = 100
            
            # Màu chính (với biến thiên nhẹ)
            color = tuple(int(c + np.random.randint(-10, 10)) for c in base_color)
            cv2.circle(img, center, radius, color, -1)
            
            # Thêm gradient để tạo hiệu ứng 3D
            overlay = img.copy()
            cv2.circle(overlay, (130, 130), 50, (255, 255, 255), -1)
            img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
            
            # Thêm highlight (bóng)
            cv2.circle(img, (130, 130), 30, (255, 255, 255), -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Lưu
            filename = f'{fruit_name}_fresh_{i+1}.jpg'
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)
    
    print(f"   ✅ Đã tạo 15 ảnh mẫu trong {output_dir}/")


def create_sample_rotten_fruit():
    """
    Tạo ảnh mô phỏng trái cây hỏng
    - Màu sắc tối, xỉn
    - Có vết nâu/đen
    - Bề mặt không đều
    """
    print("🍂 Tạo ảnh mô phỏng trái cây HỎNG...")
    
    fruits = {
        'apple': (40, 40, 100),      # Đỏ sẫm
        'banana': (0, 100, 150),     # Vàng xỉn
        'orange': (0, 80, 120)       # Cam sẫm
    }
    
    output_dir = 'data/test/rotten'
    os.makedirs(output_dir, exist_ok=True)
    
    for fruit_name, base_color in fruits.items():
        for i in range(5):
            # Tạo ảnh nền trắng
            img = np.ones((300, 300, 3), dtype=np.uint8) * 255
            
            # Vẽ hình tròn với màu tối
            center = (150, 150)
            radius = 100
            
            color = tuple(int(c + np.random.randint(-10, 10)) for c in base_color)
            cv2.circle(img, center, radius, color, -1)
            
            # Thêm vết nâu/đen (mô phỏng vết hư)
            num_spots = np.random.randint(3, 7)
            for _ in range(num_spots):
                spot_x = center[0] + np.random.randint(-60, 60)
                spot_y = center[1] + np.random.randint(-60, 60)
                spot_radius = np.random.randint(10, 25)
                
                # Màu nâu/đen
                spot_color = (np.random.randint(20, 60), 
                            np.random.randint(20, 60), 
                            np.random.randint(20, 60))
                
                cv2.circle(img, (spot_x, spot_y), spot_radius, spot_color, -1)
            
            # Thêm texture nhăn (edges)
            noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Blur nhẹ (bề mặt mất độ bóng)
            img = cv2.GaussianBlur(img, (7, 7), 0)
            
            # Lưu
            filename = f'{fruit_name}_rotten_{i+1}.jpg'
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)
    
    print(f"   ✅ Đã tạo 15 ảnh mẫu trong {output_dir}/")


def main():
    """
    Tạo ảnh mẫu để test code
    """
    print("\n" + "="*70)
    print("TẠO ẢNH MẪU ĐỂ DEMO")
    print("="*70 + "\n")
    
    print("⚠️  LƯU Ý:")
    print("   - Đây chỉ là ảnh MÔ PHỎNG đơn giản")
    print("   - Không phải ảnh thật!")
    print("   - Chỉ dùng để TEST CODE")
    print("   - Để có kết quả tốt, dùng ảnh thật từ dataset!\n")
    
    try:
        # Tạo ảnh tươi
        create_sample_fresh_fruit()
        
        # Tạo ảnh hỏng
        create_sample_rotten_fruit()
        
        print("\n" + "="*70)
        print("✅ HOÀN THÀNH!")
        print("="*70)
        print("\n📂 Cấu trúc đã tạo:")
        print("   data/test/")
        print("   ├── fresh/")
        print("   │   ├── apple_fresh_1.jpg")
        print("   │   ├── banana_fresh_1.jpg")
        print("   │   └── orange_fresh_1.jpg")
        print("   └── rotten/")
        print("       ├── apple_rotten_1.jpg")
        print("       ├── banana_rotten_1.jpg")
        print("       └── orange_rotten_1.jpg")
        
        print("\n🚀 Bước tiếp theo:")
        print("   # Test với ảnh tươi")
        print("   python classify_fresh_rotten.py data/test/fresh/apple_fresh_1.jpg")
        print("\n   # Test với ảnh hỏng")
        print("   python classify_fresh_rotten.py data/test/rotten/apple_rotten_1.jpg")
        print("\n   # Test hàng loạt")
        print("   python classify_fresh_rotten.py data/test/fresh/ --batch")
        
        print("\n" + "="*70)
        print("💡 KHUYẾN NGHỊ:")
        print("   Để có kết quả tốt nhất, hãy tải dataset thật:")
        print("   → Xem file DATASET_FRESH_ROTTEN.md")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
