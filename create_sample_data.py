"""
Script tạo dataset mẫu với ảnh từ internet
Chỉ dùng để demo - trong thực tế cần ảnh thật
"""

import cv2
import numpy as np
import os


def create_sample_images():
    """
    Tạo các ảnh mẫu đơn giản để demo (không phải ảnh thật)
    Trong thực tế, bạn cần thu thập ảnh thật!
    """
    
    print("🎨 Tạo ảnh mẫu để demo...")
    print("⚠️  LƯU Ý: Đây chỉ là ảnh mẫu đơn giản!")
    print("    Để có kết quả tốt, bạn cần sử dụng ảnh thật!\n")
    
    # Định nghĩa màu sắc cơ bản cho mỗi loại (BGR format)
    fruit_colors = {
        'táo': [(50, 50, 200), (60, 60, 220)],        # Đỏ
        'chuối': [(0, 200, 255), (0, 220, 255)],      # Vàng
        'cam': [(0, 140, 255), (0, 165, 255)],        # Cam
        'nho': [(128, 0, 128), (148, 20, 148)],       # Tím
        'dưa hấu': [(0, 150, 0), (0, 180, 0)]         # Xanh lá
    }
    
    base_dirs = ['data/train', 'data/test']
    
    for base_dir in base_dirs:
        n_images = 25 if 'train' in base_dir else 5
        
        for fruit_name, colors in fruit_colors.items():
            folder_path = os.path.join(base_dir, fruit_name)
            os.makedirs(folder_path, exist_ok=True)
            
            print(f"📁 Tạo {n_images} ảnh mẫu cho {fruit_name} trong {base_dir}/")
            
            for i in range(n_images):
                # Tạo ảnh trắng
                img = np.ones((300, 300, 3), dtype=np.uint8) * 255
                
                # Chọn màu ngẫu nhiên trong khoảng
                color = tuple(np.random.randint(c[0], c[1]) for c in zip(colors[0], colors[1]))
                
                # Vẽ hình dạng (đơn giản hóa)
                if fruit_name == 'chuối':
                    # Hình elip dài (chuối)
                    center = (150 + np.random.randint(-20, 20), 150 + np.random.randint(-20, 20))
                    axes = (80 + np.random.randint(-10, 10), 40 + np.random.randint(-5, 5))
                    angle = np.random.randint(0, 180)
                    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
                else:
                    # Hình tròn (các trái cây khác)
                    center = (150 + np.random.randint(-20, 20), 150 + np.random.randint(-20, 20))
                    radius = 70 + np.random.randint(-10, 10)
                    cv2.circle(img, center, radius, color, -1)
                
                # Thêm một chút texture/noise
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
                
                # Thêm bóng đổ nhẹ
                shadow_offset = 5
                shadow_color = tuple(int(c * 0.7) for c in color)
                if fruit_name == 'chuối':
                    cv2.ellipse(img, 
                              (center[0] + shadow_offset, center[1] + shadow_offset), 
                              axes, angle, 0, 360, shadow_color, -1)
                    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
                else:
                    cv2.circle(img, 
                             (center[0] + shadow_offset, center[1] + shadow_offset), 
                             radius, shadow_color, -1)
                    cv2.circle(img, center, radius, color, -1)
                
                # Lưu ảnh
                filename = f"{fruit_name}_{i+1}.jpg"
                filepath = os.path.join(folder_path, filename)
                cv2.imwrite(filepath, img)
            
            print(f"   ✅ Đã tạo xong {n_images} ảnh")
    
    print("\n" + "="*60)
    print("✅ ĐÃ TẠO XONG ẢNH MẪU!")
    print("="*60)
    print("\n⚠️  QUAN TRỌNG:")
    print("   - Đây chỉ là ảnh mẫu đơn giản để demo code")
    print("   - Độ chính xác sẽ không cao với ảnh mẫu này")
    print("   - Để có kết quả tốt, vui lòng:")
    print("     1. Xóa các ảnh mẫu này")
    print("     2. Thu thập ảnh thật từ internet hoặc chụp")
    print("     3. Đặt ảnh thật vào các thư mục tương ứng")
    print("\n📚 Nguồn ảnh gợi ý:")
    print("   - Google Images: 'apple fruit', 'banana fruit', ...")
    print("   - Kaggle Datasets: kaggle.com/datasets")
    print("   - Chụp ảnh thật với điện thoại")
    print("="*60 + "\n")


if __name__ == '__main__':
    try:
        create_sample_images()
        print("\n💡 Bước tiếp theo:")
        print("   python main_train.py  # Để huấn luyện model")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
