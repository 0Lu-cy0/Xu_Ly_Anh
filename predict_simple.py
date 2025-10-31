"""
Script dự đoán KHÔNG dùng Machine Learning
Chỉ sử dụng kỹ thuật Xử lý Ảnh thuần túy
Phù hợp cho môn Nhập môn Xử lý Ảnh
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simple_classifier import SimpleFruitClassifier
from utils.data_loader import FRUIT_CLASSES


def predict_simple(image_path, show_steps=True):
    """
    Dự đoán trái cây KHÔNG dùng ML
    
    Args:
        image_path: Đường dẫn ảnh
        show_steps: Hiển thị các bước xử lý
    """
    print("\n" + "="*70)
    print("NHẬN DIỆN TRÁI CÂY - PHƯƠNG PHÁP XỬ LÝ ẢNH THUẦN TÚY")
    print("(KHÔNG SỬ DỤNG MACHINE LEARNING)")
    print("="*70 + "\n")
    
    # Kiểm tra file
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return
    
    # Đọc ảnh
    print(f"📷 Đọc ảnh: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể đọc ảnh!")
        return
    
    # Tạo classifier
    classifier = SimpleFruitClassifier()
    
    # Dự đoán
    print("🔍 Đang phân tích ảnh...\n")
    result = classifier.predict(image)
    
    # Hiển thị kết quả
    print("="*70)
    print("KẾT QUẢ NHẬN DIỆN")
    print("="*70)
    print(f"\n🍎 Loại trái cây: {result['predicted_name']}")
    print(f"📊 Độ tin cậy: {result['confidence']:.2%}\n")
    
    print("📈 Xác suất cho từng loại:")
    for fruit_id, prob in sorted(result['probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True):
        fruit_name = FRUIT_CLASSES[fruit_id]
        bar = "█" * int(prob * 30)
        print(f"   {fruit_name:<25} {bar} {prob:.2%}")
    
    # Hiển thị thông tin debug nếu cần
    if show_steps:
        print("\n" + "="*70)
        print("CHI TIẾT QUÁ TRÌNH XỬ LÝ")
        print("="*70)
        
        debug = result['debug_info']
        
        print("\n1️⃣ PHÂN TÍCH MÀU SẮC (Color Thresholding):")
        for fruit, score in sorted(debug['color_scores'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"   {fruit:<15} : {score:.2%} pixel khớp màu")
        
        print("\n2️⃣ PHÂN TÍCH HÌNH DẠNG (Shape Analysis):")
        shape = debug['shape_features']
        print(f"   Circularity     : {shape['circularity']:.3f}")
        print(f"   Aspect Ratio    : {shape['aspect_ratio']:.3f}")
        
        print("\n3️⃣ ĐIỂM TƯƠNG ĐỒNG (Similarity Scores):")
        for fruit, score in sorted(debug['similarity_scores'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"   {fruit:<15} : {score:.2f}/100")
    
    print("\n" + "="*70)
    print("KỸ THUẬT SỬ DỤNG: Chỉ Xử lý Ảnh với OpenCV")
    print("✅ cv2.cvtColor() - Chuyển đổi màu")
    print("✅ cv2.inRange() - Phân ngưỡng màu") 
    print("✅ cv2.threshold() - Otsu thresholding")
    print("✅ cv2.findContours() - Tìm đường viền")
    print("✅ Tính toán đặc trưng hình học")
    print("✅ So sánh dựa trên luật (Rule-based)")
    print("\n❌ KHÔNG SỬ DỤNG: SVM, KNN, Training, Model, sklearn")
    print("="*70 + "\n")
    
    # Hiển thị ảnh nếu muốn
    if cv2.waitKey(1) != -1:  # Kiểm tra nếu có thể hiển thị
        try:
            # Hiển thị ảnh gốc
            cv2.imshow('Anh goc', image)
            
            # Hiển thị ảnh HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('HSV', hsv)
            
            # Hiển thị threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow('Threshold', thresh)
            
            print("Nhấn phím bất kỳ để đóng cửa sổ...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass


def main():
    """
    Hàm chính
    """
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Nhập đường dẫn ảnh: ").strip()
    
    predict_simple(image_path, show_steps=True)


if __name__ == '__main__':
    main()
