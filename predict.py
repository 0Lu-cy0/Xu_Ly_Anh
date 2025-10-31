"""
Script để dự đoán loại trái cây từ ảnh đầu vào
"""

import os
import sys
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processing import load_image, extract_all_features, draw_contours
from utils.data_loader import FRUIT_CLASSES
from src.train_model import FruitClassifier


def predict_fruit(image_path, model_path='models/fruit_classifier.pkl', show_image=True):
    """
    Dự đoán loại trái cây từ ảnh
    
    Args:
        image_path (str): Đường dẫn đến file ảnh
        model_path (str): Đường dẫn đến file model
        show_image (bool): Có hiển thị ảnh kết quả không
    
    Returns:
        dict: Kết quả dự đoán
    """
    # Kiểm tra file model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại tại {model_path}. Vui lòng huấn luyện model trước!")
    
    # Tải model
    print("📥 Đang tải model...")
    classifier = FruitClassifier()
    classifier.load_model(model_path)
    
    # Đọc ảnh
    print(f"📷 Đang xử lý ảnh: {image_path}")
    image = load_image(image_path)
    
    # Trích xuất đặc trưng
    print("🔍 Đang trích xuất đặc trưng...")
    features = extract_all_features(image)
    
    # Dự đoán
    print("🤖 Đang dự đoán...")
    result = classifier.predict_with_probabilities(features)
    
    predicted_class = result['predicted_class']
    fruit_name = FRUIT_CLASSES[predicted_class]
    
    print(f"\n{'='*50}")
    print(f"KẾT QUẢ DỰ ĐOÁN")
    print(f"{'='*50}")
    print(f"🍎 Loại trái cây: {fruit_name}")
    
    if 'probabilities' in result:
        print(f"\n📊 Xác suất cho các loại:")
        probs = result['probabilities']
        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {FRUIT_CLASSES[cls]:<20} : {prob:.2%}")
    
    print(f"{'='*50}\n")
    
    # Hiển thị ảnh kết quả
    if show_image:
        # Vẽ contours lên ảnh
        result_image = draw_contours(image)
        
        # Thêm text kết quả
        text = f"{fruit_name}"
        cv2.putText(result_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Hiển thị
        cv2.imshow('Ket qua nhan dien', result_image)
        cv2.imshow('Anh goc', image)
        print("Nhấn phím bất kỳ để đóng cửa sổ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def main():
    """
    Hàm chính
    """
    print("\n" + "="*60)
    print("CHƯƠNG TRÌNH NHẬN DIỆN TRÁI CÂY")
    print("="*60 + "\n")
    
    # Đường dẫn model
    model_path = 'models/fruit_classifier.pkl'
    
    # Nhập đường dẫn ảnh từ người dùng
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Nhập đường dẫn đến file ảnh: ").strip()
    
    # Kiểm tra file ảnh
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: File {image_path} không tồn tại!")
        return
    
    try:
        # Dự đoán
        result = predict_fruit(image_path, model_path)
        
    except FileNotFoundError as e:
        print(f"\n❌ Lỗi: {str(e)}")
        print("\nVui lòng chạy main_train.py trước để huấn luyện model!")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
