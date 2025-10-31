"""
Script chính để phân loại Trái cây Tươi vs Hỏng
Chỉ sử dụng kỹ thuật Xử lý Ảnh - OpenCV
KHÔNG dùng Machine Learning

Chạy: python classify_fresh_rotten.py <đường_dẫn_ảnh>
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fresh_rotten_classifier import FreshRottenClassifier


def print_technique_info():
    """
    In thông tin về các kỹ thuật xử lý ảnh được sử dụng
    """
    print("\n" + "="*80)
    print("CÁC KỸ THUẬT XỬ LÝ ẢNH ĐƯỢC SỬ DỤNG (KHÔNG DÙNG MACHINE LEARNING)")
    print("="*80)
    print("""
📚 DANH SÁCH KỸ THUẬT:

1. CHUYỂN ĐỔI KHÔNG GIAN MÀU (Color Space Conversion)
   • BGR → HSV: cv2.cvtColor()
   • Phân tích: Hue, Saturation, Value
   
2. PHÂN TÍCH MÀU SẮC (Color Analysis)
   • cv2.inRange(): Phát hiện màu nâu, đen
   • Tính tỷ lệ pixel: Vết thâm, vết hư
   • Statistical measures: Mean, Std của S và V
   
3. PHÂN TÍCH TEXTURE (Texture Analysis)
   • Sobel Operator: Phát hiện độ nhăn nheo
   • Laplacian: Phát hiện thay đổi đột ngột
   • Local Variance: Đánh giá độ đồng đều
   
4. PHÂN NGƯỠNG (Thresholding)
   • Global Threshold: cv2.threshold()
   • Adaptive Threshold: cv2.adaptiveThreshold()
   
5. PHÉP TOÁN HÌNH THÁI (Morphological Operations)
   • Opening: Loại bỏ nhiễu
   • Closing: Làm mịn đường viền
   • cv2.morphologyEx()
   
6. PHÁT HIỆN ĐƯỜNG VIỀN (Contour Detection)
   • cv2.findContours(): Tìm vết hư hỏng
   • Đếm số lượng, tính diện tích
   
7. HISTOGRAM ANALYSIS
   • cv2.calcHist(): Phân tích phân bố màu
   • Entropy: Đo độ hỗn loạn
   • Contrast: Đo độ tương phản

8. RULE-BASED CLASSIFICATION
   • Hệ thống chấm điểm dựa trên ngưỡng
   • Không học từ dữ liệu!
   • Logic if-else, so sánh trực tiếp

✅ TẤT CẢ ĐỀU LÀ KỸ THUẬT XỬ LÝ ẢNH CƠ BẢN
❌ KHÔNG CÓ: SVM, Neural Networks, Training, Model, sklearn
    """)
    print("="*80 + "\n")


def analyze_image(image_path, show_visualization=True):
    """
    Phân tích và phân loại ảnh trái cây
    
    Args:
        image_path: Đường dẫn đến ảnh
        show_visualization: Hiển thị kết quả trực quan
    """
    print("\n" + "="*80)
    print("PHÂN LOẠI TRÁI CÂY TƯƠI vs HỎNG")
    print("="*80 + "\n")
    
    # Kiểm tra file
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: File không tồn tại: {image_path}")
        return
    
    # Đọc ảnh
    print(f"📷 Đang đọc ảnh: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Lỗi: Không thể đọc ảnh!")
        return
    
    print(f"   Kích thước: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Tạo classifier
    classifier = FreshRottenClassifier()
    
    # Phân tích
    print("\n🔬 Đang phân tích...")
    result = classifier.classify(image, show_details=True)
    
    # Hiển thị kết quả
    print("\n" + "="*80)
    print("KẾT QUẢ PHÂN LOẠI")
    print("="*80)
    
    status = result['status']
    score = result['health_score']
    confidence = result['confidence']
    
    # Icon theo trạng thái
    if status == "TƯƠI":
        icon = "✅"
    elif status == "TRUNG BÌNH":
        icon = "⚠️"
    else:
        icon = "❌"
    
    print(f"\n{icon} TRẠNG THÁI: {status}")
    print(f"📊 ĐIỂM SỨC KHỎE: {score:.1f}/100")
    print(f"🎯 ĐỘ TIN CẬY: {confidence:.1%}")
    
    # Thanh progress bar
    bar_length = 50
    filled = int(score / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\n[{bar}] {score:.1f}%")
    
    # Lý do đánh giá
    if result['reasons']:
        print(f"\n📝 CHI TIẾT ĐÁNH GIÁ:")
        for i, reason in enumerate(result['reasons'], 1):
            print(f"   {i}. {reason}")
    else:
        print(f"\n✨ Trái cây trong tình trạng tốt!")
    
    # Chi tiết đặc trưng
    print("\n" + "="*80)
    print("CHI TIẾT CÁC ĐẶC TRƯNG")
    print("="*80)
    
    features = result['features']
    
    print("\n🎨 1. ĐẶC TRƯNG MÀU SẮC:")
    color = features['color']
    print(f"   • Độ bão hòa trung bình: {color['mean_saturation']:.1f}")
    print(f"   • Độ sáng trung bình: {color['mean_brightness']:.1f}")
    print(f"   • Tỷ lệ vết đen: {color['dark_spots_ratio']:.2%}")
    print(f"   • Tỷ lệ vết nâu: {color['brown_spots_ratio']:.2%}")
    
    print("\n🖼️  2. ĐẶC TRƯNG TEXTURE:")
    texture = features['texture']
    print(f"   • Cường độ edges: {texture['mean_edge_intensity']:.2f}")
    print(f"   • Laplacian: {texture['mean_laplacian']:.2f}")
    print(f"   • Biến thiên cục bộ: {texture['mean_local_variance']:.2f}")
    
    print("\n✨ 3. ĐỘ MỊN BỀ MẶT:")
    smoothness = features['smoothness']
    print(f"   • Entropy: {smoothness['entropy']:.2f}")
    print(f"   • Contrast: {smoothness['contrast']:.2f}")
    print(f"   • Tỷ lệ vùng sáng: {smoothness['bright_ratio']:.2%}")
    
    print("\n🔍 4. VẾT HƯ HỎNG:")
    defects = features['defects']
    print(f"   • Số vết thâm: {defects['num_spots']}")
    print(f"   • Diện tích vết: {defects['spot_area_ratio']:.2%}")
    
    print("\n" + "="*80)
    
    # Hiển thị ảnh nếu có thể
    if show_visualization:
        try:
            # Ảnh gốc
            original = cv2.imread(image_path)
            original_display = cv2.resize(original, (400, 400))
            
            # Ảnh kết quả với annotation
            result_img = classifier.visualize_analysis(original, result)
            
            # Tạo các ảnh phân tích
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Resize để hiển thị
            s_display = cv2.resize(s, (400, 400))
            v_display = cv2.resize(v, (400, 400))
            defect_mask = cv2.resize(defects['defect_mask'], (400, 400))
            
            # Hiển thị
            cv2.imshow('1. Anh goc', original_display)
            cv2.imshow('2. Ket qua phan loai', result_img)
            cv2.imshow('3. Saturation (Do bao hoa)', s_display)
            cv2.imshow('4. Value (Do sang)', v_display)
            cv2.imshow('5. Vet hu hong phat hien', defect_mask)
            
            print("\n📺 Đang hiển thị kết quả...")
            print("   Nhấn phím bất kỳ để đóng tất cả cửa sổ...")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"\n⚠️  Không thể hiển thị ảnh: {str(e)}")


def batch_analyze(folder_path):
    """
    Phân tích nhiều ảnh trong một thư mục
    """
    print("\n" + "="*80)
    print("PHÂN TÍCH HÀNG LOẠT")
    print("="*80 + "\n")
    
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh trong thư mục!")
        return
    
    print(f"📂 Tìm thấy {len(image_files)} ảnh")
    print(f"📊 Đang phân tích...\n")
    
    classifier = FreshRottenClassifier()
    results_summary = {'TƯƠI': 0, 'TRUNG BÌNH': 0, 'HỎNG': 0}
    
    for i, filename in enumerate(image_files, 1):
        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath)
        
        if image is None:
            print(f"   {i}. {filename:<30} ⚠️  Lỗi đọc file")
            continue
        
        result = classifier.classify(image)
        status = result['status']
        score = result['health_score']
        
        results_summary[status] += 1
        
        # Icon
        if status == "TƯƠI":
            icon = "✅"
        elif status == "TRUNG BÌNH":
            icon = "⚠️"
        else:
            icon = "❌"
        
        print(f"   {i}. {filename:<30} {icon} {status:<12} ({score:.1f}/100)")
    
    # Tổng kết
    print("\n" + "="*80)
    print("TỔNG KẾT")
    print("="*80)
    print(f"\n✅ Tươi:        {results_summary['TƯƠI']:>3} ảnh ({results_summary['TƯƠI']/len(image_files)*100:.1f}%)")
    print(f"⚠️  Trung bình: {results_summary['TRUNG BÌNH']:>3} ảnh ({results_summary['TRUNG BÌNH']/len(image_files)*100:.1f}%)")
    print(f"❌ Hỏng:        {results_summary['HỎNG']:>3} ảnh ({results_summary['HỎNG']/len(image_files)*100:.1f}%)")
    print(f"\n📊 Tổng cộng:   {len(image_files):>3} ảnh")
    print("="*80 + "\n")


def main():
    """
    Hàm chính
    """
    print_technique_info()
    
    # Kiểm tra arguments
    if len(sys.argv) < 2:
        print("Cách sử dụng:")
        print("  python classify_fresh_rotten.py <đường_dẫn_ảnh>")
        print("  python classify_fresh_rotten.py <đường_dẫn_thư_mục> --batch")
        print("\nVí dụ:")
        print("  python classify_fresh_rotten.py apple_fresh.jpg")
        print("  python classify_fresh_rotten.py data/test/fresh/ --batch")
        return
    
    path = sys.argv[1]
    
    # Kiểm tra batch mode
    if len(sys.argv) > 2 and sys.argv[2] == '--batch':
        batch_analyze(path)
    else:
        # Single image
        if os.path.isdir(path):
            print("❌ Đây là thư mục. Sử dụng --batch để phân tích hàng loạt")
            print(f"   python classify_fresh_rotten.py {path} --batch")
        else:
            analyze_image(path, show_visualization=True)


if __name__ == '__main__':
    main()
