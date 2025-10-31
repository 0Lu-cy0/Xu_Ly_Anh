"""
Module phân loại trái cây KHÔNG dùng Machine Learning
Chỉ sử dụng các kỹ thuật Xử Lý Ảnh thuần túy với OpenCV
Phù hợp cho môn Nhập môn Xử lý Ảnh
"""

import cv2
import numpy as np
import os
import json


class SimpleFruitClassifier:
    """
    Phân loại trái cây dựa trên màu sắc và hình dạng
    KHÔNG sử dụng Machine Learning
    """
    
    def __init__(self):
        """
        Khởi tạo classifier với các template màu sắc chuẩn
        """
        # Định nghĩa khoảng màu HSV cho mỗi loại trái cây
        self.fruit_color_ranges = {
            'táo': {
                'name': 'Táo (Apple)',
                'hsv_ranges': [
                    # Đỏ (2 khoảng vì Hue của đỏ ở 2 đầu)
                    {'lower': np.array([0, 100, 50]), 'upper': np.array([10, 255, 255])},
                    {'lower': np.array([170, 100, 50]), 'upper': np.array([180, 255, 255])},
                ],
                'expected_circularity': (0.7, 1.0),  # Khá tròn
                'expected_aspect_ratio': (0.8, 1.2)  # Gần vuông
            },
            'chuối': {
                'name': 'Chuối (Banana)',
                'hsv_ranges': [
                    # Vàng
                    {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
                ],
                'expected_circularity': (0.2, 0.5),  # Dài, không tròn
                'expected_aspect_ratio': (1.5, 4.0)  # Rất dài
            },
            'cam': {
                'name': 'Cam (Orange)',
                'hsv_ranges': [
                    # Cam
                    {'lower': np.array([10, 100, 100]), 'upper': np.array([25, 255, 255])},
                ],
                'expected_circularity': (0.7, 1.0),  # Tròn
                'expected_aspect_ratio': (0.8, 1.2)  # Gần vuông
            },
            'nho': {
                'name': 'Nho (Grape)',
                'hsv_ranges': [
                    # Tím
                    {'lower': np.array([120, 50, 50]), 'upper': np.array([150, 255, 255])},
                    # Xanh lá (nho xanh)
                    {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
                ],
                'expected_circularity': (0.6, 1.0),  # Khá tròn
                'expected_aspect_ratio': (0.7, 1.3)  # Gần tròn
            },
            'dưa hấu': {
                'name': 'Dưa hấu (Watermelon)',
                'hsv_ranges': [
                    # Xanh lá đậm
                    {'lower': np.array([35, 80, 50]), 'upper': np.array([85, 255, 255])},
                ],
                'expected_circularity': (0.6, 1.0),  # Khá tròn
                'expected_aspect_ratio': (0.8, 1.3)  # Gần tròn đến oval
            }
        }
        
        self.template_features = {}
    
    
    def extract_color_features(self, image):
        """
        Trích xuất đặc trưng màu sắc từ ảnh
        
        Returns:
            dict: Tỷ lệ pixel của mỗi màu
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_scores = {}
        
        for fruit_name, fruit_info in self.fruit_color_ranges.items():
            total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            # Tạo mask cho mỗi khoảng màu
            for color_range in fruit_info['hsv_ranges']:
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                total_mask = cv2.bitwise_or(total_mask, mask)
            
            # Tính tỷ lệ pixel khớp màu
            color_ratio = np.count_nonzero(total_mask) / (image.shape[0] * image.shape[1])
            color_scores[fruit_name] = color_ratio
        
        return color_scores
    
    
    def extract_shape_features(self, image):
        """
        Trích xuất đặc trưng hình dạng
        
        Returns:
            dict: Circularity và aspect ratio
        """
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {'circularity': 0, 'aspect_ratio': 1.0}
        
        # Lấy contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Tính circularity
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Tính aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 1.0
        
        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio
        }
    
    
    def calculate_similarity(self, color_scores, shape_features):
        """
        Tính độ tương đồng với từng loại trái cây
        Dựa trên màu sắc và hình dạng
        
        Returns:
            dict: Điểm số cho mỗi loại trái cây
        """
        similarity_scores = {}
        
        for fruit_name, fruit_info in self.fruit_color_ranges.items():
            # Điểm màu sắc (0-100)
            color_score = color_scores.get(fruit_name, 0) * 100
            
            # Điểm hình dạng - circularity (0-50)
            expected_circ = fruit_info['expected_circularity']
            actual_circ = shape_features['circularity']
            
            if expected_circ[0] <= actual_circ <= expected_circ[1]:
                circ_score = 50  # Khớp hoàn toàn
            else:
                # Tính khoảng cách đến khoảng mong đợi
                if actual_circ < expected_circ[0]:
                    diff = expected_circ[0] - actual_circ
                else:
                    diff = actual_circ - expected_circ[1]
                circ_score = max(0, 50 - diff * 100)
            
            # Điểm aspect ratio (0-50)
            expected_ar = fruit_info['expected_aspect_ratio']
            actual_ar = shape_features['aspect_ratio']
            
            if expected_ar[0] <= actual_ar <= expected_ar[1]:
                ar_score = 50  # Khớp hoàn toàn
            else:
                if actual_ar < expected_ar[0]:
                    diff = expected_ar[0] - actual_ar
                else:
                    diff = actual_ar - expected_ar[1]
                ar_score = max(0, 50 - diff * 50)
            
            # Tổng điểm (trọng số: màu 50%, circularity 25%, aspect_ratio 25%)
            total_score = (color_score * 0.5 + circ_score * 0.25 + ar_score * 0.25)
            similarity_scores[fruit_name] = total_score
        
        return similarity_scores
    
    
    def predict(self, image):
        """
        Dự đoán loại trái cây từ ảnh
        KHÔNG sử dụng Machine Learning
        
        Args:
            image: Ảnh đầu vào (BGR)
        
        Returns:
            dict: Kết quả dự đoán với xác suất
        """
        # Tiền xử lý
        image = cv2.resize(image, (200, 200))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Trích xuất đặc trưng màu sắc
        color_scores = self.extract_color_features(image)
        
        # Trích xuất đặc trưng hình dạng
        shape_features = self.extract_shape_features(image)
        
        # Tính độ tương đồng
        similarity_scores = self.calculate_similarity(color_scores, shape_features)
        
        # Tìm kết quả tốt nhất
        best_fruit = max(similarity_scores, key=similarity_scores.get)
        best_score = similarity_scores[best_fruit]
        
        # Chuẩn hóa thành xác suất (0-1)
        total_score = sum(similarity_scores.values())
        if total_score > 0:
            probabilities = {fruit: score/total_score 
                           for fruit, score in similarity_scores.items()}
        else:
            probabilities = {fruit: 1.0/len(similarity_scores) 
                           for fruit in similarity_scores.keys()}
        
        # Map sang ID số
        fruit_ids = {'táo': 0, 'chuối': 1, 'cam': 2, 'nho': 3, 'dưa hấu': 4}
        
        return {
            'predicted_class': fruit_ids[best_fruit],
            'predicted_name': self.fruit_color_ranges[best_fruit]['name'],
            'confidence': probabilities[best_fruit],
            'probabilities': {fruit_ids[k]: v for k, v in probabilities.items()},
            'debug_info': {
                'color_scores': color_scores,
                'shape_features': shape_features,
                'similarity_scores': similarity_scores
            }
        }
    
    
    def save_config(self, filepath):
        """
        Lưu cấu hình (không phải model ML!)
        Chỉ lưu các tham số màu sắc và hình dạng
        """
        config = {
            'type': 'simple_image_processing',
            'note': 'Không sử dụng Machine Learning',
            'fruit_color_ranges': {
                k: {
                    'name': v['name'],
                    'hsv_ranges': [
                        {'lower': r['lower'].tolist(), 'upper': r['upper'].tolist()}
                        for r in v['hsv_ranges']
                    ],
                    'expected_circularity': v['expected_circularity'],
                    'expected_aspect_ratio': v['expected_aspect_ratio']
                }
                for k, v in self.fruit_color_ranges.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Đã lưu cấu hình vào {filepath}")
    
    
    def load_config(self, filepath):
        """
        Tải cấu hình từ file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Khôi phục lại numpy arrays
        for fruit_name, fruit_data in config['fruit_color_ranges'].items():
            for range_data in fruit_data['hsv_ranges']:
                range_data['lower'] = np.array(range_data['lower'])
                range_data['upper'] = np.array(range_data['upper'])
        
        self.fruit_color_ranges = {
            k: {
                'name': v['name'],
                'hsv_ranges': v['hsv_ranges'],
                'expected_circularity': tuple(v['expected_circularity']),
                'expected_aspect_ratio': tuple(v['expected_aspect_ratio'])
            }
            for k, v in config['fruit_color_ranges'].items()
        }
        
        print(f"Đã tải cấu hình từ {filepath}")


def demonstrate_technique():
    """
    Hàm demo để hiển thị các kỹ thuật xử lý ảnh được sử dụng
    """
    print("\n" + "="*70)
    print("CÁC KỸ THUẬT XỬ LÝ ẢNH ĐƯỢC SỬ DỤNG")
    print("="*70)
    print("""
1. CHUYỂN ĐỔI KHÔNG GIAN MÀU (Color Space Conversion)
   - BGR → HSV: cv2.cvtColor()
   - Lý do: HSV tách biệt màu sắc và độ sáng

2. PHÂN NGƯỠNG MÀU (Color Thresholding)
   - cv2.inRange(): Tạo mask cho khoảng màu
   - So sánh tỷ lệ pixel khớp màu

3. THRESHOLD (Ngưỡng hóa)
   - Otsu's method: cv2.threshold()
   - Tách đối tượng khỏi nền

4. TÌM ĐƯỜNG VIỀN (Contour Detection)
   - cv2.findContours(): Tìm biên đối tượng
   - Phân tích hình học

5. TÍNH TOÁN ĐẶC TRƯNG HÌNH HỌC
   - Circularity: 4πA/P²
   - Aspect Ratio: W/H
   - Area, Perimeter

6. SO SÁNH TRỰC TIẾP (Rule-based Classification)
   - Không dùng ML!
   - Dựa trên ngưỡng màu sắc và hình dạng định trước

✅ TẤT CẢ ĐỀU LÀ KỸ THUẬT XỬ LÝ ẢNH CƠ BẢN!
❌ KHÔNG CÓ: SVM, KNN, Neural Networks, Training, etc.
    """)
    print("="*70 + "\n")
