"""
Module phân loại Trái cây Tươi vs Hỏng
Chỉ sử dụng kỹ thuật Xử lý Ảnh với OpenCV
KHÔNG dùng Machine Learning
Phù hợp cho môn Nhập môn Xử lý Ảnh
"""

import cv2
import numpy as np
import os


class FreshRottenClassifier:
    """
    Phân loại trái cây tươi hay hỏng dựa trên:
    - Màu sắc (Color Analysis)
    - Độ bóng/mờ (Brightness/Contrast)
    - Texture (Độ nhám bề mặt)
    - Vết thâm/đen (Dark spots)
    
    KHÔNG sử dụng Machine Learning
    """
    
    def __init__(self):
        """
        Khởi tạo classifier với các ngưỡng định trước
        """
        self.results_history = []
    
    
    def analyze_color_health(self, image):
        """
        Phân tích màu sắc để đánh giá độ tươi
        Trái cây tươi: Màu sắc tươi sáng, đồng đều
        Trái cây hỏng: Màu xỉn, có vệt nâu/đen
        
        Returns:
            dict: Các chỉ số màu sắc
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 1. Phân tích độ bão hòa (Saturation)
        # Trái cây tươi có màu bão hòa cao
        mean_saturation = np.mean(s)
        std_saturation = np.std(s)
        
        # 2. Phân tích độ sáng (Value)
        # Trái cây hỏng thường sẫm màu hơn
        mean_brightness = np.mean(v)
        std_brightness = np.std(v)
        
        # 3. Tìm vùng màu đen/nâu (dấu hiệu hư hỏng)
        # Threshold để tìm vùng tối
        _, dark_mask = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.count_nonzero(dark_mask) / (image.shape[0] * image.shape[1])
        
        # 4. Phân tích màu nâu (trong HSV)
        # Nâu: Hue 10-20, Saturation thấp-trung bình
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_ratio = np.count_nonzero(brown_mask) / (image.shape[0] * image.shape[1])
        
        return {
            'mean_saturation': mean_saturation,
            'std_saturation': std_saturation,
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'dark_spots_ratio': dark_ratio,
            'brown_spots_ratio': brown_ratio
        }
    
    
    def analyze_texture(self, image):
        """
        Phân tích texture bề mặt
        Trái cây tươi: Bề mặt mịn, edges nhẹ
        Trái cây hỏng: Bề mặt nhăn, edges mạnh, không đều
        
        Returns:
            dict: Các chỉ số texture
        """
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Sobel để phát hiện edges (độ nhăn nheo)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Edge intensity cao = bề mặt nhăn nheo
        mean_edge_intensity = np.mean(sobel_magnitude)
        std_edge_intensity = np.std(sobel_magnitude)
        
        # 2. Laplacian (phát hiện thay đổi đột ngột)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mean_laplacian = np.mean(np.abs(laplacian))
        
        # 3. Độ biến thiên cục bộ (Local Variance)
        # Sử dụng Standard Deviation Filter
        kernel_size = 5
        local_std = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size)) - \
                    cv2.blur(gray.astype(float), (kernel_size, kernel_size))**2
        local_std = np.sqrt(np.maximum(local_std, 0))
        mean_local_variance = np.mean(local_std)
        
        return {
            'mean_edge_intensity': mean_edge_intensity,
            'std_edge_intensity': std_edge_intensity,
            'mean_laplacian': mean_laplacian,
            'mean_local_variance': mean_local_variance
        }
    
    
    def analyze_surface_smoothness(self, image):
        """
        Phân tích độ mịn bề mặt
        Trái cây tươi: Bề mặt bóng, phản chiếu sáng
        Trái cây hỏng: Bề mặt khô, mờ đục
        
        Returns:
            dict: Các chỉ số độ mịn
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Phân tích histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Entropy (độ hỗn loạn) - càng cao càng không đồng nhất
        hist_normalized = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
        
        # 2. Contrast (độ tương phản)
        contrast = gray.std()
        
        # 3. Tìm vùng sáng (highlights - dấu hiệu bề mặt bóng)
        bright_threshold = np.percentile(gray, 85)
        bright_mask = gray > bright_threshold
        bright_ratio = np.count_nonzero(bright_mask) / (image.shape[0] * image.shape[1])
        
        return {
            'entropy': entropy,
            'contrast': contrast,
            'bright_ratio': bright_ratio
        }
    
    
    def detect_spots_and_defects(self, image):
        """
        Phát hiện vết thâm, vết hư hỏng trên bề mặt
        Sử dụng Morphological Operations
        
        Returns:
            dict: Thông tin về vết thâm
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 1. Tìm các vùng tối bất thường
        # Sử dụng adaptive threshold để phát hiện vùng tối cục bộ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # 2. Morphological operations để loại bỏ nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        
        # 3. Đếm số vết thâm
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours nhỏ (nhiễu)
        min_area = 50
        valid_spots = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        num_spots = len(valid_spots)
        
        # Tổng diện tích các vết
        total_spot_area = sum([cv2.contourArea(cnt) for cnt in valid_spots])
        image_area = image.shape[0] * image.shape[1]
        spot_area_ratio = total_spot_area / image_area
        
        return {
            'num_spots': num_spots,
            'spot_area_ratio': spot_area_ratio,
            'defect_mask': morphed  # Trả về mask để visualize
        }
    
    
    def classify(self, image, show_details=False):
        """
        Phân loại trái cây tươi hay hỏng
        
        Args:
            image: Ảnh đầu vào (BGR)
            show_details: Hiển thị chi tiết phân tích
        
        Returns:
            dict: Kết quả phân loại
        """
        # Tiền xử lý
        image = cv2.resize(image, (300, 300))
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Phân tích các đặc trưng
        color_features = self.analyze_color_health(image)
        texture_features = self.analyze_texture(image)
        smoothness_features = self.analyze_surface_smoothness(image)
        defect_features = self.detect_spots_and_defects(image)
        
        # === HỆ THỐNG CHẤM ĐIỂM (RULE-BASED) ===
        
        health_score = 100.0  # Bắt đầu với 100 điểm
        reasons = []
        
        # 1. Đánh giá màu sắc (trọng số: 30%)
        # Saturation thấp = màu xỉn (nhưng cho phép màu xanh tự nhiên)
        # Chỉ trừ điểm nếu ĐỒNG THỜI saturation thấp VÀ brightness thấp
        if color_features['mean_saturation'] < 60 and color_features['mean_brightness'] < 80:
            penalty = (60 - color_features['mean_saturation']) * 0.2
            health_score -= penalty
            reasons.append(f"Màu sắc xỉn, tối (-{penalty:.1f} điểm)")
        
        # Vết đen/tối (nghiêm trọng)
        if color_features['dark_spots_ratio'] > 0.15:
            penalty = color_features['dark_spots_ratio'] * 180
            health_score -= penalty
            reasons.append(f"Có vết đen (-{penalty:.1f} điểm)")
        
        # Vết nâu (dấu hiệu hư hỏng)
        if color_features['brown_spots_ratio'] > 0.20:
            penalty = color_features['brown_spots_ratio'] * 120
            health_score -= penalty
            reasons.append(f"Có vết nâu (-{penalty:.1f} điểm)")
        
        # 2. Đánh giá texture (trọng số: 25%)
        # Edge intensity cao = bề mặt nhăn nheo
        if texture_features['mean_edge_intensity'] > 40:
            penalty = (texture_features['mean_edge_intensity'] - 40) * 0.4
            health_score -= penalty
            reasons.append(f"Bề mặt nhăn (-{penalty:.1f} điểm)")
        
        # Local variance cao = không đồng đều (nới lỏng hơn)
        if texture_features['mean_local_variance'] > 50:
            penalty = (texture_features['mean_local_variance'] - 50) * 0.25
            health_score -= penalty
            reasons.append(f"Bề mặt không đều (-{penalty:.1f} điểm)")
        
        # 3. Đánh giá độ mịn (trọng số: 20%)
        # Entropy cao = không đồng nhất (nới lỏng)
        if smoothness_features['entropy'] > 7.0:
            penalty = (smoothness_features['entropy'] - 7.0) * 6
            health_score -= penalty
            reasons.append(f"Độ đồng nhất kém (-{penalty:.1f} điểm)")
        
        # Không trừ điểm cho bright_ratio thấp vì trái xanh tự nhiên ít bóng
        
        # 4. Đánh giá vết hư hỏng (trọng số: 25%)
        if defect_features['num_spots'] > 5:
            penalty = (defect_features['num_spots'] - 5) * 4
            health_score -= penalty
            reasons.append(f"Có {defect_features['num_spots']} vết thâm (-{penalty:.1f} điểm)")
        
        if defect_features['spot_area_ratio'] > 0.08:
            penalty = defect_features['spot_area_ratio'] * 250
            health_score -= penalty
            reasons.append(f"Vết hư hỏng lớn (-{penalty:.1f} điểm)")
        
        # Giới hạn điểm trong khoảng 0-100
        health_score = max(0, min(100, health_score))
        
        # === QUYẾT ĐỊNH CUỐI CÙNG ===
        if health_score >= 65:
            status = "TƯƠI"
            color_code = (0, 255, 0)  # Xanh lá
            confidence = health_score / 100
        elif health_score >= 35:
            status = "TRUNG BÌNH"
            color_code = (0, 165, 255)  # Cam
            confidence = 0.5 + (health_score - 35) / 60 * 0.3
        else:
            status = "HỎNG"
            color_code = (0, 0, 255)  # Đỏ
            confidence = 1.0 - (health_score / 40) * 0.3
        
        result = {
            'status': status,
            'health_score': health_score,
            'confidence': confidence,
            'color_code': color_code,
            'reasons': reasons,
            'features': {
                'color': color_features,
                'texture': texture_features,
                'smoothness': smoothness_features,
                'defects': defect_features
            }
        }
        
        self.results_history.append(result)
        
        return result
    
    
    def visualize_analysis(self, image, result):
        """
        Hiển thị kết quả phân tích trên ảnh
        
        Returns:
            numpy.ndarray: Ảnh có kết quả
        """
        # Resize để hiển thị
        display_img = cv2.resize(image, (400, 400))
        
        # Vẽ khung màu theo kết quả
        color = result['color_code']
        cv2.rectangle(display_img, (0, 0), (399, 399), color, 15)
        
        # Thêm text kết quả
        status = result['status']
        score = result['health_score']
        
        # Background cho text
        cv2.rectangle(display_img, (10, 10), (390, 80), (0, 0, 0), -1)
        cv2.rectangle(display_img, (10, 10), (390, 80), color, 3)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, f"Trang thai: {status}", (20, 40),
                   font, 0.8, (255, 255, 255), 2)
        cv2.putText(display_img, f"Diem: {score:.1f}/100", (20, 70),
                   font, 0.7, (255, 255, 255), 2)
        
        return display_img
