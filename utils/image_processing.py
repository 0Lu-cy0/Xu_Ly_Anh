"""
Module xử lý ảnh cơ bản
Chứa các hàm tiền xử lý và trích xuất đặc trưng từ ảnh
"""

import cv2
import numpy as np

def load_image(image_path):
    """
    Đọc ảnh từ file
    
    Args:
        image_path (str): Đường dẫn đến file ảnh
    
    Returns:
        numpy.ndarray: Ảnh đã đọc
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    return image


def preprocess_image(image, size=(200, 200)):
    """
    Tiền xử lý ảnh: resize, khử nhiễu, cân bằng histogram
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào
        size (tuple): Kích thước mong muốn (width, height)
    
    Returns:
        numpy.ndarray: Ảnh đã được xử lý
    """
    # Resize ảnh về kích thước chuẩn
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    # Khử nhiễu bằng Gaussian Blur
    denoised = cv2.GaussianBlur(resized, (5, 5), 0)
    
    return denoised


def extract_color_histogram(image, bins=32):
    """
    Trích xuất đặc trưng histogram màu sắc từ ảnh
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR)
        bins (int): Số bins cho histogram
    
    Returns:
        numpy.ndarray: Vector đặc trưng histogram đã được chuẩn hóa
    """
    # Chuyển sang không gian màu HSV (tốt hơn cho phân tích màu sắc)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính histogram cho mỗi kênh
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_shape_features(image):
    """
    Trích xuất đặc trưng hình dạng từ ảnh
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR)
    
    Returns:
        numpy.ndarray: Vector đặc trưng hình dạng
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng threshold để tạo ảnh nhị phân
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(7)
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tính các đặc trưng hình học
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Tránh chia cho 0
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # Extent (tỷ lệ diện tích contour / diện tích bounding box)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    
    # Hu Moments (7 đặc trưng bất biến)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform để giảm độ lớn của Hu moments
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # Kết hợp các đặc trưng
    shape_features = np.array([area, perimeter, circularity, aspect_ratio, extent])
    shape_features = np.concatenate([shape_features, hu_moments[:2]])  # Chỉ lấy 2 Hu moments đầu
    
    return shape_features


def extract_texture_features(image):
    """
    Trích xuất đặc trưng texture sử dụng Sobel và Laplacian
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR)
    
    Returns:
        numpy.ndarray: Vector đặc trưng texture
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel edges (gradient theo hướng X và Y)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Tính các thống kê từ edges
    features = []
    for edge in [sobelx, sobely, laplacian]:
        features.extend([
            np.mean(np.abs(edge)),
            np.std(edge),
            np.max(np.abs(edge))
        ])
    
    return np.array(features)


def extract_all_features(image):
    """
    Trích xuất tất cả các đặc trưng từ ảnh
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR)
    
    Returns:
        numpy.ndarray: Vector đặc trưng tổng hợp
    """
    # Tiền xử lý ảnh
    processed = preprocess_image(image)
    
    # Trích xuất các đặc trưng
    color_hist = extract_color_histogram(processed, bins=16)  # Giảm bins để giảm số chiều
    shape_feat = extract_shape_features(processed)
    texture_feat = extract_texture_features(processed)
    
    # Kết hợp tất cả đặc trưng
    all_features = np.concatenate([color_hist, shape_feat, texture_feat])
    
    return all_features


def draw_contours(image):
    """
    Vẽ contours lên ảnh để hiển thị
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR)
    
    Returns:
        numpy.ndarray: Ảnh có vẽ contours
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result
