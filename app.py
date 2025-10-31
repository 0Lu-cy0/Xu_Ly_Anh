"""
Web App - Phân loại trái cây tươi/hỏng
Sử dụng Flask để tạo giao diện web
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# Import classifier
from src.fresh_rotten_classifier import FreshRottenClassifier

# Khởi tạo Flask app
app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
app.config['ALLOWED_EXTENSIONS'] = {
    'png', 'jpg', 'jpeg', 'jpe', 'jfif',  # JPEG variants
    'bmp', 'dib',  # Bitmap
    'tiff', 'tif',  # TIFF
    'webp',  # WebP
    'gif',  # GIF
    'ico',  # Icon
    'ppm', 'pgm', 'pbm', 'pnm',  # Portable formats
    'svg', 'svgz'  # SVG (vector)
}

# Tạo thư mục uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo classifier
classifier = FreshRottenClassifier()


def allowed_file(filename):
    """Kiểm tra định dạng file"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def cv2_to_base64(image):
    """Chuyển ảnh OpenCV sang base64 để hiển thị trên web"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý upload và phân loại ảnh"""
    try:
        # Kiểm tra file
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được upload'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        if not allowed_file(file.filename):
            allowed = ', '.join(sorted(app.config['ALLOWED_EXTENSIONS']))
            return jsonify({'error': f'Định dạng file không hợp lệ. Chỉ chấp nhận: {allowed}'}), 400
        
        # Lưu file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Đọc ảnh
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Không thể đọc ảnh'}), 400
        
        # Phân loại
        raw_result = classifier.classify(image)
        
        # Chuyển đổi numpy types sang Python native types
        def convert_to_native(obj):
            """Chuyển numpy types sang Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        raw_result = convert_to_native(raw_result)
        
        # Chuẩn hóa result để match với frontend
        result = {
            'label': raw_result['status'],  # FRESH/ROTTEN -> label
            'health_score': int(raw_result['health_score']),
            'confidence': f"{raw_result['confidence']*100:.1f}%",
            'reason': ', '.join(raw_result['reasons']) if raw_result['reasons'] else 'Trái cây trong tình trạng tốt',
            'metrics': {
                'color_health': raw_result['features']['color']['mean_saturation'],
                'texture_smoothness': 100 - raw_result['features']['texture']['mean_edge_intensity'],
                'surface_quality': raw_result['features']['smoothness']['bright_ratio'] * 100,
                'defect_count': raw_result['features']['defects']['num_spots']
            }
        }
        
        # Tạo visualization
        vis_images = {}
        
        # 1. Ảnh gốc
        vis_images['original'] = cv2_to_base64(image)
        
        # 2. HSV channels
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        vis_images['hue'] = cv2_to_base64(cv2.applyColorMap(h, cv2.COLORMAP_HSV))
        vis_images['saturation'] = cv2_to_base64(cv2.cvtColor(s, cv2.COLOR_GRAY2BGR))
        vis_images['value'] = cv2_to_base64(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR))
        
        # 3. Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        vis_images['edges'] = cv2_to_base64(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        
        # 4. Defect detection
        _, thresh = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Vẽ contours lên ảnh gốc
        defect_img = image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Chỉ vẽ vết lớn
                cv2.drawContours(defect_img, [cnt], -1, (0, 0, 255), 2)
        
        vis_images['defects'] = cv2_to_base64(defect_img)
        
        # Chuẩn bị response
        response = {
            'success': True,
            'result': result,
            'visualizations': vis_images,
            'filename': filename
        }
        
        print(f"\n✅ Phân loại thành công: {filename}")
        print(f"   Kết quả: {result.get('label', 'N/A')}")
        print(f"   Điểm: {result.get('health_score', 'N/A')}/100\n")
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_msg = f'Lỗi xử lý: {str(e)}'
        print(f"\n❌ LỖI: {error_msg}")
        traceback.print_exc()
        print()
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/info')
def info():
    """Thông tin về các kỹ thuật xử lý ảnh"""
    techniques = {
        'color_analysis': {
            'name': 'Phân tích màu sắc (HSV)',
            'description': 'Chuyển ảnh sang không gian màu HSV để phân tích Hue (màu sắc), Saturation (độ bão hòa), Value (độ sáng)',
            'purpose': 'Phát hiện vùng màu bất thường (nâu, đen) - dấu hiệu trái cây hỏng'
        },
        'texture_analysis': {
            'name': 'Phân tích texture (Sobel, Laplacian)',
            'description': 'Sử dụng toán tử Sobel và Laplacian để phát hiện biên (edges)',
            'purpose': 'Đánh giá độ nhăn, độ nhám bề mặt - trái hỏng thường có texture thô'
        },
        'morphology': {
            'name': 'Phép toán hình thái học',
            'description': 'Sử dụng Erosion, Dilation, Opening, Closing',
            'purpose': 'Loại bỏ nhiễu, làm nổi bật vùng khuyết tật'
        },
        'threshold': {
            'name': 'Ngưỡng hóa (Thresholding)',
            'description': 'Adaptive Thresholding để tách vùng tối/sáng',
            'purpose': 'Phát hiện vết thâm, vết đen trên trái cây'
        },
        'contour': {
            'name': 'Phát hiện đường viền',
            'description': 'Tìm và đếm các contours (đường bao)',
            'purpose': 'Đếm số lượng vết hỏng, tính diện tích vùng khuyết tật'
        }
    }
    
    return jsonify(techniques)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 WEB APP - PHÂN LOẠI TRÁI CÂY TƯƠI/HỎNG")
    print("="*70)
    print("\n📚 Kỹ thuật xử lý ảnh sử dụng:")
    print("   ✅ Phân tích màu sắc (HSV color space)")
    print("   ✅ Phát hiện biên (Sobel, Laplacian, Canny)")
    print("   ✅ Phân tích texture (entropy, variance)")
    print("   ✅ Phép toán hình thái học (morphological operations)")
    print("   ✅ Ngưỡng hóa thích ứng (adaptive thresholding)")
    print("   ✅ Phát hiện contours (defect detection)")
    
    print("\n🚀 Khởi động server...")
    print("   📍 URL: http://localhost:5000")
    print("   🛑 Dừng: Ctrl+C")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
