# 🍎🍏 Đề tài: PHÂN LOẠI TRÁI CÂY TƯƠI VÀ HỎNG

## 📋 Giới thiệu

Dự án **phân loại trái cây tươi và hỏng** sử dụng **HOÀN TOÀN** kỹ thuật Xử lý Ảnh với OpenCV.

**KHÔNG SỬ DỤNG Machine Learning!**

### 🎯 Mục tiêu:
Phát triển hệ thống tự động đánh giá chất lượng trái cây dựa trên:
- ✅ Màu sắc (Color Analysis)
- ✅ Texture bề mặt (Texture Analysis)  
- ✅ Độ bóng/mịn (Surface Smoothness)
- ✅ Vết thâm/hư hỏng (Defect Detection)

### ✅ Phù hợp môn:
**Nhập môn Xử lý Ảnh** - 100% kỹ thuật Computer Vision thuần túy

---

## 🔧 Kỹ thuật sử dụng

### 1. Chuyển đổi không gian màu
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```
- BGR → HSV để phân tích màu sắc tốt hơn

### 2. Phân tích màu sắc
```python
cv2.inRange(hsv, lower_brown, upper_brown)  # Phát hiện màu nâu
```
- Tìm vết nâu, vết đen (dấu hiệu hư hỏng)
- Tính tỷ lệ pixel khớp màu

### 3. Phân tích Texture
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```
- Sobel: Phát hiện độ nhăn nheo
- Laplacian: Phát hiện thay đổi đột ngột
- Local Variance: Đánh giá độ đồng đều

### 4. Threshold
```python
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)
```
- Adaptive Thresholding để phát hiện vết thâm cục bộ

### 5. Morphological Operations
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```
- Opening/Closing để loại nhiễu và làm mịn

### 6. Contour Detection
```python
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, ...)
```
- Tìm và đếm vết hư hỏng
- Tính diện tích các vết

### 7. Histogram Analysis
```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
```
- Phân tích phân bố màu
- Tính Entropy, Contrast

### 8. Rule-based Classification
```python
health_score = 100.0
if color_features['dark_spots_ratio'] > 0.1:
    health_score -= penalty
if health_score >= 70:
    status = "TƯƠI"
```
- Hệ thống chấm điểm dựa trên ngưỡng
- **KHÔNG học từ dữ liệu!**

---

## 📊 Tiêu chí đánh giá

### Hệ thống chấm điểm (0-100):

| Điểm | Trạng thái | Mô tả |
|------|-----------|-------|
| 70-100 | ✅ **TƯƠI** | Màu sắc tươi sáng, bề mặt mịn, không vết thâm |
| 40-69 | ⚠️ **TRUNG BÌNH** | Có một số dấu hiệu hư, cần kiểm tra |
| 0-39 | ❌ **HỎNG** | Nhiều vết thâm, màu sẫm, bề mặt nhăn |

### Các yếu tố đánh giá:

**1. Màu sắc (30%):**
- Độ bão hòa (Saturation): Tươi = cao, Hỏng = thấp
- Độ sáng (Value): Tươi = sáng, Hỏng = tối
- Vết đen/nâu: Càng nhiều càng hỏng

**2. Texture (25%):**
- Edge Intensity: Tươi = mịn, Hỏng = nhăn
- Local Variance: Tươi = đều, Hỏng = không đều

**3. Độ mịn bề mặt (20%):**
- Entropy: Tươi = thấp, Hỏng = cao
- Bright spots: Tươi = nhiều (bóng), Hỏng = ít

**4. Vết hư hỏng (25%):**
- Số lượng vết thâm
- Diện tích vết hư

---

## 🚀 Cài đặt và Sử dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Chỉ cần:**
- opencv-python
- numpy
- pillow (cho GUI)
- matplotlib (cho visualization)

**KHÔNG CẦN:** scikit-learn (đã bỏ!)

### 2. Phân tích 1 ảnh

```bash
python classify_fresh_rotten.py anh_tao.jpg
```

**Kết quả:**
```
✅ TRẠNG THÁI: TƯƠI
📊 ĐIỂM SỨC KHỎE: 85.3/100
🎯 ĐỘ TIN CẬY: 87.5%

📝 CHI TIẾT ĐÁNH GIÁ:
   1. Màu sắc tươi sáng (+)
   2. Bề mặt mịn, ít nhăn (+)
   3. Không có vết thâm đáng kể (+)
```

### 3. Phân tích hàng loạt

```bash
python classify_fresh_rotten.py data/test/fresh/ --batch
```

---

## 📂 Cấu trúc dữ liệu

```
data/
├── fresh/              # Ảnh trái cây tươi
│   ├── apple_fresh_1.jpg
│   ├── banana_fresh_1.jpg
│   └── orange_fresh_1.jpg
│
└── rotten/             # Ảnh trái cây hỏng
    ├── apple_rotten_1.jpg
    ├── banana_rotten_1.jpg
    └── orange_rotten_1.jpg
```

---

## 📊 Dataset được đề xuất

### 🏆 Dataset khuyến nghị: **Fruits Fresh and Rotten**

**Link Kaggle:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-classification

**Thông tin:**
- ✅ 13,599 ảnh
- ✅ 6 loại trái cây: Apple, Banana, Orange, etc.
- ✅ Mỗi loại có 2 class: Fresh, Rotten
- ✅ Kích thước: 100x100 pixels
- ✅ Hoàn toàn miễn phí

**Cách tải:**

```bash
# Cài Kaggle API
pip install kaggle

# Tải dataset
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-classification

# Giải nén
unzip fruits-fresh-and-rotten-classification.zip -d data/
```

**Cấu trúc sau khi tải:**
```
dataset/
├── freshapples/
├── freshbanana/
├── freshoranges/
├── rottenapples/
├── rottenbanana/
└── rottenoranges/
```

### Dataset khác:

**2. Fresh vs Rotten Fruit (Roboflow)**
- Link: https://universe.roboflow.com/fresh-rotten-fruit
- Nhỏ hơn, dễ dùng hơn cho demo

**3. Tự chụp ảnh:**
- Chụp trái cây tươi và hỏng ở chợ/siêu thị
- Nền đơn giản (trắng/đen)
- Ít nhất 20-30 ảnh mỗi loại

---

## 🎯 Luồng xử lý

```
Ảnh đầu vào
    ↓
1. Tiền xử lý
   ├─ Resize 300x300
   └─ Gaussian Blur
    ↓
2. Phân tích màu sắc
   ├─ Chuyển BGR → HSV
   ├─ Tính mean/std của S, V
   ├─ Phát hiện vết đen (threshold)
   └─ Phát hiện vết nâu (inRange)
    ↓
3. Phân tích Texture
   ├─ Sobel edges
   ├─ Laplacian
   └─ Local variance
    ↓
4. Phân tích độ mịn
   ├─ Histogram entropy
   ├─ Contrast
   └─ Bright ratio
    ↓
5. Phát hiện vết hư
   ├─ Adaptive threshold
   ├─ Morphological ops
   └─ Contour detection
    ↓
6. Chấm điểm (Rule-based)
   ├─ Health score: 0-100
   ├─ Áp dụng các ngưỡng
   └─ Quyết định: Tươi/Trung bình/Hỏng
    ↓
Kết quả + Visualization
```

---

## 📸 Ví dụ kết quả

### Trái cây TƯƠI:
```
✅ Táo tươi:
   - Điểm: 92/100
   - Màu đỏ tươi, bề mặt bóng
   - Không có vết thâm
```

### Trái cây HỎNG:
```
❌ Táo hỏng:
   - Điểm: 25/100
   - Có vết nâu lớn
   - Bề mặt nhăn, không bóng
   - Màu sắc xỉn
```

---

## 🎓 Điểm mạnh của đề tài

### ✅ Phù hợp môn Xử lý Ảnh vì:

1. **Thuần túy Computer Vision:**
   - Không dùng ML/AI
   - Chỉ dùng OpenCV functions
   - Giải thích được từng bước

2. **Nhiều kỹ thuật CV:**
   - Color space conversion
   - Thresholding (global + adaptive)
   - Morphological operations
   - Edge detection (Sobel, Laplacian)
   - Contour analysis
   - Histogram analysis

3. **Thực tế và dễ hiểu:**
   - Ứng dụng rõ ràng
   - Dễ demo, minh họa
   - Kết quả trực quan

4. **Không cần training:**
   - Chạy ngay, không cần data lớn
   - Dễ điều chỉnh tham số
   - Debug dễ dàng

---

## 📝 Báo cáo đề xuất

### Cấu trúc báo cáo:

**1. Giới thiệu**
- Bài toán: Phân loại trái cây tươi/hỏng
- Ý nghĩa thực tế
- Mục tiêu: Chỉ dùng CV, không dùng ML

**2. Cơ sở lý thuyết**
- Không gian màu HSV
- Sobel, Laplacian operators
- Morphological operations
- Adaptive thresholding
- Contour detection

**3. Phương pháp**
- Sơ đồ luồng xử lý
- Giải thích từng bước
- Công thức tính toán
- Hệ thống chấm điểm

**4. Thực nghiệm**
- Dataset sử dụng
- Kết quả demo
- So sánh các trường hợp

**5. Kết luận**
- Ưu điểm: Đơn giản, thực tế
- Hạn chế: Độ chính xác phụ thuộc ngưỡng
- Hướng phát triển: Có thể thêm ML sau này

---

## 💡 Tips thuyết trình

### Nên làm:
- ✅ Demo trực tiếp với ảnh thật
- ✅ Hiển thị từng bước xử lý (HSV, edges, threshold)
- ✅ Giải thích rõ công thức
- ✅ So sánh ảnh tươi vs hỏng

### Không nên:
- ❌ Đề cập đến ML/AI
- ❌ Nói về "training" hay "model"
- ❌ Dùng thuật ngữ phức tạp không cần thiết

---

## 🆚 So sánh với đề tài cũ

| Tiêu chí | Nhận diện loại trái cây | Phân loại Tươi/Hỏng |
|----------|------------------------|---------------------|
| **Bài toán** | 5 classes (táo, chuối, ...) | 2 classes (tươi, hỏng) |
| **Khó** | Phân biệt loại khó hơn | Đơn giản hơn |
| **Phù hợp CV** | ⭐⭐⭐ Cần ML cho tốt | ⭐⭐⭐⭐⭐ Hoàn hảo |
| **Thực tế** | ⭐⭐⭐ Ít ứng dụng | ⭐⭐⭐⭐⭐ Rất thực tế |
| **Demo** | ⭐⭐⭐ Khó giải thích | ⭐⭐⭐⭐⭐ Trực quan |

---

## 🎯 Kết luận

Đề tài **"Phân loại Trái cây Tươi và Hỏng"** là lựa chọn **HOÀN HẢO** cho môn **Nhập môn Xử lý Ảnh**:

- ✅ 100% kỹ thuật Computer Vision
- ✅ KHÔNG dùng Machine Learning
- ✅ Nhiều kỹ thuật để trình bày
- ✅ Thực tế, dễ hiểu, dễ demo
- ✅ Không cần training, chạy ngay
- ✅ Code đơn giản, dễ giải thích

**Điểm mạnh nhất:** Giáo viên sẽ thấy rõ bạn hiểu và áp dụng được các kỹ thuật xử lý ảnh!

---

**Chúc bạn thành công với đồ án! 🎉**
