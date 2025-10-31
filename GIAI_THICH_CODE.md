# 🔬 Giải Thích Chi Tiết Code

Tài liệu này giải thích chi tiết cách hoạt động của từng phần trong dự án.

---

## 📋 Mục lục

1. [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
2. [Module xử lý ảnh](#module-xử-lý-ảnh)
3. [Module tải dữ liệu](#module-tải-dữ-liệu)
4. [Module huấn luyện](#module-huấn-luyện)
5. [Luồng hoạt động](#luồng-hoạt-động)

---

## 🏗️ Tổng quan kiến trúc

### Sơ đồ luồng xử lý

```
Ảnh đầu vào
    ↓
[1. Tiền xử lý]
    ├─ Resize về 200x200
    ├─ Khử nhiễu (Gaussian Blur)
    └─ Chuẩn hóa
    ↓
[2. Trích xuất đặc trưng]
    ├─ Đặc trưng màu sắc (Color Histogram)
    ├─ Đặc trưng hình dạng (Shape Features)
    └─ Đặc trưng texture (Texture Features)
    ↓
[3. Chuẩn hóa đặc trưng]
    └─ StandardScaler
    ↓
[4. Phân loại (ML Model)]
    └─ SVM / KNN / Decision Tree / Random Forest
    ↓
Kết quả dự đoán
```

---

## 🖼️ Module xử lý ảnh

File: `utils/image_processing.py`

### 1. Hàm `load_image()`

```python
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    return image
```

**Giải thích:**
- `cv2.imread()`: Đọc ảnh từ file, trả về array NumPy
- Format mặc định: BGR (Blue-Green-Red), khác RGB!
- Trả về `None` nếu file không tồn tại → cần kiểm tra

**Ví dụ output:**
```python
# Shape: (height, width, channels)
# Ví dụ: (480, 640, 3) - ảnh 640x480 pixels, 3 kênh màu
```

---

### 2. Hàm `preprocess_image()`

```python
def preprocess_image(image, size=(200, 200)):
    # 1. Resize ảnh
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    # 2. Khử nhiễu
    denoised = cv2.GaussianBlur(resized, (5, 5), 0)
    
    return denoised
```

**Giải thích từng bước:**

#### Bước 1: Resize
```python
cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
```
- **Mục đích**: Chuẩn hóa kích thước → đặc trưng có độ dài cố định
- **INTER_AREA**: Phương pháp nội suy tốt cho việc thu nhỏ ảnh
- **Tại sao 200x200?**: Đủ lớn để giữ thông tin, đủ nhỏ để xử lý nhanh

#### Bước 2: Gaussian Blur
```python
cv2.GaussianBlur(image, (5, 5), 0)
```
- **Mục đích**: Làm mờ ảnh để giảm nhiễu
- **(5, 5)**: Kích thước kernel (càng lớn càng mờ)
- **0**: Sigma tự động tính từ kernel size
- **Công thức Gaussian**: $G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$

**Trước và sau:**
```
Trước:  ████▓▓▒▒░░  (nhiễu, gồ ghề)
Sau:    ████████░░  (mượt mà hơn)
```

---

### 3. Hàm `extract_color_histogram()`

```python
def extract_color_histogram(image, bins=32):
    # Chuyển sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính histogram cho mỗi kênh
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    return np.array(hist_features)
```

**Giải thích chi tiết:**

#### Tại sao dùng HSV thay vì BGR?

**BGR (Blue-Green-Red):**
- 3 kênh phụ thuộc lẫn nhau
- Bị ảnh hưởng nhiều bởi ánh sáng

**HSV (Hue-Saturation-Value):**
- **H (Hue)**: Màu sắc thực (0-180°) - đỏ, xanh, vàng...
- **S (Saturation)**: Độ bão hòa (0-255) - màu đậm hay nhạt
- **V (Value)**: Độ sáng (0-255) - sáng hay tối

```
    H: Loại màu
    │
    ├─ 0-20°:   Đỏ
    ├─ 20-40°:  Cam
    ├─ 40-80°:  Vàng/Xanh lá
    ├─ 80-140°: Xanh lục
    └─ 140-180°: Xanh dương/Tím
```

#### Histogram là gì?

Histogram đếm số lượng pixels có cùng giá trị màu.

**Ví dụ:**
```
Giá trị:  0-31  32-63  64-95  96-127 128-159 160-191 192-255
Số pixel:  50    120    200     150     80      60      40
          ▂     ▅      █       ▆      ▃       ▂       ▁
```

#### Chuẩn hóa histogram

```python
hist = cv2.normalize(hist, hist)
```
- Chia mỗi giá trị cho tổng → tất cả giá trị trong [0, 1]
- Mục đích: Không phụ thuộc vào kích thước ảnh

**Output:**
- Shape: `(bins * 3,)` = (16 * 3,) = (48,) với bins=16
- Ví dụ: `[0.12, 0.08, ..., 0.05]` (48 số thực)

---

### 4. Hàm `extract_shape_features()`

```python
def extract_shape_features(image):
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 4. Tính các đặc trưng
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    # ... và các đặc trưng khác
```

**Giải thích từng bước:**

#### Bước 1: Chuyển sang ảnh xám
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
- Từ 3 kênh (B, G, R) → 1 kênh (Gray)
- Công thức: $Gray = 0.299R + 0.587G + 0.114B$

#### Bước 2: Otsu's Threshold
```python
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**Mục đích**: Tách đối tượng khỏi nền (foreground vs background)

**Otsu's method** tự động tìm ngưỡng tối ưu:
- Phân tích histogram
- Tìm điểm chia tối ưu giữa 2 lớp
- Không cần chọn ngưỡng thủ công!

```
Ảnh xám:     ░░▒▒▓▓████▓▓▒▒░░
                    ↓ (ngưỡng = 128)
Ảnh nhị phân: ░░░░░░████░░░░░░
             (0=đen)   (255=trắng)
```

#### Bước 3: Tìm Contours

**Contour** là đường bao quanh đối tượng.

```python
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

- `RETR_EXTERNAL`: Chỉ lấy contour ngoài cùng
- `CHAIN_APPROX_SIMPLE`: Nén contour (bỏ điểm thừa)

**Hình ảnh hóa:**
```
██████          Contour bao quanh:
██  ██          ┌─────┐
██  ██    →     │     │
██████          └─────┘
```

#### Bước 4: Các đặc trưng hình học

**a) Diện tích (Area)**
```python
area = cv2.contourArea(largest_contour)
```
- Đếm số pixels bên trong contour
- Trái cây lớn → area lớn

**b) Chu vi (Perimeter)**
```python
perimeter = cv2.arcLength(largest_contour, True)
```
- Độ dài đường viền
- `True` = contour đóng

**c) Độ tròn (Circularity)**
```python
circularity = 4 * π * area / (perimeter²)
```

Công thức: $C = \frac{4\pi A}{P^2}$

- Hình tròn hoàn hảo: C = 1.0
- Hình vuông: C ≈ 0.785
- Hình dài: C < 0.5

```
Ví dụ:
● Táo (tròn):      C ≈ 0.85
▬ Chuối (dài):     C ≈ 0.25
◆ Dưa hấu (tròn):  C ≈ 0.90
```

**d) Tỷ lệ khung (Aspect Ratio)**
```python
x, y, w, h = cv2.boundingRect(largest_contour)
aspect_ratio = w / h
```

- Hình rộng: AR > 1 (chuối: AR ≈ 2-3)
- Hình cao: AR < 1
- Hình vuông: AR ≈ 1 (táo, cam: AR ≈ 0.9-1.1)

**e) Hu Moments**

Đây là 7 đặc trưng bất biến với:
- Phép quay (rotation)
- Phép tỉ lệ (scale)
- Phép đối xứng (reflection - một số moment)

```python
moments = cv2.moments(largest_contour)
hu_moments = cv2.HuMoments(moments)
```

Hu moments được tính từ moment hình học:

$M_{pq} = \sum_{x,y} x^p y^q I(x,y)$

**Output của hàm:**
- Shape: `(7,)` 
- Ví dụ: `[1250.5, 3.2, 0.85, 1.05, 0.68, -2.1, 1.8]`

---

### 5. Hàm `extract_texture_features()`

```python
def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Tính thống kê
    features = []
    for edge in [sobelx, sobely, laplacian]:
        features.extend([
            np.mean(np.abs(edge)),
            np.std(edge),
            np.max(np.abs(edge))
        ])
    
    return np.array(features)
```

**Giải thích:**

#### Sobel Operator

Tìm biên (edges) bằng cách tính gradient:

**Sobel X (biên dọc):**
```
Kernel:        Ảnh:
[-1 0 +1]      ░░████
[-2 0 +2]  *   ░░████  =  Biên dọc
[-1 0 +1]      ░░████
```

**Sobel Y (biên ngang):**
```
Kernel:        Ảnh:
[-1 -2 -1]     ░░░░
[ 0  0  0]  *  ████  =  Biên ngang
[+1 +2 +1]     ████
```

Gradient magnitude: $G = \sqrt{G_x^2 + G_y^2}$

#### Laplacian

Đạo hàm bậc 2, phát hiện thay đổi đột ngột:

```
Kernel:
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

**So sánh:**
- **Sobel**: Nhạy với hướng của biên
- **Laplacian**: Không phụ thuộc hướng, phát hiện vùng thay đổi nhanh

#### Thống kê từ edges

```python
features.extend([
    np.mean(np.abs(edge)),    # Cường độ biên trung bình
    np.std(edge),             # Độ biến thiên
    np.max(np.abs(edge))      # Biên mạnh nhất
])
```

**Ý nghĩa:**
- Trái cây có texture mịn: mean, std thấp (cam, táo)
- Trái cây có texture thô: mean, std cao (dưa hấu có vân)

**Output:**
- Shape: `(9,)` - 3 operators × 3 thống kê
- Ví dụ: `[12.5, 8.3, 45.2, 15.1, 9.8, 52.3, 18.7, 11.2, 60.5]`

---

### 6. Hàm `extract_all_features()`

```python
def extract_all_features(image):
    processed = preprocess_image(image)
    
    color_hist = extract_color_histogram(processed, bins=16)
    shape_feat = extract_shape_features(processed)
    texture_feat = extract_texture_features(processed)
    
    all_features = np.concatenate([color_hist, shape_feat, texture_feat])
    
    return all_features
```

**Kết hợp đặc trưng:**

```
Vector đặc trưng cuối cùng:
├─ Color Histogram:  48 chiều (16 bins × 3 kênh)
├─ Shape Features:   7 chiều
└─ Texture Features: 9 chiều
   ───────────────────────────
   TỔNG:               64 chiều
```

**Ví dụ vector đặc trưng:**
```python
[0.12, 0.08, ..., 0.05,    # 48 giá trị color
 1250.5, 3.2, 0.85, ...,   # 7 giá trị shape
 12.5, 8.3, 45.2, ...]     # 9 giá trị texture
```

---

## 📦 Module tải dữ liệu

File: `utils/data_loader.py`

### Dictionary phân loại

```python
FRUIT_CLASSES = {
    0: 'Táo (Apple)',
    1: 'Chuối (Banana)', 
    2: 'Cam (Orange)',
    3: 'Nho (Grape)',
    4: 'Dưa hấu (Watermelon)'
}
```

- Key: Nhãn số (integer label) cho ML
- Value: Tên hiển thị

### Hàm `load_dataset()`

```python
def load_dataset(data_dir, label):
    features = []
    labels = []
    
    for filename in os.listdir(data_dir):
        image = cv2.imread(filepath)
        feat = extract_all_features(image)
        features.append(feat)
        labels.append(label)
    
    return np.array(features), np.array(labels)
```

**Input:**
- `data_dir`: `"data/train/táo"`
- `label`: `0`

**Output:**
- `features`: Array shape `(n_images, 64)`
- `labels`: Array shape `(n_images,)`

**Ví dụ:**
```python
# 30 ảnh táo
features.shape = (30, 64)  # 30 mẫu, mỗi mẫu 64 đặc trưng
labels.shape = (30,)       # [0, 0, 0, ..., 0]
```

### Hàm `load_all_datasets()`

```python
def load_all_datasets(base_dir):
    all_features = []
    all_labels = []
    
    for label, fruit_name in FRUIT_CLASSES.items():
        folder_name = fruit_name.split('(')[0].strip().lower()
        folder_path = os.path.join(base_dir, folder_name)
        
        features, labels = load_dataset(folder_path, label)
        all_features.append(features)
        all_labels.append(labels)
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y
```

**Ví dụ output:**
```python
X.shape = (150, 64)  # 150 ảnh tổng cộng, mỗi ảnh 64 đặc trưng
y.shape = (150,)     # [0,0,...,1,1,...,2,2,...,3,3,...,4,4,...]
```

---

## 🎓 Module huấn luyện

File: `src/train_model.py`

### Class `FruitClassifier`

#### 1. Khởi tạo

```python
def __init__(self, model_type='svm'):
    self.scaler = StandardScaler()
    
    if model_type == 'svm':
        self.model = SVC(kernel='rbf', C=10, gamma='scale')
```

**StandardScaler:**
```python
# Trước chuẩn hóa:
X = [100, 0.5, 1500]

# Sau chuẩn hóa:
X_scaled = [-0.5, 0.2, 1.3]
```

Công thức: $X_{scaled} = \frac{X - \mu}{\sigma}$

**Tại sao cần chuẩn hóa?**
- Các đặc trưng có thang đo khác nhau (area: 1000, circularity: 0.8)
- ML algorithms nhạy với scale
- Chuẩn hóa giúp model học tốt hơn

#### 2. Thuật toán SVM

```python
SVC(kernel='rbf', C=10, gamma='scale')
```

**Support Vector Machine:**

Tìm siêu phẳng (hyperplane) tách các lớp với khoảng cách lớn nhất (margin).

```
Không gian 2D:

    x  Táo           │         o Cam
       x             │              o
         x           │           o
    x      x    ─────│─────   o
                     │      o
         x           │           o
       x             │         o
```

**Tham số:**
- `kernel='rbf'`: Radial Basis Function - xử lý dữ liệu phi tuyến
  - $K(x, x') = \exp(-\gamma ||x - x'||^2)$
- `C=10`: Độ cứng của margin (cao = ít sai lầm, nhưng có thể overfit)
- `gamma='scale'`: Tự động tính = 1 / (n_features × X.var())

#### 3. Thuật toán KNN

```python
KNeighborsClassifier(n_neighbors=5, weights='distance')
```

**K-Nearest Neighbors:**

Dự đoán dựa trên K láng giềng gần nhất.

```
Test point: ?

    x (táo)
       x           ? ← cần dự đoán
    x     o (cam)
         o
    x       o

5 láng giềng gần nhất:
- 3 táo (x)
- 2 cam (o)
→ Kết quả: Táo (theo đa số)
```

**Tham số:**
- `n_neighbors=5`: Xét 5 láng giềng
- `weights='distance'`: Láng giềng gần có trọng số lớn hơn

#### 4. Hàm `train()`

```python
def train(self, X_train, y_train, X_val=None, y_val=None):
    # Chuẩn hóa
    X_train_scaled = self.scaler.fit_transform(X_train)
    
    # Huấn luyện
    self.model.fit(X_train_scaled, y_train)
    
    # Đánh giá
    train_pred = self.model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
```

**Quy trình:**
1. **fit_transform**: Học mean, std từ train data → chuẩn hóa
2. **fit**: Huấn luyện model
3. **predict**: Dự đoán
4. **accuracy_score**: So sánh prediction vs ground truth

#### 5. Hàm `predict()`

```python
def predict(self, features):
    features_scaled = self.scaler.transform(features.reshape(1, -1))
    prediction = self.model.predict(features_scaled)[0]
    
    if hasattr(self.model, 'predict_proba'):
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
    
    return prediction, confidence
```

**Lưu ý:**
- `transform` (KHÔNG phải fit_transform!) - dùng mean, std đã học
- `reshape(1, -1)` - chuyển từ (64,) thành (1, 64) vì model cần 2D array

---

## 🔄 Luồng hoạt động

### 1. Training Flow

```
[Bắt đầu]
    ↓
Tải ảnh từ data/train/
    ↓
Với mỗi ảnh:
    ├─ Tiền xử lý (resize, blur)
    ├─ Trích xuất color histogram
    ├─ Trích xuất shape features
    ├─ Trích xuất texture features
    └─ Kết hợp thành vector 64 chiều
    ↓
Thu được ma trận X (n_samples, 64)
và vector y (n_samples,)
    ↓
Chia train/validation (80/20)
    ↓
Với mỗi thuật toán (SVM, KNN, DT, RF):
    ├─ Chuẩn hóa dữ liệu (StandardScaler)
    ├─ Huấn luyện model
    ├─ Đánh giá accuracy
    └─ Lưu kết quả
    ↓
Chọn model tốt nhất
    ↓
Lưu model vào file .pkl
    ↓
[Kết thúc]
```

### 2. Prediction Flow

```
[Bắt đầu]
    ↓
Tải model từ file .pkl
    ↓
Đọc ảnh test
    ↓
Tiền xử lý ảnh
    ↓
Trích xuất 64 đặc trưng
    ↓
Chuẩn hóa đặc trưng (dùng scaler đã lưu)
    ↓
Đưa vào model
    ↓
Model dự đoán nhãn + xác suất
    ↓
Chuyển nhãn số thành tên trái cây
    ↓
Hiển thị kết quả
    ↓
[Kết thúc]
```

---

## 🧮 Toán học đằng sau

### 1. Gaussian Blur

Công thức kernel Gaussian 2D:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Kernel 5×5 ví dụ (σ=1):
```
[1  4  6  4  1]
[4 16 24 16  4]
[6 24 36 24  6]  × (1/256)
[4 16 24 16  4]
[1  4  6  4  1]
```

### 2. Sobel Operator

Gradient theo X:
$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I$$

Gradient theo Y:
$$G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$

Magnitude:
$$G = \sqrt{G_x^2 + G_y^2}$$

### 3. StandardScaler

$$z = \frac{x - \mu}{\sigma}$$

Trong đó:
- $\mu$ = mean
- $\sigma$ = standard deviation
- $z$ = giá trị đã chuẩn hóa

### 4. Accuracy

$$\text{Accuracy} = \frac{\text{Số dự đoán đúng}}{\text{Tổng số mẫu}}$$

$$= \frac{TP + TN}{TP + TN + FP + FN}$$

---

## 💡 Tips Optimization

### 1. Giảm số chiều đặc trưng

```python
# Giảm bins trong histogram
color_hist = extract_color_histogram(processed, bins=8)  # từ 16 → 8
# 24 chiều thay vì 48 chiều
```

### 2. Sử dụng PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=32)  # Giảm từ 64 → 32 chiều
X_reduced = pca.fit_transform(X)
```

### 3. Grid Search để tìm tham số tốt

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
```

---

## 🎯 Tổng kết

**Pipeline hoàn chỉnh:**

```
Ảnh → Tiền xử lý → Trích xuất đặc trưng → Chuẩn hóa → ML Model → Dự đoán
```

**Các kỹ thuật chính:**
1. ✅ Color Histogram (HSV space)
2. ✅ Shape Analysis (contours, moments)
3. ✅ Texture Analysis (Sobel, Laplacian)
4. ✅ Feature Scaling (StandardScaler)
5. ✅ Classification (SVM, KNN, Trees)

**Ưu điểm:**
- ✅ Không cần GPU
- ✅ Dễ hiểu, dễ implement
- ✅ Chạy nhanh
- ✅ Phù hợp cho người mới học

**Hạn chế:**
- ⚠️ Độ chính xác không cao bằng Deep Learning
- ⚠️ Cần thiết kế đặc trưng thủ công
- ⚠️ Nhạy với điều kiện chụp ảnh

---

*Tài liệu này giải thích chi tiết code cho môn học Nhập môn Xử lý Ảnh*
