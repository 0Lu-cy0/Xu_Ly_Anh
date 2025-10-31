# 🗂️ GỢI Ý DATASET - PHÂN LOẠI TRÁI CÂY TƯƠI VÀ HỎNG

## 🏆 Dataset chính được khuyến nghị

### 1. ⭐⭐⭐⭐⭐ **Fruits Fresh and Rotten Classification** (Kaggle)

**Link:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-classification

**Thông tin chi tiết:**
- **Số lượng:** 13,599 ảnh
- **Phân loại:** 
  - 🍎 Fresh Apples vs Rotten Apples
  - 🍌 Fresh Banana vs Rotten Banana  
  - 🍊 Fresh Oranges vs Rotten Oranges
- **Kích thước:** 100×100 pixels
- **Format:** JPG/PNG
- **Miễn phí:** ✅ 100%

**Ưu điểm:**
- ✅ Số lượng lớn, đủ để test
- ✅ Chất lượng tốt
- ✅ Rõ ràng: Tươi vs Hỏng
- ✅ Nền đơn giản, dễ xử lý
- ✅ Được sử dụng rộng rãi

**Cách tải:**

#### Phương pháp 1: Kaggle API (Khuyến nghị)

```bash
# Bước 1: Cài Kaggle CLI
pip install kaggle

# Bước 2: Đăng nhập Kaggle và lấy API token
# - Truy cập: https://www.kaggle.com/settings
# - Scroll xuống "API" → Click "Create New Token"
# - File kaggle.json sẽ được tải về

# Bước 3: Đặt file kaggle.json
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Bước 4: Tải dataset
cd "C:\Users\AMC\Desktop\Nam4\XuLyAnh\Code"
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-classification

# Bước 5: Giải nén
unzip fruits-fresh-and-rotten-classification.zip -d data/
# Hoặc dùng 7zip/WinRAR trên Windows
```

#### Phương pháp 2: Tải trực tiếp từ web

1. Truy cập: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-classification
2. Click "Download" (cần đăng nhập)
3. Giải nén vào thư mục `data/`

**Cấu trúc sau khi giải nén:**
```
dataset/
├── freshapples/      # Táo tươi
├── freshbanana/      # Chuối tươi
├── freshoranges/     # Cam tươi
├── rottenapples/     # Táo hỏng
├── rottenbanana/     # Chuối hỏng
└── rottenoranges/    # Cam hỏng
```

---

### 2. ⭐⭐⭐⭐ **Fresh and Rotten Fruits Dataset** (Roboflow)

**Link:** https://universe.roboflow.com/search?q=fresh+rotten+fruit

**Thông tin:**
- Nhiều dataset nhỏ khác nhau
- Có annotation (nếu cần)
- Dễ tải, dễ dùng
- Một số miễn phí

**Ưu điểm:**
- ✅ Tải nhanh
- ✅ Có sẵn augmentation
- ✅ Web interface dễ dùng

---

### 3. ⭐⭐⭐ **Fruit Quality Dataset** (Mendeley Data)

**Link:** https://data.mendeley.com/datasets/bdd69gyhv8/1

**Thông tin:**
- Dataset nghiên cứu
- Chất lượng cao
- Có metadata

---

### 4. ⭐⭐⭐⭐ **Tự tạo dataset** (Khuyến nghị cho hiểu rõ!)

#### Cách thu thập:

**A. Chụp ảnh trực tiếp:**
1. Đi chợ/siêu thị
2. Chụp trái cây tươi
3. Để trái cây hỏng (vài ngày)
4. Chụp lại trái cây đã hỏng

**B. Tìm từ Google Images:**
```
Từ khóa tìm kiếm:
- "fresh apple white background"
- "rotten apple isolated"
- "spoiled banana white background"
- "moldy orange"
- "bruised fruit"
```

**C. Dùng tool tự động (cẩn thận bản quyền!):**

```python
# Dùng bing-image-downloader
pip install bing-image-downloader

from bing_image_downloader import downloader

# Tải ảnh táo tươi
downloader.download(
    'fresh red apple white background', 
    limit=50, 
    output_dir='data/fresh/apple',
    adult_filter_off=True, 
    force_replace=False
)

# Tải ảnh táo hỏng
downloader.download(
    'rotten apple moldy spoiled', 
    limit=50, 
    output_dir='data/rotten/apple',
    adult_filter_off=True, 
    force_replace=False
)
```

**Yêu cầu ảnh:**
- ✅ Rõ nét, không mờ
- ✅ Nền đơn giản (trắng/đen)
- ✅ Trái cây chiếm phần lớn khung hình
- ✅ Kích thước: Ít nhất 200×200 pixels

---

## 📊 So sánh các nguồn dataset

| Nguồn | Số lượng | Chất lượng | Miễn phí | Dễ tải | Khuyến nghị |
|-------|----------|-----------|----------|--------|-------------|
| Kaggle (Sriramr) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Roboflow | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ Một số | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mendeley | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐ | ⭐⭐⭐ |
| Google Images | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅* | ⭐⭐⭐ | ⭐⭐⭐ |
| Tự chụp | ⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

*Lưu ý bản quyền

---

## 🚀 Hướng dẫn setup với Kaggle Dataset

### Bước 1: Tạo tài khoản Kaggle

1. Truy cập: https://www.kaggle.com
2. Sign up (miễn phí)
3. Xác nhận email

### Bước 2: Lấy API credentials

1. Đăng nhập Kaggle
2. Click avatar → Settings
3. Scroll xuống phần "API"
4. Click "Create New API Token"
5. File `kaggle.json` sẽ tự động tải về

### Bước 3: Cài đặt Kaggle CLI

```bash
pip install kaggle
```

### Bước 4: Config API token

**Windows:**
```bash
# Tạo thư mục .kaggle
mkdir C:\Users\AMC\.kaggle

# Copy file kaggle.json vào
copy kaggle.json C:\Users\AMC\.kaggle\
```

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Bước 5: Tải dataset

```bash
# Di chuyển vào thư mục dự án
cd "C:\Users\AMC\Desktop\Nam4\XuLyAnh\Code"

# Tải dataset (khoảng 500MB)
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-classification

# Giải nén
# Windows: Dùng 7zip hoặc WinRAR
# Linux/Mac: unzip fruits-fresh-and-rotten-classification.zip
```

### Bước 6: Tổ chức lại thư mục

```bash
# Tạo cấu trúc mới
mkdir data\fresh
mkdir data\rotten

# Di chuyển ảnh
move dataset\freshapples data\fresh\
move dataset\freshbanana data\fresh\
move dataset\freshoranges data\fresh\

move dataset\rottenapples data\rotten\
move dataset\rottenbanana data\rotten\
move dataset\rottenoranges data\rotten\
```

**Cấu trúc cuối cùng:**
```
data/
├── fresh/
│   ├── freshapples/
│   │   ├── fresh_apple_1.jpg
│   │   ├── fresh_apple_2.jpg
│   │   └── ...
│   ├── freshbanana/
│   └── freshoranges/
│
└── rotten/
    ├── rottenapples/
    ├── rottenbanana/
    └── rottenoranges/
```

---

## 🎯 Cách test với dataset

### Test đơn lẻ:

```bash
# Test với 1 ảnh táo tươi
python classify_fresh_rotten.py data/fresh/freshapples/fresh_apple_1.jpg

# Test với 1 ảnh táo hỏng
python classify_fresh_rotten.py data/rotten/rottenapples/rotten_apple_1.jpg
```

### Test hàng loạt:

```bash
# Test tất cả ảnh tươi
python classify_fresh_rotten.py data/fresh/freshapples/ --batch

# Test tất cả ảnh hỏng
python classify_fresh_rotten.py data/rotten/rottenapples/ --batch
```

---

## 📸 Mẫu ảnh trong dataset

### Ảnh TƯƠI (Fresh):
```
Đặc điểm:
✅ Màu sắc tươi sáng (đỏ, vàng, cam rực rỡ)
✅ Bề mặt bóng, phản chiếu ánh sáng
✅ Không có vết thâm, vết đen
✅ Hình dạng đều đặn
✅ Texture mịn
```

### Ảnh HỎNG (Rotten):
```
Đặc điểm:
❌ Màu sắc xỉn, sẫm (nâu, đen)
❌ Bề mặt khô, không bóng
❌ Có vết thâm, vết nấm mốc
❌ Hình dạng méo mó, co rúm
❌ Texture nhăn nheo, không đều
```

---

## 💡 Tips khi sử dụng dataset

### 1. Lọc ảnh kém chất lượng:

```python
import cv2
import os

def filter_quality(folder):
    """Lọc bỏ ảnh mờ, quá nhỏ"""
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"Xóa: {filename} (không đọc được)")
            os.remove(filepath)
            continue
        
        # Kiểm tra kích thước
        if img.shape[0] < 50 or img.shape[1] < 50:
            print(f"Xóa: {filename} (quá nhỏ)")
            os.remove(filepath)
            continue
        
        # Kiểm tra độ mờ (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            print(f"Xóa: {filename} (quá mờ)")
            os.remove(filepath)
```

### 2. Resize đồng loạt:

```python
def resize_all(folder, size=(300, 300)):
    """Resize tất cả ảnh về cùng kích thước"""
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        
        if img is not None:
            resized = cv2.resize(img, size)
            cv2.imwrite(filepath, resized)
            print(f"Đã resize: {filename}")
```

### 3. Augmentation (nếu cần thêm data):

```python
def augment_image(image):
    """Tạo biến thể của ảnh"""
    # Lật ngang
    flipped = cv2.flip(image, 1)
    
    # Xoay
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Thay đổi độ sáng
    brightness = 1.2
    bright = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    return [flipped, rotated, bright]
```

---

## 🎓 Tổng kết

### ✅ Dataset khuyến nghị:

**Kaggle - Fruits Fresh and Rotten Classification**

**Lý do:**
1. ✅ Số lượng lớn (13,599 ảnh)
2. ✅ Chất lượng tốt
3. ✅ Miễn phí 100%
4. ✅ Dễ tải với Kaggle API
5. ✅ Được sử dụng rộng rãi
6. ✅ Phân loại rõ ràng: Fresh vs Rotten

### 📝 Checklist chuẩn bị dataset:

- [ ] Tải dataset từ Kaggle
- [ ] Giải nén vào thư mục `data/`
- [ ] Tổ chức lại thư mục (fresh/ và rotten/)
- [ ] Test với vài ảnh mẫu
- [ ] Lọc ảnh kém chất lượng (nếu cần)
- [ ] Resize về kích thước chuẩn (nếu cần)

### 🚀 Bước tiếp theo:

```bash
# Sau khi có dataset, test ngay!
python classify_fresh_rotten.py data/fresh/freshapples/fresh_apple_1.jpg
```

---

**Chúc bạn thu thập dataset thành công! 📊**
