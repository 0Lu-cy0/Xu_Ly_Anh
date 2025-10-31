# 🗂️ Gợi Ý Bộ Dữ Liệu Trái Cây Để Huấn Luyện

## 📊 Các bộ dữ liệu được khuyến nghị

### 1. ⭐ Fruits 360 Dataset (Kaggle) - KHUYẾN NGHỊ NHẤT

**Link:** https://www.kaggle.com/datasets/moltean/fruits

**Thông tin:**
- **Số lượng:** 90,000+ ảnh của 131 loại trái cây
- **Kích thước ảnh:** 100x100 pixels
- **Nền:** Trắng, rất sạch
- **Định dạng:** JPG
- **Chia sẵn:** Train/Test
- **Miễn phí:** ✅ Hoàn toàn miễn phí

**Ưu điểm:**
- ✅ Số lượng ảnh rất lớn
- ✅ Chất lượng tốt, nền đơn giản
- ✅ Đã được chuẩn hóa sẵn
- ✅ Có tất cả loại trái cây bạn cần (táo, chuối, cam, nho, dưa hấu)
- ✅ Được sử dụng rộng rãi trong nghiên cứu

**Cách tải:**
```bash
# Cài Kaggle API
pip install kaggle

# Đăng nhập Kaggle và lấy API token từ https://www.kaggle.com/settings
# Đặt file kaggle.json vào:
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Tải dataset
kaggle datasets download -d moltean/fruits
unzip fruits.zip -d data/fruits360
```

**Cách sử dụng với code của bạn:**
```python
# Sao chép ảnh vào thư mục dự án
# Từ: fruits360/Training/Apple Red 1/ 
# Đến: data/train/táo/

# Chỉ lấy 5 loại trái cây cần thiết:
# - Apple (nhiều loại) → data/train/táo/
# - Banana → data/train/chuối/
# - Orange → data/train/cam/
# - Grape → data/train/nho/
# - Watermelon → data/train/dưa hấu/
```

---

### 2. 🍎 Fruit Images for Object Detection (Kaggle)

**Link:** https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection

**Thông tin:**
- **Số lượng:** 240 ảnh của 3 loại (apple, banana, orange)
- **Định dạng:** JPG với bounding boxes
- **Phù hợp:** Object detection, nhưng cũng dùng được cho classification

**Ưu điểm:**
- ✅ Ảnh thực tế hơn
- ✅ Nhiều góc độ khác nhau
- ✅ Có annotation nếu muốn làm detection sau này

---

### 3. 🥗 Fresh and Rotten Fruits/Vegetables (Kaggle)

**Link:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-classification

**Thông tin:**
- **Số lượng:** 13,000+ ảnh
- **Loại:** Trái cây tươi và hỏng
- **Phù hợp:** Nếu muốn mở rộng đề tài

**Ưu điểm:**
- ✅ Ảnh thực tế
- ✅ Có thể phân loại cả chất lượng trái cây

---

### 4. 🍇 Fruit Recognition Dataset (GitHub)

**Link:** https://github.com/Horea94/Fruit-Images-Dataset

**Thông tin:**
- **Nguồn:** Cùng tác giả với Fruits 360
- **Có thể clone:** `git clone https://github.com/Horea94/Fruit-Images-Dataset.git`

---

### 5. 🌐 Roboflow Universe - Fruit Datasets

**Link:** https://universe.roboflow.com/search?q=fruit

**Thông tin:**
- Nhiều dataset nhỏ về trái cây
- Có thể tải trực tiếp qua API
- Một số miễn phí, một số trả phí

**Ví dụ datasets:**
- Fruit Classification (2000+ ảnh)
- Fruit Detection Dataset
- Apple Orange Banana Dataset

---

### 6. 📸 Tự thu thập ảnh từ Google Images

**Cách làm:**

#### Phương pháp 1: Tải thủ công
```
1. Truy cập Google Images
2. Tìm kiếm:
   - "apple fruit white background"
   - "banana fruit isolated"
   - "orange fruit white background"
   - "grape fruit isolated"
   - "watermelon fruit white background"
3. Tải về 30-50 ảnh cho mỗi loại
4. Đặt vào thư mục tương ứng
```

#### Phương pháp 2: Dùng tool tự động (cẩn thận bản quyền!)

**Sử dụng `google-images-download` hoặc `bing-image-downloader`:**

```bash
# Cài đặt
pip install bing-image-downloader

# Tải ảnh
python -c "
from bing_image_downloader import downloader
downloader.download('apple fruit white background', limit=50, output_dir='data/train', image_format='jpg', verbose=True)
"
```

**⚠️ LƯU Ý:** Kiểm tra bản quyền trước khi sử dụng cho mục đích thương mại!

---

### 7. 🎓 UCI Machine Learning Repository

**Link:** https://archive.ics.uci.edu/ml/datasets.php

**Dataset:** Fruit Data with Colors (nhỏ, chỉ có thông tin màu sắc, không có ảnh)

---

## 🎯 Khuyến nghị cho dự án của bạn

### Lựa chọn tốt nhất: **Fruits 360 Dataset**

**Lý do:**
1. ✅ Số lượng lớn (>1000 ảnh/loại)
2. ✅ Chất lượng tốt, đồng nhất
3. ✅ Nền đơn giản → dễ trích xuất đặc trưng
4. ✅ Có đủ 5 loại trái cây bạn cần
5. ✅ Miễn phí, dễ tải

### Quy trình sử dụng Fruits 360:

```bash
# Bước 1: Tải dataset
kaggle datasets download -d moltean/fruits

# Bước 2: Giải nén
unzip fruits.zip

# Bước 3: Tổ chức lại
# Fruits 360 có nhiều loại Apple, chọn một số:
# - Apple Red 1, Apple Red 2, Apple Golden 1, ... → data/train/táo/
# - Banana → data/train/chuối/
# - Orange → data/train/cam/
# - Grape Blue, Grape White, ... → data/train/nho/
# - Watermelon → data/train/dưa hấu/
```

**Script Python tự động tổ chức:**

```python
import os
import shutil

# Mapping từ tên trong Fruits360 sang tên trong dự án
fruit_mapping = {
    'táo': ['Apple Red 1', 'Apple Red 2', 'Apple Red 3', 
            'Apple Golden 1', 'Apple Golden 2', 'Apple Granny Smith'],
    'chuối': ['Banana', 'Banana Red'],
    'cam': ['Orange'],
    'nho': ['Grape Blue', 'Grape Pink', 'Grape White'],
    'dưa hấu': ['Watermelon']
}

source_dir = 'fruits360/Training'
target_dir = 'data/train'

for target_fruit, source_fruits in fruit_mapping.items():
    target_path = os.path.join(target_dir, target_fruit)
    os.makedirs(target_path, exist_ok=True)
    
    for source_fruit in source_fruits:
        source_path = os.path.join(source_dir, source_fruit)
        if os.path.exists(source_path):
            for img_file in os.listdir(source_path):
                src = os.path.join(source_path, img_file)
                dst = os.path.join(target_path, f"{source_fruit}_{img_file}")
                shutil.copy(src, dst)
            print(f"✅ Đã copy {source_fruit} → {target_fruit}")

print("\n🎉 Hoàn thành!")
```

---

## 📏 Tiêu chí chọn dataset

### Dataset tốt cần có:
- ✅ Ít nhất 50-100 ảnh/loại (càng nhiều càng tốt)
- ✅ Ảnh rõ nét, không bị mờ
- ✅ Nền đơn giản (trắng, đen, hoặc đồng nhất)
- ✅ Trái cây chiếm phần lớn khung hình
- ✅ Đa dạng góc độ và ánh sáng
- ✅ Định dạng phổ biến (JPG, PNG)

### Tránh:
- ❌ Ảnh quá nhỏ (<100x100 pixels)
- ❌ Nhiều đối tượng trong 1 ảnh
- ❌ Nền quá phức tạp
- ❌ Ảnh bị che khuất nhiều
- ❌ Chất lượng kém, mờ

---

## 🚀 Hướng dẫn nhanh với Fruits 360

### 1. Tạo tài khoản Kaggle
- Truy cập: https://www.kaggle.com
- Đăng ký tài khoản miễn phí

### 2. Lấy API Token
- Vào: https://www.kaggle.com/settings
- Scroll xuống "API" → Click "Create New Token"
- File `kaggle.json` sẽ được tải về

### 3. Cài đặt Kaggle CLI

```bash
pip install kaggle

# Windows: Copy kaggle.json vào
# C:\Users\<YourUsername>\.kaggle\

# Linux/Mac:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Tải dataset

```bash
cd "C:\Users\AMC\Desktop\Nam4\XuLyAnh\Code"

# Tải Fruits 360
kaggle datasets download -d moltean/fruits

# Giải nén
unzip fruits.zip -d temp_fruits360

# (Hoặc dùng 7zip trên Windows nếu không có unzip)
```

### 5. Chạy script tổ chức ảnh

Tạo file `organize_fruits360.py` và chạy nó để tự động sao chép ảnh vào đúng thư mục.

---

## 💰 Chi phí

| Dataset | Miễn phí | Số lượng | Chất lượng |
|---------|----------|----------|------------|
| Fruits 360 | ✅ Miễn phí | ⭐⭐⭐⭐⭐ Rất nhiều | ⭐⭐⭐⭐⭐ Xuất sắc |
| Fruit Images for Object Detection | ✅ Miễn phí | ⭐⭐⭐ Trung bình | ⭐⭐⭐⭐ Tốt |
| Google Images | ✅ Miễn phí* | ⭐⭐⭐⭐ Tùy chỉnh | ⭐⭐⭐ Khác nhau |
| Roboflow | ⚠️ Một số miễn phí | ⭐⭐⭐ Khác nhau | ⭐⭐⭐⭐ Tốt |

*Lưu ý bản quyền

---

## 📞 Hỗ trợ

Nếu gặp khó khăn khi tải dataset:

1. **Không tải được từ Kaggle:**
   - Kiểm tra API token đã đúng chưa
   - Thử tải trực tiếp từ web: https://www.kaggle.com/datasets/moltean/fruits

2. **File quá lớn:**
   - Fruits 360 khoảng 800MB
   - Đảm bảo có đủ dung lượng ổ đĩa

3. **Không biết tổ chức ảnh:**
   - Xem file `organize_fruits360.py` tôi sẽ tạo

---

## 🎯 Tóm tắt

**Khuyến nghị:** Sử dụng **Fruits 360 Dataset** từ Kaggle

**Bước thực hiện:**
1. ✅ Đăng ký Kaggle
2. ✅ Tải Fruits 360 Dataset
3. ✅ Chạy script tổ chức ảnh
4. ✅ Huấn luyện model: `python main_train.py`
5. ✅ Đạt độ chính xác cao (>85%)

**Thời gian:** ~30 phút (bao gồm tải và tổ chức)

**Kết quả:** Dataset chất lượng cao, đủ lớn để huấn luyện model tốt!

---

*Cập nhật: October 23, 2025*
