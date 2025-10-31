# 📖 Hướng Dẫn Sử Dụng Chi Tiết

## 🎯 Mục lục

1. [Cài đặt môi trường](#cài-đặt-môi-trường)
2. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
3. [Huấn luyện model](#huấn-luyện-model)
4. [Sử dụng model để dự đoán](#sử-dụng-model-để-dự-đoán)
5. [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## 🔧 Cài đặt môi trường

### Bước 1: Kiểm tra Python

Mở terminal/command prompt và gõ:

```bash
python --version
```

Cần có Python 3.7 trở lên. Nếu chưa có, tải tại: https://www.python.org/

### Bước 2: Tạo môi trường ảo (Khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
# Di chuyển vào thư mục dự án
cd Code

# Cài đặt thư viện
pip install -r requirements.txt
```

Nếu gặp lỗi, cài từng thư viện:

```bash
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install pillow
pip install matplotlib
```

---

## 📂 Chuẩn bị dữ liệu

### Bước 1: Tạo cấu trúc thư mục

Cấu trúc thư mục cần có dạng:

```
data/
├── train/
│   ├── táo/          # 20-50 ảnh táo
│   ├── chuối/        # 20-50 ảnh chuối
│   ├── cam/          # 20-50 ảnh cam
│   ├── nho/          # 20-50 ảnh nho
│   └── dưa hấu/      # 20-50 ảnh dưa hấu
└── test/
    ├── táo/          # 5-10 ảnh táo để test
    ├── chuối/
    ├── cam/
    ├── nho/
    └── dưa hấu/
```

### Bước 2: Thu thập ảnh

#### Cách 1: Tìm ảnh trên Internet
- Google Images: Tìm "apple fruit", "banana fruit", v.v.
- Datasets miễn phí: Kaggle, UCI Machine Learning Repository
- Lưu ý bản quyền!

#### Cách 2: Chụp ảnh thật
- Chụp nhiều góc độ khác nhau
- Chụp với ánh sáng khác nhau
- Nền đơn giản (màu trắng hoặc đen)

### Bước 3: Yêu cầu về ảnh

✅ **Nên:**
- Ảnh rõ nét, không bị mờ
- Kích thước tối thiểu: 200x200 pixels
- Format: JPG, PNG, BMP
- Nền đơn giản
- Trái cây chiếm phần lớn ảnh

❌ **Không nên:**
- Ảnh quá nhỏ hoặc quá mờ
- Nhiều trái cây khác nhau trong 1 ảnh
- Nền quá phức tạp
- Ảnh bị che khuất nhiều

### Bước 4: Đặt ảnh vào thư mục

**Quan trọng:** Tên thư mục phải đúng:
- `táo` (chữ thường, có dấu)
- `chuối` (chữ thường, có dấu)
- `cam` (chữ thường, không dấu)
- `nho` (chữ thường, không dấu)
- `dưa hấu` (chữ thường, có dấu, có khoảng trắng)

Ví dụ đặt ảnh:
```
data/train/táo/apple1.jpg
data/train/táo/apple2.jpg
data/train/táo/apple3.jpg
...
```

---

## 🎓 Huấn luyện model

### Bước 1: Chạy script huấn luyện

```bash
python main_train.py
```

### Bước 2: Quan sát quá trình

Chương trình sẽ hiển thị:

```
============================================================
CHƯƠNG TRÌNH HUẤN LUYỆN MODEL NHẬN DIỆN TRÁI CÂY
============================================================

📂 Đang tải dữ liệu huấn luyện...
Đang xử lý 30 ảnh từ data/train/táo...
Đã tải 30 mẫu cho Táo (Apple)
Đang xử lý 25 ảnh từ data/train/chuối...
Đã tải 25 mẫu cho Chuối (Banana)
...

✅ Đã tải 150 mẫu dữ liệu
   Số chiều đặc trưng: 64
   Phân bố nhãn: {0: 30, 1: 25, 2: 35, 3: 28, 4: 32}

📊 Chia dữ liệu:
   - Training: 120 mẫu
   - Validation: 30 mẫu

🔬 Bắt đầu so sánh các thuật toán...

=== Huấn luyện model SVM ===
Số mẫu huấn luyện: 120
Độ chính xác trên tập huấn luyện: 0.9583
Độ chính xác trên tập validation: 0.8667

=== Huấn luyện model KNN ===
...
```

### Bước 3: Xem kết quả

Cuối cùng sẽ có bảng tổng kết:

```
============================================================
BẢNG TỔNG KẾT KẾT QUẢ
============================================================
Model                     Độ chính xác        
------------------------------------------------------------
Random Forest             0.9000 (90.00%)
SVM                       0.8667 (86.67%)
K-Nearest Neighbors       0.8333 (83.33%)
Decision Tree             0.8000 (80.00%)
============================================================

💾 Đã lưu model tốt nhất vào: models/fruit_classifier.pkl

✅ Hoàn thành quá trình huấn luyện!
```

### Bước 4: Model đã được lưu

File `models/fruit_classifier.pkl` chứa model đã huấn luyện, sẵn sàng sử dụng!

---

## 🔍 Sử dụng model để dự đoán

### Cách 1: Dòng lệnh (Command Line)

#### Cú pháp:
```bash
python predict.py <đường_dẫn_ảnh>
```

#### Ví dụ:
```bash
# Windows
python predict.py C:\Users\YourName\Pictures\apple.jpg

# Linux/Mac
python predict.py /home/username/images/apple.jpg
```

#### Kết quả:
```
============================================================
CHƯƠNG TRÌNH NHẬN DIỆN TRÁI CÂY
============================================================

📥 Đang tải model...
Đã tải model từ models/fruit_classifier.pkl
📷 Đang xử lý ảnh: apple.jpg
🔍 Đang trích xuất đặc trưng...
🤖 Đang dự đoán...

==================================================
KẾT QUẢ DỰ ĐOÁN
==================================================
🍎 Loại trái cây: Táo (Apple)

📊 Xác suất cho các loại:
   Táo (Apple)          : 92.45%
   Cam (Orange)         : 4.23%
   Nho (Grape)          : 2.11%
   Chuối (Banana)       : 0.89%
   Dưa hấu (Watermelon) : 0.32%
==================================================

Nhấn phím bất kỳ để đóng cửa sổ...
```

### Cách 2: Giao diện đồ họa (GUI)

#### Khởi chạy:
```bash
python demo_gui.py
```

#### Sử dụng GUI:

1. **Cửa sổ hiển thị:**
   ```
   ┌─────────────────────────────────────────┐
   │   🍎 NHẬN DIỆN TRÁI CÂY QUA ẢNH 🍊    │
   ├─────────────────────────────────────────┤
   │  [📁 Chọn ảnh]  [🔍 Nhận diện]        │
   ├───────────────────┬─────────────────────┤
   │                   │                     │
   │   Ảnh đầu vào    │   Kết quả nhận diện│
   │                   │                     │
   │                   │                     │
   └───────────────────┴─────────────────────┘
   ```

2. **Nhấn "Chọn ảnh":**
   - Chọn file ảnh từ máy tính
   - Ảnh sẽ hiển thị bên trái

3. **Nhấn "Nhận diện":**
   - Kết quả hiển thị bên phải
   - Xem xác suất cho từng loại trái cây

4. **Đóng ứng dụng:**
   - Nhấn nút X trên cửa sổ

---

## 🔧 Xử lý lỗi thường gặp

### Lỗi 1: "No module named 'cv2'"

**Nguyên nhân:** Chưa cài OpenCV

**Giải quyết:**
```bash
pip install opencv-python
```

### Lỗi 2: "Model không tồn tại"

**Nguyên nhân:** Chưa huấn luyện model

**Giải quyết:**
```bash
python main_train.py
```

### Lỗi 3: "Không có dữ liệu để tải"

**Nguyên nhân:** Thư mục data/train/ không có ảnh

**Giải quyết:**
1. Kiểm tra cấu trúc thư mục
2. Đảm bảo tên thư mục đúng: `táo`, `chuối`, `cam`, `nho`, `dưa hấu`
3. Đặt ít nhất 20 ảnh vào mỗi thư mục

### Lỗi 4: "Không thể đọc ảnh"

**Nguyên nhân:** File ảnh bị lỗi hoặc format không hỗ trợ

**Giải quyết:**
1. Kiểm tra file ảnh có mở được không
2. Đảm bảo format: JPG, PNG, BMP
3. Thử chuyển đổi sang format khác

### Lỗi 5: Độ chính xác thấp

**Nguyên nhân:** Dữ liệu huấn luyện không đủ tốt

**Giải quyết:**
1. Thêm nhiều ảnh hơn (tối thiểu 30-50 ảnh/loại)
2. Ảnh đa dạng (nhiều góc độ, ánh sáng)
3. Ảnh có nền đơn giản
4. Loại bỏ ảnh mờ, không rõ

---

## 💡 Tips & Tricks

### 1. Tăng độ chính xác

- ✅ Thêm nhiều ảnh huấn luyện (50-100 ảnh/loại)
- ✅ Ảnh có chất lượng tốt, rõ nét
- ✅ Nền đơn giản, không phức tạp
- ✅ Trái cây chiếm phần lớn khung hình

### 2. Tối ưu tốc độ

- ✅ Giảm số bins trong histogram (từ 32 xuống 16)
- ✅ Giảm kích thước ảnh xử lý (từ 200x200 xuống 150x150)
- ✅ Sử dụng KNN thay vì SVM nếu muốn nhanh hơn

### 3. Test model

```bash
# Test với 1 ảnh
python predict.py test_image.jpg

# Test với nhiều ảnh trong thư mục
# (Có thể viết script riêng hoặc dùng GUI)
```

### 4. Thêm loại trái cây mới

1. Mở file `utils/data_loader.py`
2. Thêm vào `FRUIT_CLASSES`:
   ```python
   FRUIT_CLASSES = {
       0: 'Táo (Apple)',
       1: 'Chuối (Banana)',
       2: 'Cam (Orange)',
       3: 'Nho (Grape)',
       4: 'Dưa hấu (Watermelon)',
       5: 'Xoài (Mango)',  # Thêm mới
   }
   ```
3. Tạo thư mục `data/train/xoài/`
4. Thêm ảnh xoài vào thư mục
5. Huấn luyện lại model

---

## 📞 Câu hỏi thường gặp (FAQ)

**Q: Cần bao nhiêu ảnh để huấn luyện?**
> A: Tối thiểu 20 ảnh/loại, khuyến nghị 30-50 ảnh/loại

**Q: Model có thể nhận diện trái cây khác không?**
> A: Không, chỉ nhận diện 5 loại đã định nghĩa

**Q: Làm sao để cải thiện độ chính xác?**
> A: Thêm nhiều ảnh đa dạng, chất lượng tốt

**Q: Có thể dùng webcam để nhận diện real-time không?**
> A: Có thể, nhưng cần code thêm. Xem phần mở rộng

**Q: Model này có thể chạy trên điện thoại không?**
> A: Cần chuyển đổi sang format khác (TFLite, ONNX)

---

## 🎯 Tổng kết

Sau khi hoàn thành hướng dẫn này, bạn đã:

✅ Cài đặt môi trường Python và OpenCV
✅ Chuẩn bị dữ liệu huấn luyện
✅ Huấn luyện model Machine Learning
✅ Sử dụng model để nhận diện trái cây
✅ Xử lý các lỗi cơ bản

**Chúc mừng! Bạn đã hoàn thành dự án nhận diện trái cây! 🎉**

---

*Tài liệu này được viết cho môn học Nhập môn Xử lý Ảnh*
