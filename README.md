# 🍎 Dự Án Nhận Diện Trái Cây Qua Ảnh

## 📋 Giới thiệu

Đây là dự án nhập môn xử lý ảnh sử dụng **OpenCV** để nhận diện các loại trái cây thông qua ảnh. Dự án này phù hợp cho người mới học, sử dụng các kỹ thuật xử lý ảnh cơ bản và Machine Learning đơn giản.

### Các loại trái cây được nhận diện:
- 🍎 Táo (Apple)
- 🍌 Chuối (Banana)
- 🍊 Cam (Orange)
- 🍇 Nho (Grape)
- 🍉 Dưa hấu (Watermelon)

## 🎯 Mục tiêu học tập

- Hiểu cách xử lý và tiền xử lý ảnh với OpenCV
- Học cách trích xuất đặc trưng từ ảnh (màu sắc, hình dạng, texture)
- Áp dụng Machine Learning để phân loại ảnh
- Xây dựng ứng dụng demo đơn giản

## 📁 Cấu trúc dự án

```
Code/
├── data/                          # Thư mục chứa dữ liệu
│   ├── train/                     # Dữ liệu huấn luyện
│   │   ├── táo/                   # Ảnh táo
│   │   ├── chuối/                 # Ảnh chuối
│   │   ├── cam/                   # Ảnh cam
│   │   ├── nho/                   # Ảnh nho
│   │   └── dưa hấu/               # Ảnh dưa hấu
│   └── test/                      # Dữ liệu test
│       ├── táo/
│       ├── chuối/
│       ├── cam/
│       ├── nho/
│       └── dưa hấu/
│
├── models/                        # Thư mục lưu model đã huấn luyện
│   └── fruit_classifier.pkl       # Model được lưu
│
├── utils/                         # Thư mục chứa các hàm tiện ích
│   ├── image_processing.py        # Xử lý ảnh và trích xuất đặc trưng
│   └── data_loader.py             # Tải và quản lý dữ liệu
│
├── src/                           # Thư mục chứa code chính
│   └── train_model.py             # Huấn luyện model
│
├── main_train.py                  # Script huấn luyện model
├── predict.py                     # Script dự đoán đơn lẻ
├── demo_gui.py                    # Ứng dụng GUI
├── requirements.txt               # Các thư viện cần thiết
│
└── docs/                          # Tài liệu
    ├── README.md                  # File này
    ├── HUONG_DAN_SU_DUNG.md      # Hướng dẫn sử dụng
    └── GIAI_THICH_CODE.md        # Giải thích chi tiết code
```

## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.7 trở lên
- Pip (Python package manager)

### 2. Cài đặt thư viện

```bash
# Di chuyển vào thư mục dự án
cd Code

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3. Các thư viện sử dụng
- **opencv-python**: Xử lý ảnh
- **numpy**: Tính toán số học
- **scikit-learn**: Machine Learning
- **pillow**: Xử lý ảnh cho GUI
- **matplotlib**: Vẽ đồ thị (optional)

## 🚀 Cách sử dụng

### Bước 1: Chuẩn bị dữ liệu

1. Tạo cấu trúc thư mục trong `data/train/`:
   - `táo/` - Đặt ảnh táo vào đây
   - `chuối/` - Đặt ảnh chuối vào đây
   - `cam/` - Đặt ảnh cam vào đây
   - `nho/` - Đặt ảnh nho vào đây
   - `dưa hấu/` - Đặt ảnh dưa hấu vào đây

2. Mỗi thư mục nên có ít nhất **20-30 ảnh** để model học tốt

3. Ảnh nên:
   - Rõ nét, không bị mờ
   - Có nền đơn giản
   - Kích thước hợp lý (không quá nhỏ)

### Bước 2: Huấn luyện model

```bash
python main_train.py
```

Chương trình sẽ:
- Tải tất cả ảnh từ thư mục `data/train/`
- Trích xuất đặc trưng từ ảnh
- Huấn luyện và so sánh 4 thuật toán ML
- Lưu model tốt nhất vào `models/fruit_classifier.pkl`

### Bước 3: Dự đoán trái cây

#### Cách 1: Dòng lệnh
```bash
python predict.py path/to/image.jpg
```

#### Cách 2: Giao diện GUI
```bash
python demo_gui.py
```

Sau đó:
1. Nhấn "Chọn ảnh" để chọn file ảnh
2. Nhấn "Nhận diện" để xem kết quả

## 📊 Kỹ thuật sử dụng

### 1. Tiền xử lý ảnh
- **Resize**: Chuẩn hóa kích thước ảnh về 200x200
- **Gaussian Blur**: Khử nhiễu
- **Threshold**: Tạo ảnh nhị phân để phân tích hình dạng

### 2. Trích xuất đặc trưng

#### a) Đặc trưng màu sắc (Color Histogram)
- Chuyển ảnh sang không gian màu HSV
- Tính histogram cho mỗi kênh (H, S, V)
- Chuẩn hóa histogram

#### b) Đặc trưng hình dạng (Shape Features)
- Diện tích (Area)
- Chu vi (Perimeter)
- Độ tròn (Circularity)
- Tỷ lệ khung hình (Aspect Ratio)
- Hu Moments (bất biến với phép quay, scale)

#### c) Đặc trưng texture (Texture Features)
- Sobel edges (gradient theo X và Y)
- Laplacian (đạo hàm bậc 2)
- Thống kê: mean, std, max

### 3. Thuật toán Machine Learning

Dự án so sánh 4 thuật toán:

1. **SVM (Support Vector Machine)**
   - Tìm siêu phẳng tối ưu phân tách các lớp
   - Kernel RBF để xử lý dữ liệu phi tuyến

2. **K-Nearest Neighbors (KNN)**
   - Phân loại dựa trên K láng giềng gần nhất
   - Đơn giản, dễ hiểu

3. **Decision Tree**
   - Cây quyết định dựa trên các ngưỡng đặc trưng
   - Dễ giải thích

4. **Random Forest**
   - Tập hợp nhiều Decision Trees
   - Thường cho kết quả tốt và ổn định

## 📈 Kết quả mong đợi

- Độ chính xác: **70-90%** (tùy thuộc vào chất lượng dữ liệu)
- Thời gian huấn luyện: 1-5 phút
- Thời gian dự đoán: < 1 giây/ảnh

## ⚠️ Lưu ý

1. **Chất lượng dữ liệu** rất quan trọng:
   - Càng nhiều ảnh càng tốt
   - Ảnh đa dạng (góc nhìn, ánh sáng khác nhau)
   - Nền đơn giản giúp model học tốt hơn

2. **Giới hạn**:
   - Chỉ nhận diện được 5 loại trái cây đã định nghĩa
   - Độ chính xác phụ thuộc vào điều kiện chụp ảnh
   - Nên test với ảnh có nền tương tự ảnh huấn luyện

3. **Cải thiện**:
   - Thêm nhiều ảnh huấn luyện
   - Thử các tham số khác nhau cho thuật toán
   - Thêm nhiều loại đặc trưng
   - Sử dụng Deep Learning (CNN) cho kết quả tốt hơn

## 📚 Tài liệu tham khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Image Processing với Python](https://realpython.com/image-processing-with-the-python-pillow-library/)

## 🤝 Đóng góp

Đây là dự án học tập, bạn có thể:
- Thêm nhiều loại trái cây
- Cải thiện thuật toán
- Thêm tính năng mới
- Tối ưu hóa code

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Đã cài đặt đầy đủ thư viện chưa
2. Cấu trúc thư mục dữ liệu đúng chưa
3. Có đủ ảnh huấn luyện chưa
4. Model đã được huấn luyện chưa

## 📝 License

Dự án này dành cho mục đích học tập.

---

**Chúc bạn học tập thành công! 🎓**
