"""
Demo ứng dụng với giao diện đơn giản
Cho phép chọn ảnh và nhận diện trái cây
"""

import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processing import load_image, extract_all_features, preprocess_image
from utils.data_loader import FRUIT_CLASSES
from src.train_model import FruitClassifier


class FruitRecognizerApp:
    """
    Ứng dụng GUI để nhận diện trái cây
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Trái Cây - Fruit Recognition")
        self.root.geometry("800x600")
        
        # Biến lưu trạng thái
        self.current_image = None
        self.classifier = None
        self.model_path = 'models/fruit_classifier.pkl'
        
        # Tạo giao diện
        self.create_widgets()
        
        # Tải model
        self.load_model()
    
    
    def create_widgets(self):
        """
        Tạo các thành phần giao diện
        """
        # Frame chính
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tiêu đề
        title_label = tk.Label(
            main_frame, 
            text="🍎 NHẬN DIỆN TRÁI CÂY QUA ẢNH 🍊",
            font=("Arial", 18, "bold"),
            fg="darkgreen"
        )
        title_label.pack(pady=10)
        
        # Frame cho các nút
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Nút chọn ảnh
        self.btn_select = tk.Button(
            button_frame,
            text="📁 Chọn ảnh",
            command=self.select_image,
            font=("Arial", 12),
            bg="lightblue",
            width=15,
            height=2
        )
        self.btn_select.pack(side=tk.LEFT, padx=5)
        
        # Nút nhận diện
        self.btn_predict = tk.Button(
            button_frame,
            text="🔍 Nhận diện",
            command=self.predict_image,
            font=("Arial", 12),
            bg="lightgreen",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        
        # Frame hiển thị ảnh và kết quả
        display_frame = tk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Frame hiển thị ảnh
        image_frame = tk.LabelFrame(display_frame, text="Ảnh đầu vào", font=("Arial", 10, "bold"))
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_label = tk.Label(image_frame, text="Chưa có ảnh", bg="lightgray")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame hiển thị kết quả
        result_frame = tk.LabelFrame(display_frame, text="Kết quả nhận diện", font=("Arial", 10, "bold"))
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_text = tk.Text(
            result_frame,
            font=("Arial", 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="lightyellow"
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Thông tin phía dưới
        info_label = tk.Label(
            main_frame,
            text="Sử dụng OpenCV và Machine Learning để nhận diện trái cây",
            font=("Arial", 9),
            fg="gray"
        )
        info_label.pack(pady=5)
    
    
    def load_model(self):
        """
        Tải model đã huấn luyện
        """
        try:
            if os.path.exists(self.model_path):
                self.classifier = FruitClassifier()
                self.classifier.load_model(self.model_path)
                self.update_result("✅ Model đã sẵn sàng!\n\nVui lòng chọn ảnh để bắt đầu.")
            else:
                self.update_result(
                    "⚠️ Chưa có model!\n\n"
                    "Vui lòng chạy main_train.py\n"
                    "để huấn luyện model trước."
                )
                self.btn_predict.config(state=tk.DISABLED)
        except Exception as e:
            self.update_result(f"❌ Lỗi khi tải model:\n{str(e)}")
    
    
    def select_image(self):
        """
        Chọn ảnh từ file
        """
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh trái cây",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Đọc và hiển thị ảnh
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Không thể đọc ảnh")
                
                self.display_image(self.current_image)
                self.btn_predict.config(state=tk.NORMAL)
                self.update_result("✅ Đã tải ảnh thành công!\n\nNhấn 'Nhận diện' để bắt đầu.")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc ảnh:\n{str(e)}")
    
    
    def display_image(self, cv_image):
        """
        Hiển thị ảnh OpenCV trên tkinter
        """
        # Chuyển đổi BGR sang RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize để vừa với khung hiển thị
        height, width = rgb_image.shape[:2]
        max_height = 400
        max_width = 350
        
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(rgb_image, (new_width, new_height))
        
        # Chuyển sang PIL Image
        pil_image = Image.fromarray(resized)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Hiển thị
        self.image_label.config(image=tk_image, text="")
        self.image_label.image = tk_image  # Giữ reference
    
    
    def predict_image(self):
        """
        Nhận diện trái cây từ ảnh
        """
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        if self.classifier is None:
            messagebox.showerror("Lỗi", "Model chưa được tải!")
            return
        
        try:
            # Trích xuất đặc trưng
            features = extract_all_features(self.current_image)
            
            # Dự đoán
            result = self.classifier.predict_with_probabilities(features)
            
            predicted_class = result['predicted_class']
            fruit_name = FRUIT_CLASSES[predicted_class]
            
            # Hiển thị kết quả
            result_text = f"🎯 KẾT QUẢ NHẬN DIỆN\n\n"
            result_text += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            result_text += f"🍎 Loại trái cây:\n"
            result_text += f"   {fruit_name}\n\n"
            
            if 'probabilities' in result:
                result_text += f"📊 Xác suất:\n\n"
                probs = result['probabilities']
                for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(prob * 20)
                    result_text += f"{FRUIT_CLASSES[cls][:15]:<15}\n"
                    result_text += f"{bar} {prob:.1%}\n\n"
            
            self.update_result(result_text)
            
            # Vẽ kết quả lên ảnh
            display_img = self.current_image.copy()
            cv2.putText(
                display_img,
                fruit_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            self.display_image(display_img)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi nhận diện:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    
    def update_result(self, text):
        """
        Cập nhật text kết quả
        """
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self.result_text.config(state=tk.DISABLED)


def main():
    """
    Hàm chính để chạy ứng dụng
    """
    root = tk.Tk()
    app = FruitRecognizerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
