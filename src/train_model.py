"""
Module huấn luyện và lưu model
Sử dụng các thuật toán Machine Learning đơn giản
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class FruitClassifier:
    """
    Class để huấn luyện và sử dụng model phân loại trái cây
    """
    
    def __init__(self, model_type='svm'):
        """
        Khởi tạo classifier
        
        Args:
            model_type (str): Loại model ('svm', 'knn', 'dt', 'rf')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.classes = None
        
        # Chọn model dựa trên loại
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        elif model_type == 'dt':
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Model type '{model_type}' không được hỗ trợ")
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện model
        
        Args:
            X_train (numpy.ndarray): Dữ liệu huấn luyện
            y_train (numpy.ndarray): Nhãn huấn luyện
            X_val (numpy.ndarray): Dữ liệu validation (optional)
            y_val (numpy.ndarray): Nhãn validation (optional)
        
        Returns:
            dict: Kết quả huấn luyện
        """
        print(f"\n=== Huấn luyện model {self.model_type.upper()} ===")
        print(f"Số mẫu huấn luyện: {len(X_train)}")
        
        # Chuẩn hóa dữ liệu
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Huấn luyện model
        self.model.fit(X_train_scaled, y_train)
        self.classes = np.unique(y_train)
        
        # Đánh giá trên tập huấn luyện
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Độ chính xác trên tập huấn luyện: {train_acc:.4f}")
        
        results = {'train_accuracy': train_acc}
        
        # Đánh giá trên tập validation nếu có
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Độ chính xác trên tập validation: {val_acc:.4f}")
            results['val_accuracy'] = val_acc
            
            # In báo cáo chi tiết
            print("\nBáo cáo phân loại:")
            print(classification_report(y_val, val_pred))
        
        return results
    
    
    def predict(self, features):
        """
        Dự đoán nhãn cho một mẫu
        
        Args:
            features (numpy.ndarray): Vector đặc trưng
        
        Returns:
            tuple: (predicted_label, confidence)
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện!")
        
        # Chuẩn hóa dữ liệu
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Dự đoán
        prediction = self.model.predict(features_scaled)[0]
        
        # Lấy xác suất (confidence)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return prediction, confidence
    
    
    def predict_with_probabilities(self, features):
        """
        Dự đoán với xác suất cho tất cả các lớp
        
        Args:
            features (numpy.ndarray): Vector đặc trưng
        
        Returns:
            dict: Dictionary chứa nhãn và xác suất cho mỗi lớp
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện!")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        result = {'predicted_class': int(prediction)}
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            result['probabilities'] = {int(cls): float(prob) 
                                      for cls, prob in zip(self.classes, probabilities)}
        
        return result
    
    
    def save_model(self, filepath):
        """
        Lưu model vào file
        
        Args:
            filepath (str): Đường dẫn file để lưu
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'classes': self.classes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Đã lưu model vào {filepath}")
    
    
    def load_model(self, filepath):
        """
        Tải model từ file
        
        Args:
            filepath (str): Đường dẫn file model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.classes = model_data['classes']
        
        print(f"Đã tải model từ {filepath}")


def compare_models(X_train, y_train, X_test, y_test):
    """
    So sánh hiệu suất của các model khác nhau
    
    Args:
        X_train, y_train: Dữ liệu huấn luyện
        X_test, y_test: Dữ liệu test
    
    Returns:
        dict: Kết quả so sánh các model
    """
    models = {
        'SVM': 'svm',
        'K-Nearest Neighbors': 'knn',
        'Decision Tree': 'dt',
        'Random Forest': 'rf'
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("SO SÁNH CÁC MODEL")
    print("="*50)
    
    for name, model_type in models.items():
        print(f"\n>>> Đang huấn luyện {name}...")
        
        try:
            classifier = FruitClassifier(model_type=model_type)
            classifier.train(X_train, y_train)
            
            # Đánh giá trên tập test
            X_test_scaled = classifier.scaler.transform(X_test)
            y_pred = classifier.model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': test_acc,
                'model': classifier
            }
            
            print(f"Độ chính xác trên tập test: {test_acc:.4f}")
            
        except Exception as e:
            print(f"Lỗi khi huấn luyện {name}: {str(e)}")
            results[name] = {'accuracy': 0.0, 'model': None}
    
    # Tìm model tốt nhất
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n{'='*50}")
    print(f"Model tốt nhất: {best_model_name} với độ chính xác {results[best_model_name]['accuracy']:.4f}")
    print(f"{'='*50}\n")
    
    return results
