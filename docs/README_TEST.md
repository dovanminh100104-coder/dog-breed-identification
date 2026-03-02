# Hướng dẫn Test Model Dog Breed Classifier

## 📋 Yêu cầu
- Model đã được train (`final_dog_breed_model.h5`)
- File `labels.csv` và thư mục `train/` tồn tại
- Các thư viện: tensorflow, opencv, pandas, matplotlib

## 🚀 Cách Test

### 1. Chạy Test Interactively
```bash
python test_model.py
```
Chọn option:
- **1**: Test với ảnh mẫu trong thư mục test/
- **2**: Test với ảnh riêng (nhập đường dẫn)
- **3**: Thoát

### 2. Test với Ảnh Cụ Thể
```python
from test_model import test_custom_image

# Test một ảnh cụ thể
test_custom_image("path/to/your/image.jpg")
```

### 3. Batch Test toàn bộ thư mục test
```python
from test_model import batch_test

# Test tất cả ảnh trong thư mục test
batch_test('test')
```

## 📊 Kết quả Test

### Output mẫu:
```
Image 1: 0dc570ec7086bab004a7e357164c04b8.jpg
  1. golden_retriever: 89.23%
  2. labrador_retriever: 6.45%
  3. flat_coated_retriever: 2.12%

=== Test Statistics ===
Total images tested: 10
Average confidence: 76.34%
High confidence (>80%): 6 (60.0%)
Medium confidence (60-80%): 3 (30.0%)
Low confidence (<60%): 1 (10.0%)
```

## 🔍 Các Chế độ Test

### 1. **Single Image Test**
- Hiển thị ảnh gốc
- Top 3 dự đoán với confidence score
- Phù hợp để kiểm tra nhanh

### 2. **Batch Test**
- Test toàn bộ thư mục
- Thống kê confidence scores
- Phù hợp để đánh giá model tổng thể

### 3. **Interactive Mode**
- Menu lựa chọn dễ sử dụng
- Test nhiều ảnh khác nhau
- Hiển thị kết quả trực quan

## 📈 Đánh giá Kết quả

### **Confidence Levels:**
- **>80%**: Rất tốt - Model tự tin cao
- **60-80%**: Tốt - Dự đoán hợp lý
- **<60%**: Thấp - Cần kiểm tra lại ảnh

### **Common Issues:**
1. **Ảnh không rõ**: Giảm confidence
2. **Chó không trong 120 breed**: Model dự đoán sai
3. **Nhiều chó trong ảnh**: Model có thể confused

## 🛠 Tùy chỉnh Test

### Thay đổi số lượng top predictions:
```python
# Trong predict_single_image()
results, original_img = predict_single_image(image_path, model, class_indices, top_k=5)
```

### Thay đổi threshold:
```python
# Trong batch_test()
high_confidence = sum(1 for c in confidence_scores if c > 90)  # Thay đổi threshold
```

## 📝 Tips Test tốt nhất

1. **Sử dụng ảnh chất lượng cao** (không mờ, đủ sáng)
2. **Chó chiếm phần lớn ảnh** (tránh ảnh chó nhỏ)
3. **Test với nhiều breed khác nhau**
4. **So sánh với kết quả thực tế** để biết độ chính xác

## 🎯 Kết quả mong đợi

Model này đạt **~76% accuracy** trên validation set, vì vậy:
- ~76% ảnh sẽ có prediction đúng
- Confidence trung bình ~75%
- Top-1 accuracy cao hơn Top-3 accuracy

Chúc bạn test thành công! 🐕
