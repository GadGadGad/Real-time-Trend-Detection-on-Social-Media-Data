# Hướng dẫn sử dụng Trained Classifiers

## Tổng quan

Sau khi train các classifier trên Kaggle, pipeline sẽ **TỰ ĐỘNG** phát hiện và sử dụng chúng. Không cần sửa đổi code trong notebook.

---

## Bước 1: Train Models trên Kaggle

### Taxonomy Classifier (T1-T7)
1. Mở `notebooks/train-taxonomy-kaggle.ipynb` trên Kaggle
2. Chạy tất cả cells
3. Download `taxonomy_classifier.zip` từ Output

### Sentiment Classifier (Positive/Negative/Neutral)  
1. Mở `notebooks/train-sentiment-kaggle.ipynb` trên Kaggle
2. Chạy tất cả cells
3. Download `sentiment_classifier.zip` từ Output

---

## Bước 2: Đặt Models vào đúng vị trí

```bash
# Tạo thư mục models nếu chưa có
mkdir -p models

# Giải nén taxonomy classifier
unzip taxonomy_classifier.zip -d models/
# Kết quả: models/taxonomy-classifier-vietnamese-v1/

# Giải nén sentiment classifier
unzip sentiment_classifier.zip -d models/
# Kết quả: models/sentiment-classifier-vietnamese-v1/
```

### Cấu trúc thư mục sau khi giải nén:
```
models/
├── taxonomy-classifier-vietnamese-v1/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── sentiment-classifier-vietnamese-v1/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## Bước 3: Chạy Notebook

Khi chạy `analysis-playground-v3.ipynb`, bạn sẽ thấy:

### Nếu models được phát hiện:
```
✅ Loading TRAINED Taxonomy Classifier from models/taxonomy-classifier-vietnamese-v1...
✅ Loading TRAINED Sentiment Classifier from models/sentiment-classifier-vietnamese-v1...
```

### Nếu models không có (fallback):
```
🧠 Pre-computing 7-Group Taxonomy Embeddings (fallback)...
🧠 Loading Default PhoBERT Sentiment Model (wonrax/phobert-base-vietnamese-sentiment)...
```

---

## Sử dụng trên Kaggle

Nếu bạn chạy notebook trên Kaggle và đã upload models như dataset, đảm bảo đặt đúng path:

```python
# Kaggle sẽ tự động phát hiện nếu bạn đặt models tại:
# /kaggle/input/taxonomy-classifier/taxonomy-classifier-vietnamese-v1
# /kaggle/input/sentiment-classifier/sentiment-classifier-vietnamese-v1
```

---

## Kiểm tra xem đang dùng model nào

```python
from src.core.analysis.sentiment import is_using_custom_model
from src.core.extraction.taxonomy_classifier import TaxonomyClassifier

# Kiểm tra sentiment
print(f"Using custom sentiment: {is_using_custom_model()}")

# Kiểm tra taxonomy
clf = TaxonomyClassifier()
print(f"Using transformer taxonomy: {clf.is_using_transformer()}")
```

---

## Fallback Behavior

| Model không có | Pipeline sử dụng |
|----------------|------------------|
| Taxonomy | Keyword-based + Semantic similarity |
| Sentiment | `wonrax/phobert-base-vietnamese-sentiment` |

Cả hai fallback đều hoạt động tốt, chỉ là **trained models có accuracy cao hơn**.
