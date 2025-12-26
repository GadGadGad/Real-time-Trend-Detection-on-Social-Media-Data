import json
import csv
import time
import os
import sys
import glob
import random
from datetime import datetime
from kafka import KafkaProducer

# Cấu hình Kafka
KAFKA_TOPIC = 'raw_data'
BOOTSTRAP_SERVERS = ['localhost:9092']

def create_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        print(f"✅ Đã kết nối tới Kafka tại {BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        print(f"❌ Lỗi kết nối Kafka: {e}")
        return None

# --- Logic Load dữ liệu (Mượn từ utils/data_loader.py) ---
def clean_text(text):
    import re
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_json_file(filepath):
    """Đọc file Social JSON (Facebook)"""
    posts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item.get('text') or item.get('content') or ''
                # Lấy thời gian tốt nhất có thể
                time_str = item.get('time') or item.get('time_label') or datetime.now().isoformat()
                
                clean = clean_text(text)
                if len(clean) < 20: continue 
                
                posts.append({
                    "source": f"Face: {item.get('pageName', 'Unknown')}",
                    "content": clean,
                    "url": item.get('postUrl', ''),
                    "published_at": time_str,
                    "type": "social",
                    "stats": item.get('stats', {})
                })
    except Exception as e:
        print(f"⚠️ Lỗi đọc JSON {filepath}: {e}")
    return posts

def load_csv_file(filepath):
    """Đọc file News CSV (VnExpress, v.v.)"""
    posts = []
    source_name = os.path.basename(os.path.dirname(filepath)).upper()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                content = f"{row.get('title', '')}\n{row.get('content', '')}"
                if len(content) < 20: continue
                
                posts.append({
                    "source": source_name,
                    "content": clean_text(content),
                    "url": row.get('url', ''),
                    "published_at": row.get('published_at', datetime.now().isoformat()),
                    "type": "news",
                    "stats": {} # News thường không có like/share trong file csv crawler
                })
    except Exception as e:
        print(f"⚠️ Lỗi đọc CSV {filepath}: {e}")
    return posts

# --- Logic Replay ---
def run_replay(data_dir, speed=1.0):
    """
    data_dir: Thư mục chứa file dữ liệu (data/raw_demo)
    speed: Tốc độ replay (ví dụ 0.1 là nhanh gấp 10 lần, 1.0 là 1 giây = 1 giây)
           Nhưng để demo nhanh, ta thường chỉ sleep 1 khoảng random nhỏ.
    """
    producer = create_producer()
    if not producer: return

    all_posts = []
    
    # 1. Quét tất cả file trong thư mục data_dir
    print(f"📂 Đang quét dữ liệu từ: {data_dir}...")
    
    # Tìm JSON (Social)
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    for f in json_files:
        all_posts.extend(load_json_file(f))
        
    # Tìm CSV (News)
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    for f in csv_files:
        all_posts.extend(load_csv_file(f))
        
    print(f"📊 Tổng cộng: {len(all_posts)} bài viết.")
    
    # 2. Xáo trộn ngẫu nhiên để mô phỏng dữ liệu đến từ nhiều nguồn cùng lúc
    # (Trong thực tế nên sort theo time, nhưng format time mỗi nguồn khác nhau khá phức tạp để parse chuẩn)
    random.shuffle(all_posts)
    
    print("🚀 Bắt đầu Replay vào Kafka topic 'raw_data'...")
    
    try:
        for i, post in enumerate(all_posts):
            # Gửi vào Kafka
            producer.send(KAFKA_TOPIC, value=post)
            
            # Log tiến độ
            if (i+1) % 10 == 0:
                sys.stdout.write(f"\r📤 Đã gửi: {i+1}/{len(all_posts)} messages...")
                sys.stdout.flush()
            
            # Giả lập độ trễ (Streaming delay)
            # Sleep random từ 0.05s đến 0.2s để tạo cảm giác data đang trôi về
            time.sleep(random.uniform(0.05, 0.2) * speed)
            
        producer.flush()
        print(f"\n✅ Hoàn tất replay {len(all_posts)} messages.")
        
    except KeyboardInterrupt:
        print("\n🛑 Dừng replay.")
    finally:
        producer.close()

if __name__ == "__main__":
    # Đường dẫn mặc định đến thư mục data demo
    # Giả sử chạy từ root project
    DEFAULT_DATA_DIR = "data/raw_demo"
    
    if len(sys.argv) > 1:
        DEFAULT_DATA_DIR = sys.argv[1]
        
    if not os.path.exists(DEFAULT_DATA_DIR):
        print(f"❌ Thư mục không tồn tại: {DEFAULT_DATA_DIR}")
        print("👉 Hãy tạo thư mục 'data/raw_demo' và copy file dữ liệu cũ vào đó.")
    else:
        run_replay(DEFAULT_DATA_DIR)