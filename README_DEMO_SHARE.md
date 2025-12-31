# 🚀 Hướng Dẫn Chạy Hệ Thống Phát Hiện Xu Hướng (Event Detection)

## 1. Chuẩn Bị Hệ Thống
*   **Hệ điều hành:** Khuyên dùng Linux hoặc macOS (Để chạy tốt Bash script).
*   **Docker & Docker Compose:** Để quản lý các dịch vụ hạ tầng (Kafka, Postgres, Spark).
*   **Python 3.12:** Khuyên dùng Conda hoặc virtual environment để tránh xung đột thư viện.
*   **RAM:** Tối thiểu 8GB (Hệ thống đã được tối ưu chạy trong "Lite Mode" chỉ tốn khoảng 1-2GB RAM cho các dịch vụ Java).

## 2. Cài Đặt Môi Trường
Mở Terminal tại thư mục gốc của dự án và chạy các lệnh sau:

```bash
# 1. Cài đặt các thư viện Python chuyên sâu
pip install -r requirements.txt

# 2. Cài đặt trình duyệt cho module cào dữ liệu (nếu cần chạy crawl lại)
playwright install chromium
```

## 3. Cấu Hình AI (Quan trọng)
1.  Tìm file `.env` ở thư mục gốc.
2.  Mở file và cập nhật `GEMINI_API_KEY` của bạn:
    ```env
    GEMINI_API_KEY=AIzaSy... (Điền key của bạn ở đây)
    ```
    *Nếu không có Key, AI sẽ không thể đặt tên Trend hoặc tóm tắt nội dung được.*

## 4. Kiểm Tra Dữ Liệu
Hãy đảm bảo folder bạn nhận được có đủ 2 file/thư mục sau để demo chạy được ngay:
*   `streaming/embeddings_cache.pkl`: Chứa 4,700 bài đăng đã được AI vector hóa sẵn.
*   `data/demo-ready_archieve/`: Thư mục chứa các file dữ liệu CSV gốc.

## 5. Bắt Đầu Demo 🎬
Bạn chỉ cần chạy **duy nhất một lệnh** để khởi động toàn bộ "vũ trụ" của dự án:

```bash
chmod +x run_full_system.sh
./run_full_system.sh
```

### Script này sẽ tự động:
- Bật Docker (Kafka, Postgres, Zookeeper, Spark).
- Khởi tạo Database và nạp 84 trend mẫu.
- Bật Dashboard (Streamlit).
- Bật Spark Streaming với cơ chế **Throttling** (Xử lý 500 bài/lần để không treo máy).
- Bật Producer chạy **vòng lặp vô tận** (Gửi hết 4.7k tin sẽ tự động trộn và gửi lại).

## 6. Xem Kết Quả
*   📊 **Dashboard:** [http://localhost:8501](http://localhost:8501) (Giao diện chính để xem trend).
*   🌪️ **Airflow:** [http://localhost:8080](http://localhost:8080) (Tài khoản: `admin` / `admin`).

## 7. Xử Lý Sự Cố (Troubleshooting)
*   **Lỗi "Connection Refused":** Thường do Docker khởi động chậm. Hãy đợi 30s-1 phút rồi chạy lại lệnh.
*   **Máy bị lag:** Hệ thống đã giới hạn Spark dùng 1GB RAM. Nếu vẫn lag, hãy đóng các trình duyệt không cần thiết.
*   **Dữ liệu không nhảy:** Kiểm tra file `consumer.log` hoặc `producer.log` để xem lỗi kết nối Kafka.

## 8. Reset Hệ Thống (Làm mới từ đầu)

Nếu bạn muốn xóa sạch mọi dữ liệu đã chạy và đưa hệ thống về trạng thái "vừa mới cài đặt" (ví dụ: trước khi bắt đầu bài thuyết trình thật), hãy chạy:

```bash
chmod +x reset_demo.sh
./reset_demo.sh
```

Script này sẽ xóa sạch Database, logs, và lịch sử chạy của Airflow để bạn có một khởi đầu hoàn hảo nhất. ✅

## 9. Dừng Hệ Thống
Chỉ cần nhấn `Ctrl + C` tại Terminal đang chạy script `run_full_system.sh`.
