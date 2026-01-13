# Hướng Dẫn Chạy Hệ Thống Phát Hiện Xu Hướng (Event Detection)

## 1. Chuẩn Bị Hệ Thống
*   **Hệ điều hành:** Khuyên dùng Linux hoặc macOS (Để chạy tốt Bash script).
*   **Docker & Docker Compose:** Để quản lý các dịch vụ hạ tầng (Kafka, Postgres, Spark).
*   **Python 3.12:** Khuyên dùng Conda hoặc virtual environment để tránh xung đột thư viện.
*   **RAM:** Tối thiểu 8GB (Hệ thống đã được tối ưu chạy trong "Lite Mode" chỉ tốn khoảng 1-2GB RAM cho các dịch vụ Java).

## 2. Cài Đặt Môi Trường
Mở Terminal tại thư mục gốc của dự án và chạy các lệnh sau:

```bash
# 1. Cài đặt các thư viện Python chuyên sâu
# Cách 1: Sử dụng PIP (Truyền thống)
pip install -r requirements.txt

# Cách 2: Sử dụng Conda (Khuyên dùng để tránh lỗi thư viện)
conda env create -f environment.yaml
conda activate se363-trend-detection

# 2. Cài đặt trình duyệt cho module cào dữ liệu (nếu cần chạy crawl lại)
playwright install chromium
```

## 3. Cài Đặt Dữ Liệu & Cấu Hình Nhanh
Nếu ông nhận được file **`demo_essentials.zip`** hoặc tải ở [đây](https://drive.google.com/file/d/1iIzBh21gnuyjqh3pJye-1KOFpGWHo3JS/view?usp=sharing), hãy làm theo các bước sau để thiết lập nhanh:

1.  **Giải nén file:** Đặt file zip vào thư mục gốc của dự án (cùng cấp với file này).
    ```bash
    unzip demo_essentials.zip
    ```
2.  **Kiểm tra file `.env`:** Sau khi giải nén, file `.env` sẽ xuất hiện. Mở nó và chỉnh sửa `GEMINI_API_KEY` nếu cần thiết (hoặc sử dụng key có sẵn nếu được cung cấp).

*Lưu ý: Hệ thống cần file `streaming/embeddings_cache.pkl` (Model đã vector hóa data). File này thường có sẵn khi clone git, nhưng nếu thiếu, hãy đảm bảo ông đã tải nó về.*

## 3b. Kiểm Tra Thủ Công (Nếu không dùng zip)
*   Tạo file `.env` và điền `GEMINI_API_KEY`.
*   Đảm bảo có thư mục `data/demo-ready_archieve/` chứa các file CSV.

## 4. Bắt Đầu Demo
Ông chỉ cần chạy **duy nhất một lệnh** để khởi động toàn bộ hệ thống "Pro" (Next.js + Airflow + FastAPI):

```bash
chmod +x run_pro_system.sh
./run_pro_system.sh demo
```

### Script này sẽ tự động:
- Bật Docker (Kafka, Postgres, Zookeeper, Spark).
- Khởi tạo Database và nạp 84 trend mẫu.
- Bật Dashboard (Streamlit).
- Bật Spark Streaming với cơ chế **Throttling** (Xử lý 500 bài/lần để không treo máy).
- Bật Producer chạy **vòng lặp vô tận** (Gửi hết 4.7k tin sẽ tự động trộn và gửi lại).

## 6. Xem Kết Quả
*   **Dashboard:** [http://localhost:8501](http://localhost:8501) (Giao diện chính để xem trend).
*   **Airflow:** [http://localhost:8080](http://localhost:8080) (Tài khoản: `admin` / `admin`).

## 7. Xử Lý Sự Cố (Troubleshooting)
*   **Lỗi "Connection Refused":** Thường do Docker khởi động chậm. Hãy đợi 30s-1 phút rồi chạy lại lệnh.
*   **Máy bị lag:** Hệ thống đã giới hạn Spark dùng 1GB RAM. Nếu vẫn lag, hãy đóng các trình duyệt không cần thiết.
*   **Dữ liệu không nhảy:** Kiểm tra file `consumer.log` hoặc `producer.log` để xem lỗi kết nối Kafka.

## 8. Reset Hệ Thống (Làm mới từ đầu)

Nếu ông muốn xóa sạch mọi dữ liệu đã chạy và đưa hệ thống về trạng thái "vừa mới cài đặt" (ví dụ: trước khi bắt đầu bài thuyết trình thật), hãy chạy:

```bash
chmod +x reset_demo.sh
./reset_demo.sh
```

## 9. Dừng Hệ Thống
Chỉ cần nhấn `Ctrl + C` tại Terminal đang chạy script `run_full_system.sh`.
