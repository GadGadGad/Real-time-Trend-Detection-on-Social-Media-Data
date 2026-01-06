import pandas as pd
import re
import argparse
import sys
import os

def process_cleaning(input_file, output_file):
    """
    Hàm đọc file, xử lý dữ liệu và lưu ra file mới.
    """
    # Kiểm tra file đầu vào có tồn tại không
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_file}'")
        return

    print(f"Đang đọc file: {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    # Logic xử lý từng dòng
    def clean_row(row):
        # Chuyển về string để tránh lỗi nếu ô trống
        raw_content = str(row.get('content', ''))
        title = str(row.get('title', ''))
        
        # Regex tìm ngày giờ
        date_pattern = r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s+\(GMT\+07:00\))"
        match = re.search(date_pattern, raw_content)
        
        if match:
            found_date = match.group(1)
            
            # Cập nhật Published At
            if pd.isna(row.get('published_at')) or row['published_at'] == '':
                row['published_at'] = found_date
            
            # Tách nội dung làm 2 phần: Trước ngày và Sau ngày
            parts = raw_content.split(found_date)
            pre_date = parts[0] # Chứa Tác giả, Tiêu đề lặp, Rác
            post_date = parts[1] # Chứa Sapo, Content
            
            # --- XỬ LÝ TÁC GIẢ (Phần trước ngày) ---
            if pd.isna(row.get('author')) or row['author'] == '':
                lines = [line.strip() for line in pre_date.split('\n') if line.strip()]
                ignore_list = ["Xem các bài viết", "Sao chép liên kết", "Sự kiện:", title]
                
                for line in lines:
                    # Nếu dòng không chứa từ khóa rác và không giống tiêu đề
                    if not any(junk in line for junk in ignore_list) and title not in line:
                        row['author'] = line
                        break

            # --- XỬ LÝ SAPO VÀ CONTENT (Phần sau ngày) ---
            body_lines = [p.strip() for p in post_date.split('\n') if p.strip()]
            
            if body_lines:
                # Dòng đầu tiên sau ngày tháng là Short Description (Sapo)
                if pd.isna(row.get('short_description')) or row['short_description'] == '':
                    row['short_description'] = body_lines[0]
                
                # Các dòng còn lại là Content chính
                if len(body_lines) > 1:
                    row['content'] = "\n".join(body_lines[1:])
                else:
                    row['content'] = "" # Đã bóc hết vào Sapo
        
        return row

    # Áp dụng logic
    print("Đang xử lý dữ liệu...")
    df_clean = df.apply(clean_row, axis=1)

    # Lưu file
    print(f"Đang lưu kết quả ra: {output_file}")
    df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("Hoàn tất!")

if __name__ == "__main__":
    # Thiết lập bộ nhận diện tham số dòng lệnh (Arguments Parser)
    parser = argparse.ArgumentParser(description="Tool làm sạch dữ liệu bài báo (Tách Author, Date, Sapo từ Content)")
    
    # Định nghĩa các tham số (args)
    parser.add_argument('-i', '--input', type=str, required=True, help="Đường dẫn file CSV đầu vào (Bắt buộc)")
    parser.add_argument('-o', '--output', type=str, default="cleaned_data.csv", help="Đường dẫn file CSV đầu ra (Mặc định: cleaned_data.csv)")

    # Lấy tham số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm xử lý
    process_cleaning(args.input, args.output)