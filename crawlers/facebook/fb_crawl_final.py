import time
import json
import os
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

# ================= CẤU HÌNH DỰ ÁN =================
# Lấy đường dẫn thư mục chứa script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_OUTPUT = os.path.join(SCRIPT_DIR, "fb_data.json")

TARGET_URLS = [
    "https://m.facebook.com/Theanh28",
    "https://www.facebook.com/kienkhongngu.vn", 
    "https://www.facebook.com/thongtinchinhphu",
]

PROFILE_PATH = "my_firefox_profile"

# Cấu hình số bài viết tối thiểu cần crawl
MIN_POSTS_PER_PAGE = 20  # Số bài tối thiểu muốn lấy mỗi page
MAX_SCROLL_ATTEMPTS = 15  # Số lần scroll tối đa
SCROLL_PAUSE_TIME = 3  # Thời gian chờ sau mỗi lần scroll (giây)
# ==================================================

def init_json():
    if not os.path.exists(FILE_OUTPUT):
        with open(FILE_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

def clean_number(text):
    if not text: return 0
    match = re.search(r'(\d+[.,]?\d*[MK]?)', text)
    if not match: return 0
    
    num_str = match.group(1).replace(',', '.')
    multiplier = 1
    if 'K' in num_str:
        multiplier = 1000
        num_str = num_str.replace('K', '')
    elif 'M' in num_str:
        multiplier = 1000000
        num_str = num_str.replace('M', '')
        
    try:
        return int(float(num_str) * multiplier)
    except:
        return 0

def clean_time_string(text):
    """
    Làm sạch chuỗi thời gian chứa ký tự ẩn (\u200e) và icon lạ
    Input: "‎4h‎󰞋󱙷" -> Output: "4h"
    """
    if not text: return ""
    # Giữ lại chữ cái (bao gồm tiếng Việt), số, khoảng trắng, dấu chấm
    clean = re.sub(r'[^\w\s,.]', '', text).strip()
    
    # Nếu dính icon ở cuối (VD: "4hIcon"), chỉ lấy phần text đầu tiên
    match = re.match(r'^([\w\s,.]+)', clean)
    if match:
        return match.group(1).strip()
    return clean

def is_timestamp(text):
    """
    Kiểm tra xem text có phải là định dạng thời gian không.
    """
    # Làm sạch trước khi check regex
    clean_text = clean_time_string(text).lower()
    
    if not clean_text: return False
    if len(clean_text) > 30: return False # Time label thường rất ngắn

    # Pattern: 1h, 12m, 3d, 4y, 2w (chấp nhận cả khoảng trắng "1 h")
    if re.match(r'^\d+\s*[wmhdys]$', clean_text): return True
    
    # Pattern with "hrs", "min", "mins", "day", "days"
    if re.match(r'^\d+\s*(hrs?|mins?|days?|weeks?|months?|years?)$', clean_text): return True
    
    # Pattern Tiếng Việt: 2 giờ, 5 phút, 1 ngày, 27 thg 11
    if re.match(r'^\d+\s*(giờ|phút|ngày|năm|tháng|tuần)', clean_text): return True
    if re.match(r'^\d{1,2}\s+thg\s+\d{1,2}.*', clean_text): return True
    
    # Pattern Tiếng Anh: Nov 27, December 4, Jan 15
    if re.match(r'^[a-z]{3,9}\s\d{1,2}.*', clean_text): return True
    
    # Pattern: số giờ trước (VD: "4 giờ trước", "3 hours ago")
    if re.search(r'(giờ|phút|ngày|tuần|tháng|năm|hour|minute|day|week|month|year)s?\s*(trước|ago)', clean_text): 
        return True
    
    # Các từ khóa đặc biệt
    keywords = ["yesterday", "today", "hôm qua", "hôm nay", "vừa xong", "just now", "mins", "hrs", "đã đăng"]
    if any(k in clean_text for k in keywords): return True
    
    return False

def calculate_publish_time(raw_time_text):
    """
    Chuyển đổi text Facebook (3h, 5m, Yesterday) thành thời gian thực (ISO Format)
    """
    if not raw_time_text: return None
    
    now = datetime.now()
    text = raw_time_text.lower().strip()
    
    try:
        # 1. Xử lý phút (m, min, phút)
        match = re.search(r'^(\d+)\s*[m|min|phút]', text)
        if match:
            minutes = int(match.group(1))
            return (now - timedelta(minutes=minutes)).isoformat()

        # 2. Xử lý giờ (h, hr, giờ)
        match = re.search(r'^(\d+)\s*[h|hr|giờ]', text)
        if match:
            hours = int(match.group(1))
            return (now - timedelta(hours=hours)).isoformat()

        # 3. Xử lý ngày (d, day, ngày)
        match = re.search(r'^(\d+)\s*[d|day|ngày]', text)
        if match:
            days = int(match.group(1))
            return (now - timedelta(days=days)).isoformat()
            
        # 4. Xử lý "Hôm qua" / "Yesterday"
        if "yesterday" in text or "hôm qua" in text:
            return (now - timedelta(days=1)).isoformat()
            
        # 5. Xử lý năm (1y)
        match = re.search(r'^(\d+)\s*y', text)
        if match:
            years = int(match.group(1))
            return (now - timedelta(days=years*365)).isoformat()

    except:
        pass
        
    return None

def expand_all(driver):
    try:
        btns = driver.find_elements(By.XPATH, "//span[contains(text(), 'See more') or contains(text(), 'Xem thêm') or text()='…']")
        if btns:
            for btn in btns:
                try: driver.execute_script("arguments[0].click();", btn)
                except: pass
            time.sleep(2) 
    except: pass

def scroll_and_wait_for_content(driver, min_posts=MIN_POSTS_PER_PAGE, max_attempts=MAX_SCROLL_ATTEMPTS):
    """
    Cuộn trang để tải thêm bài viết (lazy loading).
    Pattern: repeat: down -> up (half-page) until no more new posts
    Trả về số bài viết đã phát hiện.
    """
    print(f"  📜 Đang cuộn trang để tải bài viết (mục tiêu: {min_posts} bài)...")
    
    scroll_count = 0
    no_new_content_count = 0
    last_height = 0
    
    def count_posts():
        try:
            stream_container = driver.find_element(By.XPATH, "//div[@data-type='vscroller']")
            blocks = stream_container.find_elements(By.XPATH, "./div")
            return len([b for b in blocks if 'data-testid="post-profile-image' in b.get_attribute('outerHTML')])
        except:
            return 0
    
    def get_page_height():
        return driver.execute_script("return document.body.scrollHeight")
    
    def get_current_scroll():
        return driver.execute_script("return window.pageYOffset")
    
    def smooth_scroll(target_y, duration=1.0):
        """Cuộn mượt đến vị trí target_y trong khoảng duration giây"""
        start_y = get_current_scroll()
        distance = target_y - start_y
        steps = 20  # Số bước cuộn
        step_delay = duration / steps
        
        for i in range(1, steps + 1):
            # Easing function (ease-out)
            progress = i / steps
            eased_progress = 1 - (1 - progress) ** 2
            new_y = start_y + (distance * eased_progress)
            driver.execute_script(f"window.scrollTo(0, {int(new_y)});")
            time.sleep(step_delay)
    
    def scroll_to_bottom():
        target = get_page_height()
        smooth_scroll(target, duration=1.5)
    
    def scroll_to_half():
        target = get_page_height() // 2
        smooth_scroll(target, duration=1.0)
    
    while scroll_count < max_attempts:
        current_posts = count_posts()
        
        if current_posts >= min_posts:
            print(f"  ✅ Đã tải đủ {current_posts} bài viết!")
            break
        
        scroll_count += 1
        
        # === BƯỚC 1: Cuộn xuống cuối trang ===
        scroll_to_bottom()
        time.sleep(SCROLL_PAUSE_TIME)
        
        # === BƯỚC 2: Cuộn lên nửa trang ===
        scroll_to_half()
        time.sleep(1)
        
        # Mở rộng các bài viết bị thu gọn
        expand_all(driver)
        
        # Kiểm tra chiều cao mới
        new_height = get_page_height()
        current_posts = count_posts()
        
        print(f"    Lần cuộn {scroll_count}: đã tìm thấy ~{current_posts} bài viết")
        
        if new_height == last_height:
            no_new_content_count += 1
            if no_new_content_count >= 3:
                print(f"  ⚠️ Không còn nội dung mới sau {scroll_count} lần cuộn.")
                break
        else:
            no_new_content_count = 0
            last_height = new_height
    
    return count_posts()


def parse_stream(driver, page_name_slug):
    posts_data = []
    
    try:
        stream_container = driver.find_element(By.XPATH, "//div[@data-type='vscroller']")
        blocks = stream_container.find_elements(By.XPATH, "./div")
    except:
        return []

    current_post = {
        "text": [], "images": [], "videos": [], 
        "likes": 0, "comments": 0, "shares": 0,
        "time_text": "",
        "has_avatar": False
    }

    for block in blocks:
        html_block = block.get_attribute('outerHTML')
        
        # 1. BẮT ĐẦU BÀI MỚI (Dựa vào Avatar)
        if 'data-testid="post-profile-image' in html_block:
            if current_post["has_avatar"] and (current_post["text"] or current_post["images"]):
                posts_data.append(current_post)
            
            current_post = {
                "text": [], "images": [], "videos": [],
                "likes": 0, "comments": 0, "shares": 0,
                "time_text": "",
                "has_avatar": True
            }
            
            # ===== TÌM THỜI GIAN NGAY TRONG BLOCK AVATAR =====
            # Facebook thường đặt timestamp gần avatar (ở header của post)
            try:
                # Phương pháp 1: Tìm trong abbr tag (có thể có data-utime)
                abbr_elems = block.find_elements(By.TAG_NAME, "abbr")
                for abbr in abbr_elems:
                    time_attr = abbr.get_attribute("data-utime") or abbr.get_attribute("title")
                    if time_attr:
                        current_post["time_text"] = time_attr
                        break
                
                # Phương pháp 2: Tìm các span/div chứa text thời gian
                if not current_post["time_text"]:
                    time_candidates = block.find_elements(By.XPATH, ".//span | .//a")
                    for tc in time_candidates:
                        raw_t = driver.execute_script("return arguments[0].textContent;", tc).strip()
                        if raw_t and is_timestamp(raw_t):
                            current_post["time_text"] = clean_time_string(raw_t)
                            break
            except:
                pass
            continue

        if not current_post["has_avatar"]: continue

        # 2. QUÉT TEXT AREA (Xử lý cả Time label lẫn Content tại đây)
        try:
            # Tìm tất cả TextArea (vì trên mobile, time label cũng nằm trong TextArea)
            text_elems = block.find_elements(By.XPATH, ".//div[contains(@data-mcomponent, 'TextArea')]")
            
            for elem in text_elems:
                # Lấy text thô (raw)
                raw_txt = driver.execute_script("return arguments[0].textContent;", elem).strip()
                if not raw_txt: continue
                
                # --- CHECK XEM CÓ PHẢI LÀ TIME KHÔNG ---
                # Chỉ check nếu bài hiện tại chưa có time
                if not current_post["time_text"]:
                    if is_timestamp(raw_txt):
                        # Làm sạch (bỏ icon trái đất, bỏ ký tự ẩn)
                        clean_t = clean_time_string(raw_txt)
                        current_post["time_text"] = clean_t
                        # Nếu đã là time thì bỏ qua, không add vào content
                        continue 
                # ---------------------------------------

                # NẾU KHÔNG PHẢI TIME THÌ LÀ CONTENT
                txt_lower = raw_txt.lower()
                
                # Logic lọc rác cũ của bạn
                if page_name_slug.lower() in txt_lower and len(raw_txt) < 50:
                    continue # Bỏ qua tên page lặp lại
                
                if "comment" not in txt_lower and "share" not in txt_lower:
                    clean_txt = raw_txt.replace("... See more", "").replace("... Xem thêm", "")
                    current_post["text"].append(clean_txt)
        except: pass

        # 3. ẢNH/VIDEO
        try:
            imgs = block.find_elements(By.TAG_NAME, "img")
            for img in imgs:
                alt = img.get_attribute("alt")
                if alt and len(alt) > 15 and "profile picture" not in alt.lower():
                    current_post["images"].append(alt)
            
            if 'data-type="video"' in html_block or 'aria-label="Video player"' in html_block:
                current_post["videos"].append("Video")
        except: pass

        # 4. STATS
        try:
            btns = block.find_elements(By.XPATH, ".//div[@role='button']")
            for btn in btns:
                label = btn.get_attribute("aria-label")
                if not label: continue
                
                if 'like' in label.lower() and 'comment' not in label.lower():
                    current_post["likes"] = clean_number(label)
                elif 'comment' in label.lower():
                    current_post["comments"] = clean_number(label)
                elif 'share' in label.lower():
                    current_post["shares"] = clean_number(label)
        except: pass

    if current_post["has_avatar"] and (current_post["text"] or current_post["images"]):
        posts_data.append(current_post)

    return posts_data

def save_to_json(new_posts, page_name):
    if not new_posts: return 0
    clean_data = []
    crawl_timestamp = datetime.now().isoformat()
    
    for p in new_posts:
        full_text = "\n".join(p["text"])
        
        # Bỏ qua bài quá ngắn và không có ảnh/video
        if len(full_text) < 5 and not p["images"] and not p["videos"]:
            continue
            
        # --- TÍNH TOÁN GIỜ ĐĂNG ---
        calculated_time = calculate_publish_time(p["time_text"])
        if not calculated_time:
            calculated_time = crawl_timestamp # Fallback nếu không tính được
        # --------------------------

        post_obj = {
            "page_name": page_name,
            "published_time": calculated_time,
            "crawl_time": crawl_timestamp,
            "time_label": p["time_text"],      # Label đã được clean (VD: 4h)
            "content": full_text,
            "media": {
                "images": p["images"],
                "videos": p["videos"]
            },
            "stats": {
                "likes": p["likes"],
                "comments": p["comments"],
                "shares": p["shares"]
            }
        }
        clean_data.append(post_obj)
        
        print(f"    ✅ [POST] Time: {p['time_text']} | Content: {full_text[:30]}...")

    try:
        with open(FILE_OUTPUT, 'r', encoding='utf-8') as f:
            try: current_data = json.load(f)
            except: current_data = []
        
        current_data.extend(clean_data)
        
        with open(FILE_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"❌ Lỗi ghi file: {e}")

    return len(clean_data)

def run_crawler():
    init_json() # khởi tạo json
    print("🚗 Đang khởi động Firefox (Fixed Time Logic)...")
    
    options = Options()
    # User Agent giả lập Android để ép về giao diện m.facebook nhẹ nhất
    mobile_ua = "Mozilla/5.0 (Linux; Android 11; SAMSUNG SM-G973U) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/14.2 Chrome/87.0.4280.141 Mobile Safari/537.36"
    options.set_preference("general.useragent.override", mobile_ua)
    
    abs_profile_path = os.path.abspath(PROFILE_PATH)
    if not os.path.exists(abs_profile_path):
        print(f"📁 Tạo mới profile folder tại: {abs_profile_path}")
        os.makedirs(abs_profile_path)
        
    if os.path.exists(abs_profile_path):
        options.add_argument("-profile")
        options.add_argument(abs_profile_path)
    
    driver = webdriver.Firefox(options=options)
    
    try:
        print("🌐 Đang vào Facebook...")
        driver.get("https://m.facebook.com")
        time.sleep(3)
        if "login" in driver.current_url:
            print("⚠️ CHƯA ĐĂNG NHẬP! Hãy đăng nhập thủ công rồi quay lại đây bấm Enter...")
            input()
        
        for url in TARGET_URLS:
            page_name_slug = url.split('/')[-1]
            print(f"\n--- Đang xử lý Page: {page_name_slug} ---")
            driver.get(url)
            time.sleep(5)
            
            # Cuộn trang để tải thêm bài viết (lazy loading)
            scroll_and_wait_for_content(driver, MIN_POSTS_PER_PAGE, MAX_SCROLL_ATTEMPTS)
            
            # Mở rộng tất cả bài viết trước khi parse
            expand_all(driver)
            
            print("  🔄 Đang phân tích luồng bài viết...")
            posts = parse_stream(driver, page_name_slug)
            
            saved = save_to_json(posts, page_name_slug)
            print(f"  🏁 Đã lưu {saved} bài vào {FILE_OUTPUT}.")

    except Exception as e:
        print(f"❌ Lỗi Critical: {e}")
    finally:
        print("🛑 Kết thúc session.")
        # driver.quit() # Mở dòng này nếu muốn tự động đóng trình duyệt

if __name__ == "__main__":
    run_crawler()