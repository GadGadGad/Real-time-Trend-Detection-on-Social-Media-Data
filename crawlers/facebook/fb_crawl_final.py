import time
import json
import os
import re
import argparse
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import print as rprint

console = Console()


# ================= CẤU HÌNH DỰ ÁN =================
# Cấu hình sẽ được parse từ argparse
# ==================================================

def init_json(file_output):
    if not os.path.exists(file_output):
        with open(file_output, 'w', encoding='utf-8') as f:
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

def scroll_and_wait_for_content(driver, min_posts=20, max_attempts=15, scroll_pause_time=3):
    """
    Cuộn trang để tải thêm bài viết (lazy loading).
    Pattern: repeat: down -> up (half-page) until no more new posts
    Trả về số bài viết đã phát hiện.
    """
    console.print(f"  [cyan]📜 Đang cuộn trang để tải bài viết (mục tiêu: {min_posts} bài)...[/cyan]")
    
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
            console.print(f"  [bold green]✅ Đã tải đủ {current_posts} bài viết![/bold green]")
            break
        
        scroll_count += 1
        
        # === BƯỚC 1: Cuộn xuống cuối trang ===
        scroll_to_bottom()
        time.sleep(scroll_pause_time)
        
        # === BƯỚC 2: Cuộn lên nửa trang ===
        scroll_to_half()
        time.sleep(1)
        
        # Mở rộng các bài viết bị thu gọn
        expand_all(driver)
        
        # Kiểm tra chiều cao mới
        new_height = get_page_height()
        current_posts = count_posts()
        
        console.print(f"    [dim]Lần cuộn {scroll_count}: đã tìm thấy ~{current_posts} bài viết[/dim]")
        
        if new_height == last_height:
            no_new_content_count += 1
            if no_new_content_count >= 3:
                console.print(f"  [yellow]⚠️ Không còn nội dung mới sau {scroll_count} lần cuộn.[/yellow]")
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

def save_to_json(new_posts, page_name, file_output):
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
        
        console.print(f"    [green]✅[/green] [bold]POST[/bold] Time: [cyan]{p['time_text']}[/cyan] | {full_text[:40]}...")

    try:
        with open(file_output, 'r', encoding='utf-8') as f:
            try: current_data = json.load(f)
            except: current_data = []
        
        current_data.extend(clean_data)
        
        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        console.print(f"[bold red]❌ Lỗi ghi file: {e}[/bold red]")

    return len(clean_data)

def create_browser(browser_type, profile_path=None):
    """
    Tạo browser driver dựa trên loại browser được chọn.
    Hỗ trợ: firefox, chrome, chromium
    """
    # Use iOS user agent instead of Android to avoid intent:// redirects
    # iOS doesn't have intent protocol, so Facebook won't try to open mobile app
        # mobile_ua = "Mozilla/5.0 (Linux; Android 11; SAMSUNG SM-G973U) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/14.2 Chrome/87.0.4280.141 Mobile Safari/537.36"
    mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    if browser_type == "firefox":
        console.print("[yellow]🦊 Đang khởi động Firefox...[/yellow]")
        options = FirefoxOptions()
        options.set_preference("general.useragent.override", mobile_ua)
        
        if profile_path:
            abs_profile_path = os.path.abspath(profile_path)
            if not os.path.exists(abs_profile_path):
                console.print(f"[dim]📁 Tạo mới profile folder tại: {abs_profile_path}[/dim]")
                os.makedirs(abs_profile_path)
            options.add_argument("-profile")
            options.add_argument(abs_profile_path)
        
        return webdriver.Firefox(options=options)
    
    elif browser_type in ["chrome", "chromium"]:
        browser_name = "Chrome" if browser_type == "chrome" else "Chromium"
        console.print(f"[yellow]🌐 Đang khởi động {browser_name}...[/yellow]")
        
        options = ChromeOptions()
        options.add_argument(f"--user-agent={mobile_ua}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Prevent Facebook from redirecting to intent:// (Android app) URLs
        options.add_argument("--disable-external-intents-redirect")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--ignore-certificate-errors")
        
        # Disable Chrome's built-in intent handling
        prefs = {
            "protocol_handler.excluded_schemes": {
                "intent": True
            },
            "profile.default_content_setting_values.notifications": 2
        }
        options.add_experimental_option("prefs", prefs)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        # Cho Chromium, cần chỉ định đường dẫn binary
        if browser_type == "chromium":
            # Common Chromium paths on Linux
            chromium_paths = [
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
                "/snap/bin/chromium",
                "/usr/lib/chromium/chromium",
            ]
            for path in chromium_paths:
                if os.path.exists(path):
                    options.binary_location = path
                    break
            else:
                console.print("[yellow]⚠️ Không tìm thấy Chromium, thử dùng Chrome...[/yellow]")
        
        if profile_path:
            abs_profile_path = os.path.abspath(profile_path)
            if not os.path.exists(abs_profile_path):
                console.print(f"[dim]📁 Tạo mới profile folder tại: {abs_profile_path}[/dim]")
                os.makedirs(abs_profile_path)
            options.add_argument(f"--user-data-dir={abs_profile_path}")
        
        return webdriver.Chrome(options=options)
    
    else:
        raise ValueError(f"Browser không được hỗ trợ: {browser_type}. Chọn: firefox, chrome, chromium")

def run_crawler(browser_type="firefox", target_urls=None, file_output="fb_data.json", min_posts=20, max_scrolls=15, scroll_pause=3, profile_path=None):
    if target_urls is None:
        target_urls = []

    init_json(file_output) # khởi tạo json
    
    browser_display = {
        "firefox": "Firefox 🦊",
        "chrome": "Chrome 🌐", 
        "chromium": "Chromium 🌐"
    }.get(browser_type, browser_type)
    
    console.print(Panel.fit(
        f"[bold cyan]Facebook Crawler[/bold cyan]\n[dim]Using {browser_display} + Selenium[/dim]",
        border_style="blue"
    ))
    
    # Tạo profile path riêng cho mỗi browser nếu chưa có
    if not profile_path:
        profile_path = f"my_{browser_type}_profile"
        
    driver = create_browser(browser_type, profile_path)
    
    try:
        console.print("[cyan]🌐 Đang vào Facebook...[/cyan]")
        driver.get("https://m.facebook.com")
        time.sleep(3)
        
        # Kiểm tra đăng nhập - check nhiều điều kiện để hoạt động với mọi browser
        def is_logged_in():
            current_url = driver.current_url.lower()
            page_source = driver.page_source.lower()
            
            # Chưa đăng nhập nếu:
            # 1. URL chứa "login" hoặc "checkpoint"
            # 2. Trang có form đăng nhập
            # 3. Trang có nút "Log In" hoặc "Đăng nhập"
            login_url_keywords = ["login", "checkpoint", "recover", "identify"]
            if any(kw in current_url for kw in login_url_keywords):
                return False
            
            # Kiểm tra có form login không
            login_indicators = [
                'name="email"',
                'name="pass"', 
                'id="loginbutton"',
                'data-sigil="login_button"',
                'data-sigil="m_login_button"'
            ]
            if any(indicator in page_source for indicator in login_indicators):
                return False
                
            return True
        
        if not is_logged_in():
            console.print("[bold yellow]⚠️ CHƯA ĐĂNG NHẬP![/bold yellow]")
            console.print("[yellow]👉 Hãy đăng nhập Facebook thủ công trong cửa sổ trình duyệt.[/yellow]")
            console.print("[yellow]👉 Sau khi đăng nhập xong, quay lại đây và bấm Enter để tiếp tục...[/yellow]")
            input()
            
            # Đợi thêm sau khi user bấm Enter để đảm bảo trang đã load xong
            time.sleep(2)
        
        for url in target_urls:
            page_name_slug = url.split('/')[-1]
            
            console.print()
            console.rule(f"[bold blue]{page_name_slug}[/bold blue]")
            
            driver.get(url)
            print(driver.current_url)

            time.sleep(5)
            
            # Cuộn trang để tải thêm bài viết (lazy loading)
            scroll_and_wait_for_content(driver, min_posts, max_scrolls, scroll_pause)
            
            # Mở rộng tất cả bài viết trước khi parse
            expand_all(driver)
            
            console.print("  [cyan]🔄 Đang phân tích luồng bài viết...[/cyan]")
            posts = parse_stream(driver, page_name_slug)
            
            saved = save_to_json(posts, page_name_slug, file_output)
            console.print(f"  [bold green]🏁 Đã lưu {saved} bài vào {file_output}.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]❌ Lỗi Critical: {e}[/bold red]")
    finally:
        console.print()
        console.print(Panel.fit("[bold]🛑 Kết thúc session.[/bold]", border_style="red"))
        # driver.quit() # Mở dòng này nếu muốn tự động đóng trình duyệt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Facebook Crawler - Crawl posts from Facebook pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Ví dụ sử dụng:
            python fb_crawl_final.py                    # Mặc định dùng Firefox, crawl URLs mặc định
            python fb_crawl_final.py --browser chrome   # Dùng Chrome
            python fb_crawl_final.py --urls https://www.facebook.com/kienthuc.net.vn
            python fb_crawl_final.py --min-posts 50 --max-scrolls 20
        """
    )
    
    default_urls = [
        "https://m.facebook.com/Theanh28",
        "https://m.facebook.com/kienkhongngu.vn", 
        "https://m.facebook.com/thongtinchinhphu",    # Vietnam News (TTXVN)
        "https://m.facebook.com/VnExpress",           # VnExpress
        "https://m.facebook.com/baotuoitre",          # Tuổi Trẻ
        "https://m.facebook.com/thanhnien",           # Thanh Niên
        "https://m.facebook.com/vietnamnet.vn",       # Vietnamnet
        "https://m.facebook.com/baodantridientu",     # Dân Trí
        "https://m.facebook.com/laodongonline",       # Báo Lao Động
        "https://m.facebook.com/nhandanonline",       # Báo Nhân Dân
        "https://m.facebook.com/profile.php?id=100089883616175",
        # "https://m.facebook.com/hhsb.vn/",
        "https://m.facebook.com/tintucvietnammoinong/",
        "https://www.facebook.com/tintucvtv24",
        "https://www.facebook.com/doisongvnn",
        "https://www.facebook.com/VnProCon",
        "https://www.facebook.com/tapchitrithucznews.vn",
    ]

    parser.add_argument("-b", "--browser", type=str, choices=["firefox", "chrome", "chromium"], default="firefox", help="Chọn trình duyệt (mặc định: firefox)")
    parser.add_argument("-u", "--urls", type=str, nargs="+", default=default_urls, help="Danh sách URL cần crawl")
    parser.add_argument("-o", "--output", type=str, default="fb_data.json", help="Đường dẫn file output JSON")
    parser.add_argument("--min-posts", type=int, default=500, help="Số bài viết tối thiểu mỗi page")
    parser.add_argument("--max-scrolls", type=int, default=200, help="Số lần scroll tối đa")
    parser.add_argument("--scroll-pause", type=int, default=3, help="Thời gian chờ sau mỗi lần scroll (giây)")
    parser.add_argument("--profile-path", type=str, default="my_firefox_profile", help="Đường dẫn thư mục profile (nếu không set, tự động tạo theo browser)")
    
    args = parser.parse_args()

    # Xử lý đường dẫn tuyệt đối cho output nếu cần
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_output = args.output
    if not os.path.isabs(file_output):
        file_output = os.path.join(script_dir, file_output)

    run_crawler(
        browser_type=args.browser,
        target_urls=args.urls,
        file_output=file_output,
        min_posts=args.min_posts,
        max_scrolls=args.max_scrolls,
        scroll_pause=args.scroll_pause,
        profile_path=args.profile_path
    )