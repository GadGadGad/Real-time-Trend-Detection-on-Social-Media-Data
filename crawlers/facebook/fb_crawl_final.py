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
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import print as rprint

console = Console()

# ================= CẤU HÌNH VÀ HELPER =================

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
    if not text: return ""
    clean = re.sub(r'[^\w\s,.]', '', text).strip()
    match = re.match(r'^([\w\s,.]+)', clean)
    if match:
        return match.group(1).strip()
    return clean

def is_timestamp(text):
    clean_text = clean_time_string(text).lower()
    if not clean_text: return False
    if len(clean_text) > 30: return False 

    if re.match(r'^\d+\s*[wmhdys]$', clean_text): return True
    if re.match(r'^\d+\s*(hrs?|mins?|days?|weeks?|months?|years?)$', clean_text): return True
    if re.match(r'^\d+\s*(giờ|phút|ngày|năm|tháng|tuần)', clean_text): return True
    if re.match(r'^\d{1,2}\s+thg\s+\d{1,2}.*', clean_text): return True
    if re.match(r'^[a-z]{3,9}\s\d{1,2}.*', clean_text): return True
    if re.search(r'(giờ|phút|ngày|tuần|tháng|năm|hour|minute|day|week|month|year)s?\s*(trước|ago)', clean_text): return True
    
    keywords = ["yesterday", "today", "hôm qua", "hôm nay", "vừa xong", "just now", "mins", "hrs", "đã đăng"]
    if any(k in clean_text for k in keywords): return True
    
    return False

def calculate_publish_time(raw_time_text):
    if not raw_time_text: return None
    now = datetime.now()
    text = raw_time_text.lower().strip()
    try:
        match = re.search(r'^(\d+)\s*[m|min|phút]', text)
        if match: return (now - timedelta(minutes=int(match.group(1)))).isoformat()

        match = re.search(r'^(\d+)\s*[h|hr|giờ]', text)
        if match: return (now - timedelta(hours=int(match.group(1)))).isoformat()

        match = re.search(r'^(\d+)\s*[d|day|ngày]', text)
        if match: return (now - timedelta(days=int(match.group(1)))).isoformat()
            
        if "yesterday" in text or "hôm qua" in text:
            return (now - timedelta(days=1)).isoformat()
            
        match = re.search(r'^(\d+)\s*y', text)
        if match: return (now - timedelta(days=int(match.group(1))*365)).isoformat()
    except: pass
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
        "time_text": "", "has_avatar": False
    }

    for block in blocks:
        try:
            # === BẮT ĐẦU VÙNG AN TOÀN (Bỏ qua nếu Element bị Facebook xóa) ===
            html_block = block.get_attribute('outerHTML')
            
            # 1. BẮT ĐẦU BÀI MỚI (Dựa vào Avatar)
            if 'data-testid="post-profile-image' in html_block:
                if current_post["has_avatar"] and (current_post["text"] or current_post["images"]):
                    posts_data.append(current_post)
                
                current_post = {
                    "text": [], "images": [], "videos": [],
                    "likes": 0, "comments": 0, "shares": 0,
                    "time_text": "", "has_avatar": True
                }
                
                # Tìm thời gian
                try:
                    abbr_elems = block.find_elements(By.TAG_NAME, "abbr")
                    for abbr in abbr_elems:
                        time_attr = abbr.get_attribute("data-utime") or abbr.get_attribute("title")
                        if time_attr:
                            current_post["time_text"] = time_attr
                            break
                    
                    if not current_post["time_text"]:
                        time_candidates = block.find_elements(By.XPATH, ".//span | .//a")
                        for tc in time_candidates:
                            raw_t = driver.execute_script("return arguments[0].textContent;", tc).strip()
                            if raw_t and is_timestamp(raw_t):
                                current_post["time_text"] = clean_time_string(raw_t)
                                break
                except: pass
                continue

            if not current_post["has_avatar"]: continue

            # 2. QUÉT TEXT
            try:
                text_elems = block.find_elements(By.XPATH, ".//div[contains(@data-mcomponent, 'TextArea')]")
                for elem in text_elems:
                    raw_txt = driver.execute_script("return arguments[0].textContent;", elem).strip()
                    if not raw_txt: continue
                    
                    if not current_post["time_text"]:
                        if is_timestamp(raw_txt):
                            current_post["time_text"] = clean_time_string(raw_txt)
                            continue 

                    txt_lower = raw_txt.lower()
                    if page_name_slug.lower() in txt_lower and len(raw_txt) < 50: continue
                    
                    if "comment" not in txt_lower and "share" not in txt_lower:
                        clean_txt = raw_txt.replace("... See more", "").replace("... Xem thêm", "")
                        current_post["text"].append(clean_txt)
            except: pass

            # 3. QUÉT MEDIA
            try:
                imgs = block.find_elements(By.TAG_NAME, "img")
                for img in imgs:
                    alt = img.get_attribute("alt")
                    if alt and len(alt) > 15 and "profile picture" not in alt.lower():
                        current_post["images"].append(alt)
                
                if 'data-type="video"' in html_block or 'aria-label="Video player"' in html_block:
                    current_post["videos"].append("Video")
            except: pass

            # 4. QUÉT STATS
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
            
            # === KẾT THÚC VÙNG AN TOÀN ===

        except StaleElementReferenceException:
            # Nếu element này bị lỗi (do FB refresh DOM), bỏ qua và đi tiếp
            continue

    if current_post["has_avatar"] and (current_post["text"] or current_post["images"]):
        posts_data.append(current_post)

    return posts_data

def save_to_json(new_posts, page_name, file_output):
    if not new_posts: return 0
    clean_data = []
    crawl_timestamp = datetime.now().isoformat()
    
    for p in new_posts:
        full_text = "\n".join(p["text"])
        
        if len(full_text) < 5 and not p["images"] and not p["videos"]:
            continue
            
        calculated_time = calculate_publish_time(p["time_text"])
        if not calculated_time:
            calculated_time = crawl_timestamp 

        post_obj = {
            "page_name": page_name,
            "published_time": calculated_time,
            "crawl_time": crawl_timestamp,
            "time_label": p["time_text"],
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
        console.print(f"    [green]✅[/green] [bold]LƯU POST[/bold] | {full_text[:40]}...")

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

# ================= CORE LOGIC ĐƯỢC SỬA ĐỔI =================

def process_page_scrolling(driver, page_name_slug, file_output, min_posts, max_attempts, scroll_pause_time):
    console.print(f"  [cyan]📜 Đang quét bài viết (Mục tiêu: {min_posts} bài)...[/cyan]")
    
    total_saved = 0
    processed_signatures = set()
    last_height = driver.execute_script("return document.body.scrollHeight")
    stuck_count = 0 # Đếm số lần bị kẹt liên tiếp

    # Helper: Cuộn xuống từ từ thay vì nhảy cóc
    def human_scroll_down(step_delay=0.5):
        # Lấy chiều cao màn hình hiện tại
        viewport_height = driver.execute_script("return window.innerHeight")
        current_pos = driver.execute_script("return window.pageYOffset")
        doc_height = driver.execute_script("return document.body.scrollHeight")
        
        # Cuộn từng đoạn bằng 80% chiều cao màn hình
        while current_pos < doc_height:
            current_pos += int(viewport_height * 0.8)
            driver.execute_script(f"window.scrollTo(0, {current_pos});")
            time.sleep(step_delay) # Nghỉ ngắn giữa mỗi cú vuốt
            
            # Cập nhật lại doc_height vì có thể FB đã load thêm ngay trong lúc đang vuốt
            new_doc_height = driver.execute_script("return document.body.scrollHeight")
            if new_doc_height > doc_height:
                doc_height = new_doc_height
            
            # Nếu đã chạm đáy thực tế
            if current_pos >= doc_height:
                break

    # Helper: Thao tác giải cứu khi bị kẹt
    def unstuck_maneuver():
        console.print("    [yellow]⚠️ Có vẻ bị kẹt, đang thử cuộn ngược để kích hoạt...[/yellow]")
        # Cuộn lên 1 khoảng khá xa (khoảng 3 màn hình)
        driver.execute_script("window.scrollBy(0, -1500);")
        time.sleep(2)
        # Cuộn từ từ xuống lại đáy
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    loop_count = 0
    while total_saved < min_posts and loop_count < max_attempts:
        loop_count += 1
        
        # 1. THỰC HIỆN CUỘN
        # Nếu đang bị kẹt, dùng biện pháp mạnh, nếu không thì cuộn bình thường
        if stuck_count >= 1:
            unstuck_maneuver()
        else:
            human_scroll_down(step_delay=0.2)
        
        time.sleep(scroll_pause_time)
        
        # 2. KIỂM TRA CHIỀU CAO ĐỂ BIẾT CÓ LOAD THÊM KHÔNG
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stuck_count += 1
        else:
            stuck_count = 0 # Reset nếu đã load được thêm
            last_height = new_height
            
        # 3. EXPAND & PARSE (Giữ nguyên logic cũ)
        expand_all(driver)
        raw_posts = parse_stream(driver, page_name_slug)
        
        new_batch = []
        for p in raw_posts:
            # Check trùng lặp
            content_sig = "\n".join(p["text"])[:50]
            unique_key = f"{p['time_text']}_{content_sig}"
            
            if unique_key not in processed_signatures:
                processed_signatures.add(unique_key)
                new_batch.append(p)
        
        if new_batch:
            saved_count = save_to_json(new_batch, page_name_slug, file_output)
            total_saved += saved_count
            # Nếu lưu được bài mới, coi như không bị kẹt, reset stuck_count
            stuck_count = 0
        
        console.print(f"    [dim]Lần {loop_count}: Thêm {len(new_batch)} bài. Tổng: {total_saved}/{min_posts}. (Stuck: {stuck_count})[/dim]")
        
        if total_saved >= min_posts:
            console.print(f"  [bold green]✅ Đã thu thập đủ {total_saved} bài![/bold green]")
            break
            
    if total_saved < min_posts:
        console.print(f"  [yellow]⚠️ Dừng vòng lặp. Đã lưu: {total_saved}[/yellow]")

    return total_saved

def create_browser(browser_type, profile_path=None):
    mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    if browser_type == "firefox":
        console.print("[yellow]🦊 Đang khởi động Firefox...[/yellow]")
        options = FirefoxOptions()
        options.set_preference("general.useragent.override", mobile_ua)
        if profile_path:
            abs_profile_path = os.path.abspath(profile_path)
            if not os.path.exists(abs_profile_path): os.makedirs(abs_profile_path)
            options.add_argument("-profile")
            options.add_argument(abs_profile_path)
        return webdriver.Firefox(options=options)
    
    elif browser_type in ["chrome", "chromium"]:
        browser_name = "Chrome" if browser_type == "chrome" else "Chromium"
        console.print(f"[yellow]🌐 Đang khởi động {browser_name}...[/yellow]")
        options = ChromeOptions()
        options.add_argument(f"--user-agent={mobile_ua}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-external-intents-redirect")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--ignore-certificate-errors")
        prefs = {"protocol_handler.excluded_schemes": {"intent": True}, "profile.default_content_setting_values.notifications": 2}
        options.add_experimental_option("prefs", prefs)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        if browser_type == "chromium":
            chromium_paths = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/snap/bin/chromium", "/usr/lib/chromium/chromium"]
            for path in chromium_paths:
                if os.path.exists(path):
                    options.binary_location = path
                    break
        
        if profile_path:
            abs_profile_path = os.path.abspath(profile_path)
            if not os.path.exists(abs_profile_path): os.makedirs(abs_profile_path)
            options.add_argument(f"--user-data-dir={abs_profile_path}")
        
        return webdriver.Chrome(options=options)
    else:
        raise ValueError(f"Browser không được hỗ trợ: {browser_type}")

def run_crawler(browser_type="firefox", target_urls=None, file_output="fb_data.json", min_posts=20, max_scrolls=15, scroll_pause=3, profile_path=None):
    if target_urls is None: target_urls = []
    init_json(file_output)
    
    driver = create_browser(browser_type, profile_path)
    
    try:
        console.print("[cyan]🌐 Đang vào Facebook...[/cyan]")
        driver.get("https://m.facebook.com")
        time.sleep(3)
        
        def is_logged_in():
            current_url = driver.current_url.lower()
            page_source = driver.page_source.lower()
            login_url_keywords = ["login", "checkpoint", "recover", "identify"]
            if any(kw in current_url for kw in login_url_keywords): return False
            login_indicators = ['name="email"', 'name="pass"', 'id="loginbutton"', 'data-sigil="login_button"']
            if any(indicator in page_source for indicator in login_indicators): return False
            return True
        
        if not is_logged_in():
            console.print("[bold yellow]⚠️ CHƯA ĐĂNG NHẬP![/bold yellow]")
            console.print("[yellow]👉 Hãy đăng nhập Facebook thủ công, sau đó quay lại đây và bấm Enter...[/yellow]")
            input()
            time.sleep(2)
        
        for url in target_urls:
            page_name_slug = url.split('/')[-1]
            console.print()
            console.rule(f"[bold blue]{page_name_slug}[/bold blue]")
            
            driver.get(url)
            time.sleep(5)
            
            # Gọi hàm process mới (Vừa scroll vừa lưu)
            process_page_scrolling(
                driver=driver,
                page_name_slug=page_name_slug,
                file_output=file_output,
                min_posts=min_posts,
                max_attempts=max_scrolls,
                scroll_pause_time=scroll_pause
            )

    except Exception as e:
        console.print(f"[bold red]❌ Lỗi Critical: {e}[/bold red]")
    finally:
        console.print()
        console.print(Panel.fit("[bold]🛑 Kết thúc session.[/bold]", border_style="red"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facebook Crawler")
    
    default_urls = [
        "https://m.facebook.com/Theanh28",
        "https://m.facebook.com/kienkhongngu.vn", 
        "https://m.facebook.com/thongtinchinhphu",
        "https://m.facebook.com/VnExpress",
        "https://m.facebook.com/baotuoitre",
        "https://m.facebook.com/thanhnien",
        "https://m.facebook.com/vietnamnet.vn",
        "https://m.facebook.com/baodantridientu",
        "https://m.facebook.com/laodongonline",
        "https://m.facebook.com/nhandanonline",
        "https://m.facebook.com/tintucvietnammoinong/",
        "https://www.facebook.com/tintucvtv24",
        "https://www.facebook.com/doisongvnn",
        "https://www.facebook.com/VnProCon",
        "https://www.facebook.com/tapchitrithucznews.vn",
    ]

    parser.add_argument("-b", "--browser", type=str, choices=["firefox", "chrome", "chromium"], default="firefox")
    parser.add_argument("-u", "--urls", type=str, nargs="+", default=default_urls)
    parser.add_argument("-o", "--output", type=str, default="fb_data.json")
    parser.add_argument("--min-posts", type=int, default=500)
    parser.add_argument("--max-scrolls", type=int, default=200)
    parser.add_argument("--scroll-pause", type=int, default=3)
    parser.add_argument("--profile-path", type=str, default="my_firefox_profile")
    
    args = parser.parse_args()

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