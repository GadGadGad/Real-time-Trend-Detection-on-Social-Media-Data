import time
import os
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

# --- CẤU HÌNH ---
TARGET_URL = "https://m.facebook.com/Theanh28" # Page muốn kiểm tra
PROFILE_PATH = "my_firefox_profile"
OUTPUT_HTML_FILE = "debug_facebook_source.html"

def expand_details(driver):
    """Thử click vào các nút xem thêm để bung HTML ra"""
    try:
        btns = driver.find_elements(By.XPATH, "//span[contains(text(), 'Xem thêm') or text()='…']")
        for btn in btns:
            try: driver.execute_script("arguments[0].click();", btn)
            except: pass
        time.sleep(2)
    except: pass

def capture_html():
    print("🕵️ Đang khởi động 'Thám tử HTML'...")
    
    options = Options()
    mobile_ua = "Mozilla/5.0 (Linux; Android 11; SAMSUNG SM-G973U) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/14.2 Chrome/87.0.4280.141 Mobile Safari/537.36"
    options.set_preference("general.useragent.override", mobile_ua)
    
    abs_profile_path = os.path.abspath(PROFILE_PATH)
    if os.path.exists(abs_profile_path):
        options.add_argument("-profile")
        options.add_argument(abs_profile_path)
    
    driver = webdriver.Firefox(options=options)
    
    try:
        print(f"🌐 Truy cập: {TARGET_URL}")
        driver.get(TARGET_URL)
        time.sleep(5)
        
        # 1. Cuộn nhẹ vài lần để load bài viết thật
        print("⬇️ Đang cuộn trang để kích hoạt Javascript...")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
        # 2. Bung các chi tiết (nếu cần)
        expand_details(driver)
        
        # 3. LẤY TOÀN BỘ HTML ĐANG HIỂN THỊ
        print("📸 Đang chụp lại toàn bộ mã HTML...")
        full_html = driver.page_source
        
        # 4. Lưu ra file
        with open(OUTPUT_HTML_FILE, "w", encoding="utf-8") as f:
            f.write(full_html)
            
        print(f"✅ Đã lưu xong! File nằm tại: {os.path.abspath(OUTPUT_HTML_FILE)}")
        print("👉 Bạn hãy mở file này bằng Notepad/VSCode, tìm đoạn chứa 'bài viết' và gửi cho tôi.")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    capture_html()