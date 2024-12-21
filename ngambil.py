import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

def ngambil_youtube(url, num_comments, timeout=30):
    # Inisialisasi Chrome WebDriver dengan opsi headless
    options = Options()
    options.add_argument("--headless")  
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Path absolut ke driver
    service = Service(r"C:\Users\untu0\Downloads\chromedriver-win64 (2)\chromedriver-win64\chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    
    # Buka URL YouTube
    driver.get(url)
    time.sleep(2)
    
    # Scroll ke bawah untuk memuat komentar
    driver.execute_script("window.scrollTo(0, 600);")
    time.sleep(3)
    
    # Inisialisasi daftar komentar
    comments = set()
    actions = ActionChains(driver)
    start_time = time.time()
    
    while len(comments) < num_comments:
        # Periksa waktu timeout
        if time.time() - start_time > timeout:
            print(f"Timeout pada URL: {url}")
            break
        
        # Scroll ke bawah untuk memuat lebih banyak komentar
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)
        
        # Ambil elemen komentar yang sudah dimuat
        comment_elements = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
        
        # Menambah komentar baru ke dalam set untuk menghindari duplikasi
        initial_count = len(comments)
        for element in comment_elements[len(comments):]:
            comments.add(element.text)
        
        # Cek jika tidak ada komentar baru, hentikan pengulangan
        if len(comments) == initial_count:
            print(f"Tidak ada komentar baru pada URL: {url}")
            break

        # Cetak jumlah komentar yang sudah diambil sebagai log
        print(f"{len(comments)} komentar terkumpul untuk URL: {url}")
        
    # Menutup driver
    driver.quit()
    
    # Mengembalikan komentar dalam bentuk list hingga jumlah yang diminta
    return list(comments)[:num_comments]