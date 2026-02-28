import undetected_chromedriver as uc
import time
from bs4 import BeautifulSoup

try:
    print("Iniciando Chrome...")
    options = uc.ChromeOptions()
    # options.add_argument('--headless=new')
    driver = uc.Chrome(options=options)
    url = "https://fbref.com/en/comps/24/2023/schedule/2023-Serie-A-Scores-and-Fixtures"
    print("Navigating to:", url)
    driver.get(url)
    time.sleep(5)
    print("Page title:", driver.title)
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'stats_table'})
    if table:
        print("Table found!")
    else:
        print("Table NOT found! Title:", driver.title)
        
    driver.quit()
    print("Successfully closed.")
except Exception as e:
    print("Erro critico:", e)
