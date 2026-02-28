import logging
import time
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

options = uc.ChromeOptions()
options.add_argument('--headless=new')
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options)

try:
    url = "https://fbref.com/en/comps/24/2023/schedule/2023-Serie-A-Scores-and-Fixtures"
    print("Navigating to:", url)
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    with open('fbref_debug.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("HTML saved to fbref_debug.html")

    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'stats_table'})
    if not table:
        print("TABLE NOT FOUND. Looking for current season fallback.")
        url = "https://fbref.com/en/comps/24/schedule/Serie-A-Scores-and-Fixtures"
        print("Navigating to fallback:", url)
        driver.get(url)
        time.sleep(5)
        html = driver.page_source
        with open('fbref_debug.html', 'w', encoding='utf-8') as f:
            f.write(html)
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', {'class': 'stats_table'})
    
    if table:
        print("Table found:", len(table.find_all('tr')), "rows.")
        tbody = table.find('tbody')
        rows = tbody.find_all('tr') if tbody else table.find_all('tr')
        valid_rows = 0
        for r in rows:
            home = r.find('td', {'data-stat': 'home_team'})
            if home and home.text.strip():
                valid_rows += 1
        print("Valid home_team rows:", valid_rows)
    else:
        print("Still no table.")
except Exception as e:
    print("Error:", e)
finally:
    try:
        driver.quit()
    except Exception as e:
        print("Quit error:", e)
