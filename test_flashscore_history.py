import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1920,1080")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = "https://www.flashscore.com.br/futebol/espanha/laliga/resultados/"
print("Acessando:", url)
driver.get(url)
time.sleep(5)

# Tentar clicar em Mostrar mais jogos
clicks = 0
while True:
    try:
        btn = driver.find_element(By.CSS_SELECTOR, "a.event__more--static")
        if btn.is_displayed():
            driver.execute_script("arguments[0].click();", btn)
            clicks += 1
            print(f"Clicou 'Mostrar mais jogos' ({clicks})")
            time.sleep(3)
        else:
            break
    except Exception as e:
        print("Botão não encontrado ou não clicável.")
        break

html_source = driver.page_source
soup = BeautifulSoup(html_source, 'html.parser')

partidas = soup.find_all("div", class_="event__match")
print(f"Total de partidas antigas carregadas: {len(partidas)}")

driver.quit()
