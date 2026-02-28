import os
import time
from datetime import datetime
import pandas as pd
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

url = "https://www.flashscore.com.br/futebol/espanha/laliga/#/dINOZk9Q/table/overall"
print("Acessando:", url)
driver.get(url)
time.sleep(5)

html_source = driver.page_source
soup = BeautifulSoup(html_source, 'html.parser')

try:
    campeonato = soup.find("div", class_="heading__name").text.strip()
except:
    campeonato = "Desconhecido"
print("Campeonato:", campeonato)

# Tabela
try:
    tabela = driver.find_element(By.CLASS_NAME, "ui-table__body")
    linhas = tabela.find_elements(By.CLASS_NAME, "ui-table__row")
    print(f"Encontradas {len(linhas)} linhas na tabela de classificação.")
except Exception as e:
    print("Erro na tabela:", e)

# Partidas
div = soup.find("div", class_="sportName soccer")
partidas = div.find_all("div", class_="event__match") if div else []
print(f"Encontradas {len(partidas)} partidas na página principal.")

driver.quit()
