import logging
import random
import time

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


def fetch_fbref_page_stealth(url: str, min_delay: float = 4.0, max_delay: float = 10.0) -> str:
    """
    Acessa uma URL do FBref emulando um navegador real no notebook local
    para contornar bloqueios WAF e proteções anti-bot.

    Retorna o HTML completo da página renderizada.
    """
    options = uc.ChromeOptions()
    # Oculta a janela para não atrapalhar seu uso do notebook,
    # mas mantém a renderização do motor Blink ativa.
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        logger.info(f"Iniciando navegador furtivo para: {url}")
        driver = uc.Chrome(options=options)

        # Pausa humana antes da requisição
        time.sleep(random.uniform(1.0, 3.0))

        driver.get(url)

        # Aguarda até que uma tabela de estatísticas básica seja carregada no DOM
        # Isso garante que a página passou pelos checks de Cloudflare/JS
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "table")))

        # Jitter de leitura (simula o tempo que um humano passa na página)
        read_time = random.uniform(min_delay, max_delay)
        logger.debug(f"Página carregada. Simulando leitura por {read_time:.2f} segundos...")
        time.sleep(read_time)

        html_content = driver.page_source
        return html_content

    except Exception as e:
        logger.error(f"Falha ao extrair a página {url} via emulação: {e}")
        return ""
    finally:
        if driver:
            driver.quit()
