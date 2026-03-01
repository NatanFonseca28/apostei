import logging
import re
import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

# ─── Limpeza de nomes de times ────────────────────────────────────────────────

_TEAM_SUFFIX_RE = re.compile(
    r"\s*(Avança na competição|Eliminado|Rebaixado|Promovido|Despromovido)\s*$",
    re.IGNORECASE,
)
_TEAM_DIGIT_SUFFIX_RE = re.compile(r"\d+\s*$")


def _clean_team_name(name: str) -> str:
    """Remove sufixos contextuais e números de time reserva adicionados pelo Flashscore."""
    name = _TEAM_SUFFIX_RE.sub("", name)
    name = _TEAM_DIGIT_SUFFIX_RE.sub("", name)
    return name.strip()


# ============================================================================
# LÓGICA DO USUÁRIO
# ============================================================================

PESO = {
    "Muito alta ofensividade": 5,
    "Alta ofensividade": 4,
    "Média ofensividade": 3,
    "Baixa ofensividade": 2,
    "Muito baixa ofensividade": 1,
    "Muito alta defensividade": 1,
    "Alta defensividade": 2,
    "Média defensividade": 3,
    "Baixa defensividade": 4,
    "Muito baixa defensividade": 5,
}


def classificar_ofensividade(media_gols):
    if media_gols > 2.5:
        return "Muito alta ofensividade"
    elif 2 < media_gols <= 2.5:
        return "Alta ofensividade"
    elif 1.5 < media_gols <= 2:
        return "Média ofensividade"
    elif 1 < media_gols <= 1.5:
        return "Baixa ofensividade"
    else:
        return "Muito baixa ofensividade"


def classificar_defensividade(media_gols):
    if media_gols > 2.5:
        return "Muito baixa defensividade"
    elif 2 < media_gols <= 2.5:
        return "Baixa defensividade"
    elif 1.5 < media_gols <= 2:
        return "Média defensividade"
    elif 1 < media_gols <= 1.5:
        return "Alta defensividade"
    else:
        return "Muito alta defensividade"


def classificar_probabilidade_gol(of_casa, df_fora, of_fora, df_casa):
    ataque_casa = PESO[of_casa] - PESO[df_fora]
    ataque_fora = PESO[of_fora] - PESO[df_casa]
    score_total = ataque_casa + ataque_fora

    if score_total >= 7:
        return "Muito Alta"
    elif score_total >= 5:
        return "Alta"
    elif score_total >= 3:
        return "Média"
    elif score_total >= 1:
        return "Baixa"
    else:
        return "Muito Baixa"


def determinar_melhor_chance(ataque_casa, defesa_fora, ataque_fora, defesa_casa):
    score_casa = PESO[ataque_casa] - PESO[defesa_fora]
    score_fora = PESO[ataque_fora] - PESO[defesa_casa]

    if score_casa > score_fora:
        return "Casa"
    elif score_fora > score_casa:
        return "Fora"
    else:
        return "Equilibrado"


# ============================================================================
# EXTRATOR
# ============================================================================


class FlashscoreExtractor:
    """Extrai dados e avaliações probabilísticas do Flashscore."""

    def __init__(self):
        self.driver = None

    def _init_driver(self):
        if not self.driver:
            logger.info("Inicializando Chrome (Headless)...")
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.debug(f"Erro ao fechar webdriver: {e}")
            finally:
                self.driver = None

    def fetch_data(self, urls: list[str]) -> pd.DataFrame:
        """Processa as URLs do Flashscore para buscar a tabela, jogos do dia e historico."""
        self._init_driver()
        data = []
        data_hoje = datetime.now().strftime("%d.%m.")

        for url in urls:
            logger.info(f"Processando: {url}")
            try:
                # 1. Obter Tabela Atual e Campeonato
                self.driver.get(url)
                time.sleep(5)
                html_source = self.driver.page_source
                soup = BeautifulSoup(html_source, "html.parser")

                try:
                    campeonato = soup.find("div", class_="heading__name").text.strip()
                except:
                    campeonato = "Desconhecido"

                # Extrair estatisticas da tabela "ui-table"
                estatisticas = {}
                try:
                    tabela = self.driver.find_element(By.CLASS_NAME, "ui-table__body")
                    linhas = tabela.find_elements(By.CLASS_NAME, "ui-table__row")
                    for linha in linhas:
                        try:
                            nome_time = linha.find_element(By.CLASS_NAME, "tableCellParticipant__name").text.strip()
                            colunas = linha.find_elements(By.CLASS_NAME, "table__cell")
                            if len(colunas) > 6:
                                jogos_disputados = colunas[2].text.strip()
                                gols_info = colunas[6].text.strip()
                                estatisticas[nome_time] = (jogos_disputados, gols_info)
                        except:
                            continue
                except Exception as e:
                    logger.warning(f"Tabela não encontrada para {campeonato}: {e}")

                # 2. Obter Historico (tudo de "resultados")
                base_url = url.split("#")[0].rstrip("/")
                hist_url = f"{base_url}/resultados/"
                logger.info(f"Acessando histórico: {hist_url}")
                self.driver.get(hist_url)
                time.sleep(5)

                # Clicar 'Mostrar mais jogos'
                while True:
                    try:
                        btn = self.driver.find_element(By.CSS_SELECTOR, "a.event__more--static")
                        if btn.is_displayed():
                            self.driver.execute_script("arguments[0].click();", btn)
                            time.sleep(3)
                        else:
                            break
                    except:
                        break

                hist_soup = BeautifulSoup(self.driver.page_source, "html.parser")
                div_hist = hist_soup.find("div", class_="sportName soccer")
                partidas_hist = div_hist.find_all("div", class_="event__match") if div_hist else []
                logger.info(f"Encontradas {len(partidas_hist)} partidas históricas para {campeonato}")

                # 3. Obter Jogos do Dia (da página principal q a gente ja extraiu no comeco ou calendario)
                # O script original lia da page principal
                self.driver.get(url)
                time.sleep(5)
                hoje_soup = BeautifulSoup(self.driver.page_source, "html.parser")
                div_hoje = hoje_soup.find("div", class_="sportName soccer")
                partidas_hoje = div_hoje.find_all("div", class_="event__match") if div_hoje else []

                # Consolida todas partidas (historico + hoje) para analise
                todas_partidas = partidas_hist + partidas_hoje

                # Evitar duplicadas pelo match id (div id="g_1_XXXX") se existir
                processadas = set()

                for partida in todas_partidas:
                    match_id = partida.get("id", str(hash(partida.text)))
                    if match_id in processadas:
                        continue
                    processadas.add(match_id)

                    status = "Indisponível"
                    to_start = partida.find("div", class_="event__time")
                    started = partida.find("div", class_="event__stage--block")

                    if to_start:
                        status = to_start.get_text().strip()
                    elif started:
                        status = started.get_text().strip()

                    # Extrair os times
                    try:
                        time_casa_elem = partida.find("div", class_=lambda x: x and "homeParticipant" in x)
                        time_fora_elem = partida.find("div", class_=lambda x: x and "awayParticipant" in x)
                        time_casa = _clean_team_name(time_casa_elem.text.strip())
                        time_fora = _clean_team_name(time_fora_elem.text.strip())
                    except Exception as e:
                        logger.debug(f"Falha ao encontrar times: {e}")
                        continue

                    placar_casa = None
                    placar_fora = None
                    try:
                        score_home = partida.find("span", class_=lambda x: x and "score--home" in x)
                        score_away = partida.find("span", class_=lambda x: x and "score--away" in x)
                        if score_home and score_away:
                            placar_casa = int(score_home.text.strip())
                            placar_fora = int(score_away.text.strip())
                    except:
                        pass

                    # Cruzando com tabela de classificacao
                    jogos_casa_raw, gols_casa_raw = estatisticas.get(time_casa, ("N/A", "N/A"))
                    jogos_fora_raw, gols_fora_raw = estatisticas.get(time_fora, ("N/A", "N/A"))

                    try:
                        jogos_casa = int(jogos_casa_raw) if jogos_casa_raw != "N/A" else 0
                    except:
                        jogos_casa = 0

                    try:
                        jogos_fora = int(jogos_fora_raw) if jogos_fora_raw != "N/A" else 0
                    except:
                        jogos_fora = 0

                    try:
                        gols_marcados_casa, gols_sofridos_casa = map(int, gols_casa_raw.split(":"))
                    except:
                        gols_marcados_casa, gols_sofridos_casa = 0, 0

                    try:
                        gols_marcados_fora, gols_sofridos_fora = map(int, gols_fora_raw.split(":"))
                    except:
                        gols_marcados_fora, gols_sofridos_fora = 0, 0

                    media_gols_marcados_casa = round(gols_marcados_casa / jogos_casa, 2) if jogos_casa > 0 else 0.0
                    media_gols_sofridos_casa = round(gols_sofridos_casa / jogos_casa, 2) if jogos_casa > 0 else 0.0
                    media_gols_marcados_fora = round(gols_marcados_fora / jogos_fora, 2) if jogos_fora > 0 else 0.0
                    media_gols_sofridos_fora = round(gols_sofridos_fora / jogos_fora, 2) if jogos_fora > 0 else 0.0

                    class_marcados_casa = classificar_ofensividade(media_gols_marcados_casa)
                    class_sofridos_casa = classificar_defensividade(media_gols_sofridos_casa)
                    class_marcados_fora = classificar_ofensividade(media_gols_marcados_fora)
                    class_sofridos_fora = classificar_defensividade(media_gols_sofridos_fora)

                    probabilidade_gol = classificar_probabilidade_gol(class_marcados_casa, class_sofridos_fora, class_marcados_fora, class_sofridos_casa)

                    melhor_chance = determinar_melhor_chance(class_marcados_casa, class_sofridos_fora, class_marcados_fora, class_sofridos_casa)

                    # Data (tratando timestamp do status)
                    dt = None
                    import re

                    match_dt = re.search(r"(\d{2})\.(\d{2})\.\s+(\d{2}:\d{2})", status)
                    if match_dt:
                        try:
                            day, month, time_str = match_dt.groups()
                            year = datetime.now().year
                            dt = datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M")
                        except Exception as e:
                            logger.debug(f"Erro no parsing de data {status}: {e}")
                    elif data_hoje in status:
                        dt = datetime.now()

                    data.append(
                        {
                            "id": match_id,
                            "campeonato": campeonato,
                            "status": status,
                            "data": dt,
                            "placar_casa": placar_casa,
                            "placar_fora": placar_fora,
                            "time_casa": time_casa,
                            "jogos_casa": jogos_casa,
                            "gols_marcados_casa": gols_marcados_casa,
                            "gols_sofridos_casa": gols_sofridos_casa,
                            "media_marcados_casa": media_gols_marcados_casa,
                            "media_sofridos_casa": media_gols_sofridos_casa,
                            "time_fora": time_fora,
                            "jogos_fora": jogos_fora,
                            "gols_marcados_fora": gols_marcados_fora,
                            "gols_sofridos_fora": gols_sofridos_fora,
                            "media_marcados_fora": media_gols_marcados_fora,
                            "media_sofridos_fora": media_gols_sofridos_fora,
                            "ofensividade_casa": class_marcados_casa,
                            "defensividade_casa": class_sofridos_casa,
                            "ofensividade_fora": class_marcados_fora,
                            "defensividade_fora": class_sofridos_fora,
                            "probabilidade_gol": probabilidade_gol,
                            "melhor_chance": melhor_chance,
                        }
                    )

            except Exception as e:
                logger.error(f"Erro ao processar a liga {url}: {e}")

        self.close()

        df = pd.DataFrame(data)
        logger.info(f"Extracao completa: {len(df)} partidas processadas.")
        return df
