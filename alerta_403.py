import requests
import time
import sys
import winsound

URL = "https://example.com"  # Altere para o endpoint desejado

while True:
    try:
        response = requests.get(URL)
        if response.status_code == 403:
            print("[ALERTA] Erro 403 detectado! IP possivelmente bloqueado.")
            winsound.Beep(1000, 1000)  # Alerta sonoro (Windows)
            # Pausa para ação manual
            print("Aguarde, troque o IP (modo avião no celular) e pressione Enter...")
            input()
        else:
            print(f"Status: {response.status_code}")
        time.sleep(5)  # Aguarda 5 segundos antes da próxima requisição
    except Exception as e:
        print(f"Erro: {e}")
        time.sleep(5)
