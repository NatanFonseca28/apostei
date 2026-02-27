import sys, os

# Adiciona o diretório raiz ao path para importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
