import logging
from src.data.extractor import FBrefReader

logging.basicConfig(level=logging.DEBUG)
reader = FBrefReader()
try:
    df = reader.read_season_fixtures('24', 'Serie-A', 2023)
    print("DataFrame shape:", df.shape)
    print(df.head())
finally:
    reader.close()
