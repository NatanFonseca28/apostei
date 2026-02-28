from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class FlashscoreMatch(Base):
    __tablename__ = "flashscore_matches"

    id = Column(String, primary_key=True) 
    campeonato = Column(String)
    status = Column(String)
    data = Column(DateTime, nullable=True) # Data da partida
    
    placar_casa = Column(Integer, nullable=True)
    placar_fora = Column(Integer, nullable=True)
    
    time_casa = Column(String)
    jogos_casa = Column(Integer)
    gols_marcados_casa = Column(Integer)
    gols_sofridos_casa = Column(Integer)
    media_marcados_casa = Column(Float)
    media_sofridos_casa = Column(Float)
    
    time_fora = Column(String)
    jogos_fora = Column(Integer)
    gols_marcados_fora = Column(Integer)
    gols_sofridos_fora = Column(Integer)
    media_marcados_fora = Column(Float)
    media_sofridos_fora = Column(Float)
    
    ofensividade_casa = Column(String)
    defensividade_casa = Column(String)
    ofensividade_fora = Column(String)
    defensividade_fora = Column(String)
    
    probabilidade_gol = Column(String)
    melhor_chance = Column(String)
    
    def __repr__(self):
        return f"<FlashscoreMatch({self.time_casa} vs {self.time_fora})>"

def get_engine(db_path="sqlite:///flashscore_data.db"):
    return create_engine(db_path)

def create_tables(engine):
    Base.metadata.create_all(engine)

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
