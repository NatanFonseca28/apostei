from sqlalchemy import Column, Date, DateTime, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class FlashscoreMatch(Base):
    __tablename__ = "flashscore_matches"

    id = Column(String, primary_key=True)
    campeonato = Column(String)
    status = Column(String)
    data = Column(DateTime, nullable=True)  # Data da partida

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


class AIPredictionCache(Base):
    __tablename__ = "ai_predictions_cache"

    id = Column(String, primary_key=True)  # event_id + outcome ex: "bdab12f_H"
    event_id = Column(String)
    home_team = Column(String)
    away_team = Column(String)
    outcome = Column(String)
    insight_text = Column(String)
    created_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<AIPredictionCache({self.id})>"


class MatchAdvancedStats(Base):
    """
    Estatísticas avançadas por partida coletadas do FBref via soccerdata.
    Contém xG, chutes no alvo e posse de bola para cada jogo.

    A chave única é (home_team, away_team, date) — o mesmo critério usado
    no LEFT JOIN em load_dataset de trainer.py.
    """

    __tablename__ = "match_advanced_stats"
    __table_args__ = (UniqueConstraint("home_team", "away_team", "date", name="uq_match_adv"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    league = Column(String)
    season = Column(String)
    home_xg = Column(Float, default=0.0)
    away_xg = Column(Float, default=0.0)
    home_shots_target = Column(Float, default=0.0)
    away_shots_target = Column(Float, default=0.0)
    home_possession = Column(Float, default=0.0)
    away_possession = Column(Float, default=0.0)
    source = Column(String, default="fbref")

    def __repr__(self):
        return f"<MatchAdvancedStats({self.home_team} vs {self.away_team} | {self.date})>"


def get_engine(db_path="sqlite:///flashscore_data.db"):
    return create_engine(db_path)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
