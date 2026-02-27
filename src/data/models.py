from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)  # Understat match ID
    league = Column(String)
    season = Column(Integer)
    date = Column(DateTime)
    home_team = Column(String)
    away_team = Column(String)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_xG = Column(Float)
    away_xG = Column(Float)
    is_result = Column(Boolean)

    # Stats de jogo (football-data.co.uk)
    home_shots = Column(Float)
    away_shots = Column(Float)
    home_shots_target = Column(Float)
    away_shots_target = Column(Float)

    # Odds -- Bet365
    odds_home_b365 = Column(Float)
    odds_draw_b365 = Column(Float)
    odds_away_b365 = Column(Float)

    # Odds -- Pinnacle (sharp / referencia)
    odds_home_pin = Column(Float)
    odds_draw_pin = Column(Float)
    odds_away_pin = Column(Float)

    # Odds -- Media do mercado
    odds_home_avg = Column(Float)
    odds_draw_avg = Column(Float)
    odds_away_avg = Column(Float)

    # Odds -- Maxima do mercado (best available)
    odds_home_max = Column(Float)
    odds_draw_max = Column(Float)
    odds_away_max = Column(Float)

    def __repr__(self):
        return f"<Match(id={self.id}, {self.home_team} vs {self.away_team}, Season={self.season})>"


class MatchFeatures(Base):
    """
    Armazena as Rolling Features (EWMA) calculadas por jogo.
    Relação 1-para-1 com a tabela matches via match_id.
    """
    __tablename__ = "match_features"

    match_id = Column(Integer, ForeignKey("matches.id"), primary_key=True)

    # EWMA do time mandante
    ewma5_xg_pro_home  = Column(Float)   # EWMA-5  xG produzido (ataque) — mandante
    ewma10_xg_pro_home = Column(Float)   # EWMA-10 xG produzido (ataque) — mandante
    ewma5_xg_con_home  = Column(Float)   # EWMA-5  xG concedido (defesa) — mandante
    ewma10_xg_con_home = Column(Float)   # EWMA-10 xG concedido (defesa) — mandante

    # EWMA do time visitante
    ewma5_xg_pro_away  = Column(Float)   # EWMA-5  xG produzido (ataque) — visitante
    ewma10_xg_pro_away = Column(Float)   # EWMA-10 xG produzido (ataque) — visitante
    ewma5_xg_con_away  = Column(Float)   # EWMA-5  xG concedido (defesa) — visitante
    ewma10_xg_con_away = Column(Float)   # EWMA-10 xG concedido (defesa) — visitante

    def __repr__(self):
        return f"<MatchFeatures(match_id={self.match_id})>"


def get_engine(db_path="sqlite:///understat_premier_league.db"):
    return create_engine(db_path)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
