import os
import google.generativeai as genai
from typing import Dict, Any

class AIBettingAgent:
    """Agent that generates actionable betting insights using Gemini."""
    
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")
        
        genai.configure(api_key=key)
        
        # Using gemini-1.5-flash since we just need text generation
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def generate_insight(self, match_data: Dict[str, Any]) -> str:
        """
        Generates a betting insight based on match statistics and model probabilities.
        """
        prompt = f"""
        Você é um apostador esportivo de elite especializado em dados quantitativos.
        Você deve fornecer uma análise rica e palpites para o seguinte jogo:
        
        **Confronto:** {match_data['home_team']} (Mandante) vs {match_data['away_team']} (Visitante)
        **Maior Probabilidade Apontada pelo Modelo:** {match_data['outcome_label']}
        **Confiança do Modelo ML:** {match_data['model_prob'] * 100:.1f}%
        
        **Estatísticas (Médias Locais do Banco de Dados para os times na condição de mandante/visitante):**
        Mandante ({match_data['home_team']}):
        - Gols Marcados (Média): {match_data['features'].get('media_marcados_casa', 'N/A')}
        - Gols Sofridos (Média): {match_data['features'].get('media_sofridos_casa', 'N/A')}
        
        Visitante ({match_data['away_team']}):
        - Gols Marcados (Média): {match_data['features'].get('media_marcados_fora', 'N/A')}
        - Gols Sofridos (Média): {match_data['features'].get('media_sofridos_fora', 'N/A')}

        Baseado unicamente nestes dados estatísticos cruzados, construa um laudo conciso para o apostador cobrindo o seguinte ecossistema de apostas. Não enfeite, vá direto ao ponto e não divida a resposta com saudações. O formato obrigatório é o texto puro com os seguintes tópicos em Markdown bullet points:
        
        * **1X2 & Dupla Hipótese:** Um breve veredito sobre quem vence (ou se o jogo tende ao empate) e se uma Dupla Hipótese (Ex: Vitória Mandante ou Empate) faz mais sentido estratégico devido à falta de confiança para cravar o ganhador.
        * **Placar Exato:** Palpite frio do placar realista baseado nas médias dos gols.
        * **Mercado de Gols (Over/Under & BTTS):** Recomende expressamente um "Mais de" ou "Menos de" (Ex: Mais de 2.5 gols) ou "Ambas Marcam" baseando-se no potencial ofensivo e vazamento defensivo de ambos.
        * **Handicap Asiático:** Sugira um Handicap *se houver* amplo favoritismo e força de ataque em um dos lados (Ex: Time A -1.0). Se for equilibrado, a recomendação é 'Nenhuma indicação forçada'.
        * **Estatísticas Adicionais (Cantos/Cartões):** Dedutivamente, pela análise quantitativa do embate das médias, especule a propensão a Escanteios elevados (se os dois times atacam muito) ou Cartões (jogos enroscados).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Erro ao gerar insight: {str(e)}"
