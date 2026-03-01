import os
from typing import Dict, Any

class AIBettingAgent:
    """Agent that generates actionable betting insights using Deterministic Heuristics (Symbolic AI)."""
    
    def __init__(self, api_key: str = None):
        # API Key is kept for backward compatibility with main.py, but we no longer depend on Google GenAI API
        # preventing 429 Rate Limits and Deprecation errors completely.
        pass
        
    def generate_insight(self, match_data: Dict[str, Any]) -> str:
        """
        Gera um laudo de IA padronizado (Formato JSON fixo) usando o Gemini atualizado.
        """
        prompt = f"""
        Você é um Analista Quantitativo Sênior de Apostas Esportivas.
        Sua função é gerar um laudo analítico interpretando o choque entre o nosso modelo preditivo e a precificação do mercado para o seguinte jogo:
        
        **Confronto:** {match_data['home_team']} (Mandante) vs {match_data['away_team']} (Visitante)
        
        **Dados do Mercado vs Modelo:**
        - Aposta Sugerida pelo Sistema: {match_data['outcome_label']}
        - Probabilidade Real (Nosso Modelo): {match_data['model_prob'] * 100:.1f}%
        - Odd Oferecida pela Casa de Aposta: @{match_data.get('odds_taken', 0.0)} (Probabilidade Implícita: {match_data.get('implied_prob', 0.0) * 100:.1f}%)
        - Valor Esperado (EV): {match_data['ev_pct']}%
        
        **Estatísticas de Desempenho (Médias do Flashscore):**
        Mandante ({match_data['home_team']}):
        - Gols Marcados em Casa: {match_data['features'].get('media_marcados_casa', 'N/A')}
        - Gols Sofridos em Casa: {match_data['features'].get('media_sofridos_casa', 'N/A')}
        
        Visitante ({match_data['away_team']}):
        - Gols Marcados Fora: {match_data['features'].get('media_marcados_fora', 'N/A')}
        - Gols Sofridos Fora: {match_data['features'].get('media_sofridos_fora', 'N/A')}

        Forneça EXATAMENTE este formato JSON contendo 4 chaves textuais descrevendo o cenário:
        {{
            "analise_ev": "Explique brevemente por que o mercado está precificando essa odd de forma errada.",
            "dinamica": "Desenhe o roteiro provável do jogo.",
            "validacao": "A matemática valida a tip {match_data['outcome_label']} ou apoia zebra?",
            "mercado": "Indique tendência matemática para Over 2.5 ou Under 2.5."
        }}
        
        NÃO retorne formatação Markdown no início ou no fim do JSON (ex: ```json). Retorne apenas as chaves.
        """
        
        import time
        import json
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Erro: GEMINI_API_KEY ausente."
        
        # O Client mais simples utilizando a biblioteca que o Google recomendou
        client = genai.Client(api_key=api_key)
        
        max_retries = 5
        base_delay = 20 # Segundos para início do backoff

        for attempt in range(max_retries):
            try:
                print("Respeitando limite gratuito da IA (Aguardando 4.1s)...")
                time.sleep(4.1)
                
                # Exige retorno obrigatoriamente formatado na string JSON pedida acima
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                
                txt = response.text.strip()
                # Tratamento preventivo se o Gemini alucinar as tags Markdown para Json
                if txt.startswith("```json"):
                    txt = txt[7:-3]
                    
                data = json.loads(txt)
                
                # Renderiza o UI limpinho da string JSON
                laudo = (
                    f"* **Análise do Valor (EV):** {data.get('analise_ev', 'Não gerado.')}\n"
                    f"* **Dinâmica de Jogo Esperada:** {data.get('dinamica', 'Não gerado.')}\n"
                    f"* **Validação da Tip (1X2):** {data.get('validacao', 'Não gerado.')}\n"
                    f"* **Mercado Secundário de Gols (Over/Under):** {data.get('mercado', 'Não gerado.')}"
                )
                return laudo
                
            except json.JSONDecodeError:
                 return f"A IA não retornou o Padrão JSON esperado. Tente novamente."
            except Exception as e:
                err_str = str(e).lower()
                # Filtro pra Timeout nativo rate limit do genai error handling genérico
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        print(f"Gemini Rate Limit Excedido (429). Aguardando {sleep_time} segundos... (Tentativa {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        return f"API do Google Ocupada (Rate Limit Excedido). O laudo detalhado não pôde ser gerado para este jogo a tempo."
                else:    
                    return f"Erro ao gerar insight: {str(e)}"
        
        return "Erro desconhecido ao conversar com a IA do Google."
