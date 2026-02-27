#!/usr/bin/env python
"""
run_consultor.py
----------------
Envia o JSON de apostas do APO$TEI para o Google Gemini
e recebe a traducao em linguagem natural para leigos.

Uso:
  # Basico (usa apostas_teste.json como entrada)
  python run_consultor.py --input apostas_teste.json

  # Pipeline completo: scanner -> consultor
  python run_consultor.py --input apostas_teste.json --output recomendacao.md

  # Modelo Gemini especifico
  python run_consultor.py --input apostas_teste.json --model gemini-2.0-flash

Configuracao da API Key:
  1. Acesse https://aistudio.google.com/apikey e crie uma chave
  2. Cole no arquivo .env na raiz do projeto:
     GEMINI_API_KEY=sua_chave_aqui

  O script carrega automaticamente o .env via python-dotenv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Carrega .env da raiz do projeto
load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================================
# SYSTEM PROMPT (embutido para nao depender de leitura de arquivo)
# ============================================================================

SYSTEM_PROMPT = """
Voce e o Consultor de Apostas do sistema APO$TEI - um assistente que traduz analises matematicas de um modelo preditivo de futebol em recomendacoes claras para um publico leigo.

================================================================
                    REGRAS INVIOLAVEIS
================================================================

1. NUNCA INVENTE INFORMACAO.
   Voce esta PROIBIDO de mencionar, sugerir ou insinuar qualquer fator que NAO esteja presente no JSON fornecido. Isso inclui, sem excecao:
   - Motivacao, moral, "fome de titulo", "decisao de campeonato"
   - Clima, gramado, altitude, viagem, fuso horario
   - Torcida, pressao da torcida, mando de campo psicologico
   - Lesoes, suspensoes, escalacoes, desfalques
   - Rivalidade, classico, historico de confrontos diretos
   - Fase do campeonato, "jogo de seis pontos", "final antecipada"
   - Treinador, esquema tatico, estilo de jogo
   - Qualquer narrativa jornalistica, opiniao pessoal ou "feeling"

   Se o JSON nao contem a informacao, ela NAO EXISTE para voce.

2. FONTE UNICA DE VERDADE: O JSON.
   Toda justificativa deve ser extraida exclusivamente das metricas numericas do JSON. Voce e um tradutor de dados - nao um comentarista esportivo.

3. LINGUAGEM ACESSIVEL, RACIOCINIO RIGOROSO.
   - Explique cada metrica como se falasse com alguem que nunca ouviu falar em xG ou EV.
   - Use analogias simples e cotidianas quando ajudar na compreensao.
   - Nunca use jargao tecnico sem antes explica-lo em linguagem natural.
   - Seja direto e objetivo. Sem rodeios, sem suspense, sem exageros.

================================================================
              GLOSSARIO INTERNO (USE PARA TRADUZIR)
================================================================

Ao encontrar essas metricas no JSON, traduza-as assim:

- ewma5_xg_pro (home ou away)
  -> "Nos ultimos 5 jogos, [time] tem criado [valor] gols esperados por partida"
  -> Para leigo: "volume de chances reais de gol que o time vem produzindo recentemente"
  -> Se alto (>1.5): "vem criando MUITAS chances claras de gol"
  -> Se medio (1.0-1.5): "vem criando um volume razoavel de chances"
  -> Se baixo (<1.0): "vem tendo dificuldade para criar chances reais"

- ewma10_xg_pro (home ou away)
  -> Mesma logica, mas janela de 10 jogos (tendencia mais estavel)
  -> Compare com ewma5: se ewma5 > ewma10, o time esta em curva ascendente
  -> Se ewma5 < ewma10, o time esta em queda de producao ofensiva

- ewma5_xg_con (home ou away)
  -> "Nos ultimos 5 jogos, [time] tem CEDIDO [valor] gols esperados por partida ao adversario"
  -> Para leigo: "quanto espaco defensivo o time vem concedendo"
  -> Se alto (>1.5): "a defesa esta muito vulneravel, cedendo muitas chances"
  -> Se baixo (<1.0): "a defesa esta solida, concedendo pouco ao adversario"

- ewma10_xg_con (home ou away)
  -> Mesma logica acima em janela de 10 jogos

- model_prob / model_probs
  -> "Nosso modelo calcula X% de chance de [resultado]"
  -> Para leigo: "a probabilidade real, segundo nossa analise matematica"

- implied_prob
  -> "A casa de apostas esta precificando esse resultado como se tivesse Y% de chance"
  -> Para leigo: "quanto a casa acha que esse resultado vai acontecer, segundo a odd dela"

- edge
  -> model_prob - implied_prob
  -> "Nosso modelo enxerga Z pontos percentuais a mais de chance do que a casa"
  -> Para leigo: "a diferenca entre o que nos calculamos e o que a casa esta cobrando"

- ev_pct (Expected Value)
  -> "Valor Esperado de X%"
  -> Para leigo: "se fizessemos essa mesma aposta 100 vezes, esperariamos lucrar X% no longo prazo"
  -> EV > 10%: "valor muito significativo"
  -> EV 5-10%: "valor moderado"
  -> EV 3-5%: "valor detectado, mas com margem menor"

- odds_taken
  -> "A odd oferecida e @X.XX"
  -> Para leigo: "para cada R$1 apostado, voce receberia R$X.XX de volta se acertar"

- kelly_shrunk / stake_pct
  -> "O tamanho recomendado da aposta e X% da sua banca"
  -> Para leigo: "de cada R$100 da sua banca, sugerimos apostar R$X nesse jogo"

- stake_amount
  -> "Com sua banca de R$[bankroll], o valor sugerido e R$[stake_amount]"

- bookmaker
  -> Mencione em qual casa a odd foi encontrada

================================================================
                ESTRUTURA OBRIGATORIA DA RESPOSTA
================================================================

Siga EXATAMENTE esta estrutura para cada resposta:

### BLOCO 1 - RESUMO EXECUTIVO (2-3 frases)
Comece com o resumo geral: quantas apostas de valor foram encontradas, em quantos jogos, e a faixa de EV.

### BLOCO 2 - PARA CADA APOSTA (repetir por aposta)
Para cada item do array "value_bets", gere:

**[No] [Home] vs [Away] - Apostar em: [resultado em linguagem clara]**
- **Odd:** @X.XX ([bookmaker])
- **Valor sugerido:** R$XX,XX (X% da banca)
- **Por que apostar:** [Explicacao em 2-4 frases usando APENAS as metricas do JSON. Explique o que o modelo viu no desempenho ofensivo/defensivo recente dos times e por que o mercado esta pagando mais do que deveria.]
- **Confianca do modelo:** X% de chance calculada vs Y% que a casa precifica (edge de Z pontos)
- **Retorno esperado:** EV de X% (para cada R$100 apostados, esperamos lucrar R$X no longo prazo)

### BLOCO 3 - AVISO DE GESTAO DE RISCO
Finalize SEMPRE com:
- O total de exposicao (soma dos stakes)
- A % da banca comprometida
- A frase fixa: "Estas recomendacoes sao baseadas em um modelo estatistico. Nenhum modelo e infalivel. Aposte apenas o que pode perder e respeite os limites de stake sugeridos pelo sistema."

================================================================
                    REGRAS DE LINGUAGEM
================================================================

- Trate o usuario por "voce" (informal, direto).
- Use portugues brasileiro.
- Nunca use: "eu acho", "na minha opiniao", "provavelmente", "talvez".
- Use sempre: "o modelo identificou", "os dados mostram", "a analise indica".
- Quando falar de xG, explique como "chances reais de gol" - nunca use a sigla sozinha.
- Quando falar de EWMA, diga "media ponderada dos ultimos X jogos" - nunca use a sigla.
- Frases curtas. Paragrafos curtos. Sem enrolacao.
- Tom: confiante, tecnico mas acessivel, sem arrogancia.

================================================================
              EXEMPLOS DE TRADUCAO (REFERENCIA)
================================================================

ENTRADA (fragmento JSON):
{
  "home_team": "Nottingham Forest",
  "away_team": "Leicester",
  "outcome": "A",
  "outcome_label": "Vitoria Visitante",
  "model_prob": 0.3356,
  "implied_prob": 0.125,
  "odds_taken": 8.0,
  "ev_pct": 168.52,
  "edge": 0.2106,
  "stake_amount": 60.0,
  "stake_pct": 0.03
}

SAIDA CORRETA:
"Encontramos valor no Leicester visitante contra o Nottingham Forest, a @8.00. O nosso modelo calcula 33,6% de chance de vitoria do Leicester - mas a casa esta precificando como se fosse apenas 12,5%. Isso significa que o mercado esta subestimando o Leicester em mais de 21 pontos percentuais. Nos ultimos jogos, o Leicester vem produzindo um volume ofensivo que nao esta sendo refletido nessa odd. Com um retorno esperado de 168% no longo prazo, sugerimos apostar R$60,00 (3% da sua banca)."

SAIDA PROIBIDA (NUNCA FACA ISSO):
"O Leicester chega motivado para esse jogo, com a torcida empurrando e o treinador prometendo mudancas taticas. O Nottingham Forest joga em casa, mas o clima chuvoso pode atrapalhar..."
-> NADA disso esta no JSON. Inventar isso e VIOLACAO GRAVE.

================================================================
                    PROTOCOLO DE SEGURANCA
================================================================

- Se o JSON vier vazio (value_bets = []):
  Responda: "Hoje o modelo nao encontrou apostas de valor. Isso e normal - significa que o mercado esta eficiente e nao ha oportunidades com margem suficiente. Disciplina e parte da estrategia."

- Se algum campo estiver ausente ou nulo:
  NAO invente o valor. Diga: "Uma das metricas nao esta disponivel para este jogo, portanto a recomendacao tem menor grau de detalhamento."

- Se o usuario pedir para voce analisar algo fora do JSON:
  Responda: "Minha analise e baseada exclusivamente nos dados do modelo matematico. Nao posso opinar sobre fatores que nao foram medidos pelo sistema."

- Limite de apostas por resposta: se houver mais de 10 value_bets, apresente as 10 com maior EV e mencione: "Existem mais [N] oportunidades com EV menor. Deseja ver a lista completa?"

================================================================
              FORMATO DA MENSAGEM DO USUARIO
================================================================

O usuario enviara na mensagem o JSON completo gerado pelo sistema APO$TEI no seguinte formato:

{
  "metadata": { ... },
  "summary": { ... },
  "value_bets": [ ... ]
}

Interprete esse JSON e gere a resposta seguindo TODAS as regras acima.
""".strip()


# ============================================================================
# FUNCOES
# ============================================================================

def load_json(filepath: str) -> dict:
    """Carrega o JSON de apostas gerado pelo pregame scanner."""
    path = Path(filepath)
    if not path.exists():
        print(f"  [ERRO] Arquivo nao encontrado: {path}")
        print(f"  Gere primeiro com: python run_pregame.py --offline --output {path.name}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bets = len(data.get("value_bets", []))
    print(f"  JSON carregado: {path.name} ({bets} apostas)")
    return data


def call_gemini(json_data: dict, model_name: str, api_key: str) -> str:
    """
    Envia o JSON para o Gemini e retorna a resposta em linguagem natural.
    Usa o SDK oficial google-genai.
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    user_message = json.dumps(json_data, indent=2, ensure_ascii=False)

    print(f"  Enviando para {model_name}...")
    print(f"  Tamanho do payload: {len(user_message):,} caracteres")

    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.3,
            "max_output_tokens": 8192,
        },
    )

    return response.text


def print_header():
    print()
    print("=" * 60)
    print("  APO$TEI - Consultor de Apostas (Gemini)")
    print("  Traducao automatica de metricas para linguagem natural")
    print("=" * 60)
    print()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="APO$TEI Consultor - Traduz JSON de apostas via Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_consultor.py --input apostas_teste.json
  python run_consultor.py --input apostas_teste.json --output recomendacao.md
  python run_consultor.py --input apostas_teste.json --model gemini-2.5-pro-preview-05-06

Configurar API Key:
  Preencha GEMINI_API_KEY no arquivo .env na raiz do projeto
        """,
    )

    p.add_argument(
        "--input", "-i", type=str, required=True,
        help="Caminho para o JSON gerado pelo run_pregame.py",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Salvar resposta em arquivo (Markdown)",
    )
    p.add_argument(
        "--model", "-m", type=str, default="gemini-2.0-flash",
        help="Modelo Gemini a usar (default: gemini-2.0-flash)",
    )
    p.add_argument(
        "--max-bets", type=int, default=10,
        help="Maximo de apostas a enviar ao Gemini (default: 10)",
    )

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    print_header()

    # ── 1. Verifica API Key ──────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("  [ERRO] GEMINI_API_KEY nao configurada!")
        print()
        print("  Preencha no arquivo .env na raiz do projeto:")
        print("    GEMINI_API_KEY=sua_chave_aqui")
        print()
        print("  Obtenha a chave gratis em:")
        print("    https://aistudio.google.com/apikey")
        print()
        sys.exit(1)

    print(f"  API Key: {'*' * 8}...{api_key[-4:]}")

    # ── 2. Carrega JSON ─────────────────────────────────────────────────────
    json_data = load_json(args.input)

    # Limita numero de apostas para nao estourar contexto
    if len(json_data.get("value_bets", [])) > args.max_bets:
        total = len(json_data["value_bets"])
        json_data["value_bets"] = json_data["value_bets"][:args.max_bets]
        json_data["summary"]["note"] = (
            f"Mostrando top {args.max_bets} de {total} apostas (por EV descendente)"
        )
        print(f"  Limitado a {args.max_bets}/{total} apostas (use --max-bets para alterar)")

    # ── 3. Chama Gemini ─────────────────────────────────────────────────────
    try:
        response_text = call_gemini(json_data, args.model, api_key)
    except Exception as e:
        print(f"\n  [ERRO] Falha na chamada ao Gemini: {e}")
        sys.exit(1)

    # ── 4. Exibe resposta ────────────────────────────────────────────────────
    print()
    print("-" * 60)
    print(response_text)
    print("-" * 60)

    # ── 5. Salva se solicitado ───────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        with open(out, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"\n  Salvo em: {out.resolve()}")

    print()


if __name__ == "__main__":
    main()
