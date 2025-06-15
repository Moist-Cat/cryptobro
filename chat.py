import numpy as np
from datetime import datetime, timedelta
import requests

import pandas as pd

from meta import DB_DIR, SUMMARIZER_ENDPOINT
from retrieval import fundamental

# foreign
from g4f.client import Client

MINUTES = 1
HOURS = MINUTES * 3600
DAYS = HOURS * 24
WEEKS = DAYS * 7
MONTHS = WEEKS * 4

VALUE_TO_TERM = {
    "risk": (
        (0.3, "Very Low risk tolerance"),
        (0.6, "Moderate risk tolerance"),
        (0.8, "High risk tolerance"),
        (99, "Very High risk tolerance"),
    ),
    "auto_close": (
        (0.25, "Intraday operation (hours)"),
        (1, "Day trading context"),
        (3, "Short-term swing trade (2-3 days)"),
        (7, "Swing trade (weekly horizon)"),
        (99, "Position trade (long-term)"),
    ),
    "timeframe": (
        (MINUTES, "Ultra-short-term tick data"),
        (HOURS, "Hourly charts"),
        (DAYS, "Daily candles"),
        (WEEKS, "Weekly charts"),
        (MONTHS, "Monthly timeframes"),
    ),
    "position_size": (
        (0.05, "Very small position <5%"),
        (0.15, "Standard position 5-15%"),
        (99, "Large position >15%"),
    ),
}


class Chatbot:
    def __init__(self, model):
        self.model = model
        self._last_response = 0
        self.client = Client()

        self.freq_penalty = 0.2
        self.temperature = 0.5

    def reply(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            frequency_penalty=self.freq_penalty,
            temperature=self.temperature,
            messages=[{"role": "system", "content": prompt}],
        )
        self._last_response = response
        return response.choices[0].message.content


chatbot = Chatbot("gpt-4")


def summarize_document(text, max_length=150, algorithm="kmeans"):
    key_phrases = extract_key_phrases(
        text=text,
        # fixed
        top_k=5,
    )

    # Step 2: T5 abstraction (fluent summary generation)
    summary = abstractive_summary(
        text=key_phrases,
        max_length=max_length,
    )

    return summary


def summarize_news(articles: list, max_length=150, target=5 + 1) -> list:
    """Chain TinyBERT extraction with T5 abstraction"""
    processed = []

    print("INFO - Summarizing", len(articles), "articles")

    prompt = "- News Summary:"
    last_index = 0
    for index in range(len(articles) // target, len(articles), len(articles) // target):
        # we use a set to avoid duplicates
        # when summarizing data from aggregators
        # NOTE In this step, we want to know the *general topic* of the article
        # thus, we use the cosine similarity algorithm (most central vectors/embeddings)
        # to extract it
        # Three newlines to differenciate between articles
        block = "\n\n\n".join(
            set(
                [
                    summarize_document(article["content"], algorithm="cosim")
                    for article in articles[last_index:index]
                ]
            )
        )
        #print("DEBUG - Summary block:", block)
        # Here we want to know the *most important topics* discussed in the
        # blockーgiven the block contains several articles, there will be more than one
        # different topics even if they are not central to all articles.
        # We want to avoid redundancy as well.
        summary = summarize_document(block, max_length, algorithm="kmeans")
        # print("DEBUG - Result:", summary)

        prompt += "\n    * " + summary

        last_index = index

    return prompt


def extract_key_phrases(text: str, top_k=10) -> list:
    params = {
        "text": text,
        "num_sentences": top_k,
        "mode": "extractive",
        "model": "huawei-noah/TinyBERT_General_4L_312D",
    }
    try:
        return requests.post(SUMMARIZER_ENDPOINT, json=params).json()["summary"]
    except Exception as exc:
        print("ERROR -", exc)
        return text


def abstractive_summary(text, max_length=150) -> str:
    """Generate abstractive summary from key phrases"""
    # Prepare input
    params = {
        "text": text,
        "max_length": max_length,
        "mode": "abstractive",
        "model": "t5-small",
    }
    try:
        return requests.post(SUMMARIZER_ENDPOINT, json=params).json()["summary"]
    except Exception as exc:
        print("ERROR -", exc)
        return text


def summarize_rag_results(rag_results: tuple, max_age_days: int = 90) -> str:
    """
    Summarizes RAG results with fundamental and price data, flagging obsolete information.

    Args:
        rag_results: Tuple of (commits_df, price_df) from lsa_rag_retrieve()
        max_age_days: Maximum allowed age for data to be considered relevant

    Returns:
        Summary string with fundamental insights and price impact analysis
    """
    commits_df, price_df = rag_results
    current_date = datetime.now()
    summary_parts = []

    if (commits_df is None) or commits_df.empty:
        print("WARNING - RAG returned no data")
        return ""
    print("INFO - Summarising RAG data")

    # 1. Fundamental Data Summary (Commits/Releases)
    # Convert dates and find most recent
    latest_commit = commits_df["Date"].max()
    days_since_commit = (current_date - latest_commit).days

    # Check obsolescence
    commit_summaries = []
    if days_since_commit <= max_age_days:
        # Process most significant commits (top 3 by recency)
        recent_commits = commits_df

        for _, row in recent_commits.iterrows():
            content = row["Content"]

            # these are usually a bunch of bullet points
            commit_summaries.append(summarize_document(content, algorithm="kmeans"))

        summary_parts.append("Recent protocol developments:")
        summary_parts.extend([f"• {s}" for s in commit_summaries])

    # 2. Price Impact Analysis
    # Don't do anything if we don't have recent commits
    if commit_summaries:
        # Merge fundamental events with price data
        merged_df = pd.merge(
            commits_df, price_df, on="Date", suffixes=("_fund", "_price")
        )
        if not merged_df.empty:
            # Calculate price impact metrics
            impact_summary = []
            for _, row in merged_df.iterrows():
                impact_summary.append(f"{row['Date']}: {row['Percent_Change']:.2f}%)")

            summary_parts.append("\n📊 Price impact of protocol events:")
            summary_parts.extend([f"• {s}" for s in impact_summary])
        else:
            print(
                "ERROR - Somehow, the date of the commits and the events never matched"
            )

    print("INFO - Created", len(summary_parts), "summaries on RAG data")

    return "\n".join(summary_parts)


def lsa_rag_retrieve(symbol: str, auto_close: int):
    """
    - Get latest protocol update/relevant news
    - Find similar
    """
    pro = fundamental.ProtocolScraper()
    dat = fundamental.DataProcessor()

    tech_filename = DB_DIR / f"{symbol}.csv"
    if not tech_filename.exists():
        print(f"WARNING - No data for `{symbol}`")
        return None

    try:
        fun_filename = pro.raw_data_filename(
            symbol if "USDT" not in symbol else symbol[:-4]
        )
    except StopIteration:
        print("WARNING - No data for", symbol)
        return None, None

    # XXX repeated code 2 strikes
    fun_data = dat.process_file(fun_filename)
    tech_data = pd.read_csv(
        tech_filename,
        parse_dates=[
            "Date",
        ],
    )

    merged = pd.merge(fun_data, tech_data, on="Date")
    res = dat.model.predict(merged.iloc[-1].Content)

    matching = tech_data.loc[tech_data.Date.isin(res), "Date"]

    tech_data["Percent_Change"] = 100 * (
        np.log(tech_data["Close"].shift(-auto_close) / tech_data["Close"])
    )

    return fun_data[fun_data.Date.isin(res)], tech_data[tech_data.Date.isin(res)]


def expand_query(
    auto_close: float,  # Max days until auto-close
    risk: float,  # 0.0-1.0 risk tolerance
    timeframe: int = DAYS,  # e.g. "1h", "4h", "daily"
    position_size: float = None,  # As percentage of capital
    indicators: dict = None,  # { "rsi": 65, "macd": "bullish", ... }
) -> str:
    """
    Transforms trading parameters into natural language context.
    Returns an expanded prompt that guides the LLM toward trade-aware responses.
    """

    # --- CORE PARAMETER EXPANSION ---
    position_characterisaton = []
    for param in ("risk", "timeframe", "position_size"):
        for value, term in VALUE_TO_TERM[param]:
            if locals()[param] <= value:
                position_characterisaton.append(term)
                break

    # 2. Indicator-Based Expansion (if provided)
    indicator_context = []
    if indicators:
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            # no need to mention anything if the
            # results are inconclusive
            if rsi:
                indicator_context.append(
                    f"RSI shows {'overbought' if rsi < 0 else 'oversold'} conditions"
                )

        if "expected_value" in indicators:
            ev = indicators["expected_value"]
            if ev:
                indicator_context.append(
                    f"Expected Tendency: {'Bullish' if ev > 0 else 'Bearish'}"
                )
            # else do nothing

        if "hlc" in indicators:
            high, low, close = indicators["hlc"]
            indicator_context.append(
                f"Price Levels | Time-Frame High: {high:.2f}, Time-Frame Low: {low:.2f}, Current: {close:.2f}"
            )

        if "ci" in indicators:
            low, high = indicators["ci"]
            indicator_context.append(f"Confidence Interval: [{low:.2f}, {high:.2f}]")

    position = "    * " + "\n    * ".join(position_characterisaton)
    notes = "    * " + "\n    * ".join(indicator_context)
    # --- PROMPT ASSEMBLY ---
    base_context = (
        "Trading Context:\n"
        "- Operation details:\n"
        f"{position}\n"
        "- Technical Notes:\n"
        f"{notes}\n"
        f"Auto-Close Trigger: Position will automatically close within {auto_close:.1f} days"
    )

    return base_context


def assemble_prompt(
    expanded_query: str,
    rag_summary: str,
    news_summary: str,
    user_query: str,
) -> str:
    return f"""
## ROLE: Senior Quantitative Trading Analyst
You're assisting a trader with access to real-time market data and proprietary models.
Current time: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## CONTEXT BLOCKS

### FUNDAMENTAL PROTOCOL ANALYSIS
{rag_summary or "⚠️ No fundamental data available"}

### MARKET NEWS SUMMARY
{news_summary or "⚠️ No recent news available"}

### TRADER'S QUERY
"{user_query}"

### EXPANDED CONTEXT
{expanded_query}

## INSTRUCTIONS
1. CROSS-VALIDATE technical and fundamental signals
2. HIGHLIGHT time-sensitive opportunities (inside the user's preferred timeframe)
3. FLAG conflicting signals between models/news
4. PROPOSE concrete actions: ENTRY/EXIT/HOLD with price targets
5. QUANTIFY confidence level (1-10) based on evidence
6. WARN about black swan risks if detected
"""


if __name__ == "__main__":
    expanded = expand_query(
        user_query="Should I long BTC here?",
        auto_close=3.5,
        risk=0.72,
        timeframe=HOURS,
        position_size=0.12,
        indicators={
            "rsi": 1,
            "expected_value": 0.003,
            "ci": (10000, 20000),
            "hlc": (10000, 20000, 150000),
        },
    )
    print(expanded)
