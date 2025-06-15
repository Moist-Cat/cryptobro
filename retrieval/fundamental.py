import os
import json
import requests
from pathlib import Path
import pickle
from functools import cache

# from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch
from typing import Dict, List, Optional
import hashlib

from meta import DB_DIR, BIN_DIR
from retrieval.utils import Model, Document, DIMENSIONS

# --------------------------
# 1. Asset Metadata Structure
# --------------------------

ASSET_METADATA = {
    "LTC": {
        "name": "Litecoin",
        "sources": [
            {
                "type": "github_release",
                "url": "https://github.com/litecoin-project/litecoin",
            }
        ],
    },
    "ETH": {
        "name": "Ethereum",
        "sources": [
            {
                "type": "github_release",
                "url": "https://github.com/ethereum/go-ethereum",
            },
            {
                "type": "github_commit",
                "url": "https://github.com/ethereum/go-ethereum",
            },
            # {
            #    "type": "blog",
            #    "url": "https://blog.ethereum.org",
            # }
        ],
    },
    "SOL": {
        "name": "Solana",
        "sources": [
            {
                "type": "github_commit",
                "url": "https://github.com/solana-foundation/solana-improvement-documents",
            },
        ],
    },
    "TRX": {
        "name": "Tron",
        "sources": [
            {
                "type": "github_release",
                "url": "https://github.com/tronprotocol/java-tron",
            },
            {
                "type": "github_commit",
                "url": "https://github.com/tronprotocol/java-tron",
            },
        ],
    },
}


def transform_cyptopanic(data):
    if "results" not in data or not data["results"]:
        return data

    return [
        {
            "title": item["title"],
            "published_at": item["published_at"],
            "content": item["description"],
        }
        for item in data["results"]
    ]

def transform_messari(data):
    if "data" not in data or not data["data"]:
        return []
    return [
        {
            "title": item["title"],
            "published_at": item["published_at"],
            "content": item["content"],  # Full article text
            "tags": item["tags"],  # e.g. ['defi', 'regulation']
        }
        for item in data["data"]
    ]


NEWS_SOURCES = {
    "messari": {
        "endpoint": "https://data.messari.io/api/v1/news/",
        "transform": transform_messari,
        "params": {},
    },
    "cyptopanic": {
        "endpoint": "https://cryptopanic.com/api/developer/v2/posts/",
        "transform": transform_cyptopanic,
        "headers": {},
        "params": {
            "auth_token": os.getenv("API_CRYPTOPANIC"),
            "public": "false",
            "kind": "news",
            # given during runtime
            "symbol": "currencies",
        },
    },
}


# In-memory cache to reduce load times
# I don't care if someone published something right after I cached the response
@cache
def fetch_news(symbol: str, max_articles=50) -> list:
    """Fetch news from multiple sources with failover"""
    results = []

    # Try sources in priority order
    for source, config in NEWS_SOURCES.items():
        cfg = config["params"].copy()
        if not cfg:
            # assume positional args
            endpoint = config["endpoint"] + symbol
        else:
            endpoint = config["endpoint"]
        print("INF0 - Fetching news from", endpoint, "...")

        # dynamically set volatile params
        for k, v in cfg.copy().items():
            if k in ("symbol", "max_articles"):
                cfg[cfg.pop(k)] = locals()[k]

        response = requests.get(
            endpoint,
            params=cfg,
        )
        data = response.json()
        results.extend(config["transform"](data))
        if len(results) >= max_articles:
            print("INFO - Got enough news")
            break

    if not results:
        return []

    return sorted(results, key=lambda x: x["published_at"], reverse=True)[:max_articles]


# --------------------------
# 2. Scraping Agent
# --------------------------


class ProtocolScraper:
    def __init__(self, storage_path: str = "./fundamental/raw"):
        self.storage_path = DB_DIR / storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize common headers for scraping
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; ProtocolScraper/1.0)"}

    def _get_key(self, source: str):
        return {
            "github_commit": "sha",
            "github_release": "id",
            "blog": "published_at",
        }[source]

    def raw_data_filename(self, symbol, source=None):
        directory = self.storage_path / symbol
        if source is None:
            return next(directory.glob("*.json"))
        return directory / f"{symbol}_{source}".json

    def get_files(self):
        return self.storage_path.glob("**/*.json")

    def fetch_github_commits(self, url: str) -> List[Dict]:
        """Fetch recent commits from GitHub repository"""
        api_url = url.replace("github.com", "api.github.com/repos") + "/commits"
        response = requests.get(api_url, headers=self.headers)
        return response.json()

    def fetch_github_releases(self, url: str) -> List[Dict]:
        """Fetch recent releases from GitHub repository"""
        api_url = url.replace("github.com", "api.github.com/repos") + "/releases"
        response = requests.get(api_url, headers=self.headers)
        return response.json()

    def fetch_blog_updates(self, url: str) -> List[Dict]:
        """Scrape blog posts with protocol updates"""
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        # XXX Implement specific parsing logic per blog structure
        return []

    def save_raw_data(self, symbol: str, data: List[Dict], source_type: str) -> str:
        """Save raw data to filesystem with timestamp"""
        filename = f"{symbol}_{source_type}.json"
        filepath = self.storage_path / symbol / filename

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        old_data = []
        if filepath.exists():
            with open(filepath, "r") as file:
                old_data = json.load(file)

        new = []
        for item in data:
            arr = [
                item[self._get_key(source_type)] == old[self._get_key(source_type)]
                for old in old_data
            ]
            if not any(arr):
                new.append(item)
        old_data.extend(new)

        with open(filepath, "w") as file:
            json.dump(old_data, file)

        return filepath

    def scrape_asset(self, symbol: str) -> List[str]:
        """Main scraping method for an asset"""
        asset = ASSET_METADATA.get(symbol)
        if not asset:
            return []

        stored_files = []
        for source in asset["sources"]:
            if source["type"] == "github_release":
                data = self.fetch_github_releases(source["url"])
            if source["type"] == "github_commit":
                data = self.fetch_github_commits(source["url"])
            elif source["type"] == "blog":
                data = self.fetch_blog_updates(source["url"])

            if data:
                stored_files.append(self.save_raw_data(symbol, data, source["type"]))

        return stored_files


# --------------------------
# 3. Processing Agent
# --------------------------


class DataProcessor:
    def __init__(self, es_host: str = "localhost:9200"):
        try:
            self.es = Elasticsearch(
                os.environ.get("ELASTIC_URL", es_host),
                ca_certs=os.environ["ELASTIC_CERT"],
                basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
            )
            print(f"INFO - {self.es.info()}")
        except Exception as exc:
            print(f"WARNING - Elasticsearch is down ({exc})")
            self.es = None
        self.index_name = "protocol_updates"

        # Initialize text embedding model
        self.model = self.load() or Model()

    def create_es_index(self):
        """Create Elasticsearch index with proper mapping"""
        self.es.options(ignore_status=(400,)).indices.create(
            index=self.index_name,
            body={
                "mappings": {
                    "properties": {
                        "symbol": {"type": "keyword"},
                        "update_date": {"type": "date"},
                        "storage_date": {"type": "date"},
                        "source_type": {"type": "keyword"},
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": DIMENSIONS,  # Match model dimension
                        },
                        "metadata": {"type": "object", "enabled": False},
                    }
                }
            },
        )

    def process_file(self, filepath: str):
        """Process a single raw data file"""
        with open(filepath) as f:
            data = json.load(f)
        print("INFO - Processing file", filepath)

        filename = os.path.basename(filepath)[:-5]

        symbol = filename.split("_")[0]
        source_type = filename.split("_", maxsplit=1)[1]

        # Extract relevant information based on source type
        res = []
        for item in reversed(data):
            processed = None
            if source_type == "github_release":
                processed = self.process_github_release(item)
            elif source_type == "github_commit":
                processed = self.process_github_commit(item)
            elif source_type == "blog":
                processed = self.process_blog_data(item)
            else:
                raise Exception(f"Invalid data type {source_type}")

            res.append(
                (
                    datetime.strptime(processed[0].split("T")[0], "%Y-%m-%d"),
                    processed[1],
                )
            )

        print("INFO - Creating DataFrame with processed data")

        return pd.DataFrame(res, columns=("Date", "Content"))

    def send(self, processed):
        # Generate vector embedding
        vector = self.model.encode(processed["content"])

        # Prepare ES document
        doc = {
            "symbol": symbol,
            "update_date": processed["date"],
            "storage_date": datetime.utcnow().isoformat(),
            "source_type": source_type,
            "content_vector": vector.tolist(),
            "metadata": {
                "source_path": filepath,
                "hash": hashlib.md5(processed["content"].encode()).hexdigest(),
            },
        }

        # Index in Elasticsearch
        self.es.index(index=self.index_name, document=doc)

        return processed

    def process_github_release(self, data: dict) -> dict:
        """Process GitHub commit/release data"""
        # Implement specific parsing logic
        return data["published_at"], data["body"]

    def process_github_commit(self, data: dict) -> dict:
        """Process GitHub commit/release data"""
        # Implement specific parsing logic
        return data["commit"]["committer"]["date"], data["commit"]["message"]

    def process_blog_data(self, data: dict) -> dict:
        """Process blog post data"""
        # Implement specific parsing logic
        return data["published_date"], data["content"]

    def save(self):
        with open(BIN_DIR / "lsi_model.pickle", "wb") as file:
            pickle.dump(self.model, file)

    def load(self):
        if not Path(BIN_DIR / "lsi_model.pickle").exists():
            return None
        print("INFO - Loading LSI model")
        with open(BIN_DIR / "lsi_model.pickle", "rb") as file:
            return pickle.load(file)

def update():
    scraper = ProtocolScraper()
    processor = DataProcessor()

    for symbol in ["BTC", "ETH", "LTC", "SOL", "TRX"]:
        stored_files = scraper.scrape_asset(symbol)
    stored_files = scraper.get_files()

    # Process new files
    data = []
    for filepath in stored_files:
        for doc in processor.process_file(filepath).itertuples():
            data.append(Document(identifier=doc.Date, text=doc.Content))
    processor.model.fit(data)
    processor.save()


# --------------------------
# 4. Main Execution Flow
# --------------------------

if __name__ == "__main__":
    # Initialize components
    scraper = ProtocolScraper()
    processor = DataProcessor()
    # processor.create_es_index()

    stored_files = scraper.get_files()

    # Process new files
    data = []
    for filepath in stored_files:
        for doc in processor.process_file(filepath).itertuples():
            data.append(Document(identifier=doc.Date, text=doc.Content))
    processor.model.fit(data)
    processor.save()
