import os
import json
import requests
from pathlib import Path

# from bs4 import BeautifulSoup
from datetime import datetime
from elasticsearch import Elasticsearch
from typing import Dict, List, Optional
import hashlib

from meta import DB_DIR

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
        ],
    },
}

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
    def __init__(self, model=None, es_host: str = "localhost:9200"):
        self.es = Elasticsearch(
            os.environ.get("ELASTIC_URL", es_host),
            ca_certs=os.environ["ELASTIC_CERT"],
            basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
        )
        print(f"INFO - {self.es.info()}")
        self.index_name = "protocol_updates"

        # Initialize text embedding model
        self.model = model

    def create_es_index(self):
        """Create Elasticsearch index with proper mapping"""
        self.es.indices.create(
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
                            "dims": 384,  # Match model dimension
                        },
                        "metadata": {"type": "object", "enabled": False},
                    }
                }
            },
        )
        self.es.options(ignore_status=(400,))

    def process_file(self, filepath: str):
        """Process a single raw data file"""
        with open(filepath) as f:
            data = json.load(f)

        symbol = os.path.basename(filepath).split("_")[0]
        source_type = os.path.basename(filepath).split("_")[1]

        # Extract relevant information based on source type
        if source_type == "github_release":
            processed = self.process_github_release(data)
        elif source_type == "github_commit":
            processed = self.process_github_commit(data)
        elif source_type == "blog":
            processed = self.process_blog_data(data)

        return processed

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
        return {"content": data["body"], "date": data["published_at"]}

    def process_github_commit(self, data: dict) -> dict:
        """Process GitHub commit/release data"""
        # Implement specific parsing logic
        return {
            "content": data["commit"]["message"],
            "date": data["commit"]["committer"]["date"],
        }

    def process_blog_data(self, data: dict) -> dict:
        """Process blog post data"""
        # Implement specific parsing logic
        return {"content": data["content"], "date": data["published_date"]}


# --------------------------
# 4. Main Execution Flow
# --------------------------

if __name__ == "__main__":
    # Initialize components
    scraper = ProtocolScraper()
    processor = DataProcessor()
    processor.create_es_index()

    # Scrape and process all assets
    # for symbol in ["BTC", "ETH", "LTC", "SOL", "TRX"]:
    #    stored_files = scraper.scrape_asset(symbol)

    # Process new files
    # for filepath in stored_files:
    #    processor.process_file(filepath)
