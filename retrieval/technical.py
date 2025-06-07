import os
import csv
from datetime import datetime, timedelta
import requests
import pandas as pd

from meta import DB_DIR

CRYPTO = "CRYPTO"
COMMODITIES = "COMMODITIES"
STOCKS = "STOCKS"


class AssetScraper:
    def __init__(self, storage_dir=DB_DIR):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Configure data sources
        self.sources = {
            CRYPTO: BinanceSource(),
            COMMODITIES: AlphaVantageSource(api_key=os.getenv("API_VANTAGE")),
            STOCKS: TwelveDataSource(api_key=os.getenv("API_TWELVE")),
        }

    def update_assets(self, asset_list):
        """Main update method for all assets"""
        results = []
        for asset in asset_list:
            source = self._get_source(asset["type"])
            try:
                data = self._process_asset(asset, source)
                results.append((asset["symbol"], True, "Updated"))
            except Exception as e:
                results.append((asset["symbol"], False, str(e)))
        return results

    def _get_source(self, asset_type):
        for key in self.sources:
            if key in asset_type.upper():
                return self.sources[key]
        raise ValueError(f"No source found for {asset_type}")

    def _process_asset(self, asset, source):
        # Get existing data or create new
        filepath = os.path.join(self.storage_dir, f"{asset['symbol']}.csv")
        existing_data = self._load_existing(filepath)

        # Get new data
        new_data = source.fetch(
            symbol=asset["symbol"], start_date=self._get_start_date(existing_data)
        )

        # Merge and validate
        combined = self._merge_data(existing_data, new_data)
        self._validate_data(combined)

        # Save with consistent format
        combined.to_csv(filepath, index=False)
        return combined

    def _load_existing(self, filepath):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=["Date"], date_format="%Y-%m-%d")
            return df.sort_values("Date")
        return pd.DataFrame(columns=["Date", "Close"])

    def _get_start_date(self, existing_data):
        if not existing_data.empty:
            return existing_data["Date"].max() + timedelta(days=1)
        return datetime(2015, 1, 1)  # Default history

    def _merge_data(self, old, new):
        if not len(new):
            return old.sort_values("Date")
        if not len(old):
            return new.sort_values("Date")
        # make the dates consistent with each other
        old["Date"] = old["Date"].dt.date

        combined = pd.concat([old, new])
        combined = combined.drop_duplicates("Date")
        return combined.sort_values("Date").reset_index(drop=True)

    def _validate_data(self, df):
        if len(df) < 365:
            raise ValueError("Insufficient historical data")
        if df["Close"].isnull().any():
            raise ValueError("Missing closing prices detected")


# ====================
# Data Source Implementations
# ====================


class DataSource:
    """Base class for all data sources"""

    def fetch(self, symbol, start_date):
        raise NotImplementedError


class BinanceSource(DataSource):
    def fetch(self, symbol, start_date):
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": int(start_date.timestamp() - 1) * 1000,
        }
        response = requests.get("https://api.binance.com/api/v3/klines", params=params)
        data = response.json()

        dates = [(datetime.fromtimestamp(int(c[0]) / 1000)).date() for c in data]
        close = [c[4] for c in data]

        return pd.DataFrame(
            {
                "Date": dates,
                "Close": close,
            }
        )


class TwelveDataSource(DataSource):
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch(self, symbol, start_date):
        params = {
            "symbol": symbol,
            "interval": "1day",
            "apikey": self.api_key,
            "start_date": start_date.strftime("%Y-%m-%d"),
        }
        response = requests.get("https://api.twelvedata.com/time_series", params=params)
        if "values" not in response.json():
            raise Exception(response.json().get("message"))
        data = response.json()["values"]

        dates = [datetime.fromisoformat(i["datetime"]).date() for i in data]
        close = [i["close"] for i in data]

        return pd.DataFrame(
            {
                "Date": dates,
                "Close": close,
            }
        )


class AlphaVantageSource(DataSource):
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch(self, symbol, start_date):
        params = {
            "function": symbol,
            "interval": "daily",
            "apikey": self.api_key,
            "outputsize": "full",
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        data = response.json()["data"]

        data = [
            (datetime.fromisoformat(v["date"]).date(), v["value"])
            for v in data
            if datetime.strptime(v["date"], "%Y-%m-%d") >= start_date
            and v["value"].split(".")[0].isnumeric()
        ]

        dates = [i[0] for i in data]
        close = [i[1] for i in data]

        return pd.DataFrame(
            {
                "Date": dates,
                "Close": close,
            }
        )
