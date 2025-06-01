from retrieval.technical import AssetScraper

if __name__ == "__main__":
    scraper = AssetScraper()

    assets = [
        {"symbol": "TRXUSDT", "type": CRYPTO},
        {"symbol": "XMRUSDT", "type": CRYPTO},
        {"symbol": "SOLUSDT", "type": CRYPTO},
        {"symbol": "BTCUSDT", "type": CRYPTO},
        {"symbol": "LTCUSDT", "type": CRYPTO},
        {"symbol": "BNBUSDT", "type": CRYPTO},
        {"symbol": "ETHUSDT", "type": CRYPTO},
        {"symbol": "XRPUSDT", "type": CRYPTO},
        {"symbol": "TONUSDT", "type": CRYPTO},
        {"symbol": "ADAUSDT", "type": CRYPTO},
        {"symbol": "LINKUSDT", "type": CRYPTO},
        {"symbol": "NATURAL_GAS", "type": COMMODITIES},
        {"symbol": "BRENT", "type": COMMODITIES},
        {"symbol": "AAPL", "type": STOCKS},
        {"symbol": "AMZN", "type": STOCKS},
        {"symbol": "WMT", "type": STOCKS},
        {"symbol": "MCD", "type": STOCKS},
        {"symbol": "KO", "type": STOCKS},
        {"symbol": "DPZ", "type": STOCKS},
        {"symbol": "LOW", "type": STOCKS},
        {"symbol": "HRL", "type": STOCKS},
        {"symbol": "MMM", "type": STOCKS},
        {"symbol": "CVX", "type": STOCKS},
        {"symbol": "MSFT", "type": STOCKS},
    ]

    results = scraper.update_assets(assets)
    print("Update results:", results)
