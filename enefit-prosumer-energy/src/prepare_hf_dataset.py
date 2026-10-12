"""Build data_hf/: per-table parquet (lazy-loadable), big weather tables sharded by month, MANIFEST.json with sha256.
Cloud hosts fetch only the shards they need with hf_hub_download. Source CSVs stay local in data/ (CC BY-NC-SA, competition use)."""
import os, json, hashlib, shutil
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, O = f"{ROOT}/data", f"{ROOT}/data_hf"
shutil.rmtree(O, ignore_errors=True); os.makedirs(O)
man = {}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def put(df, rel):
    p = f"{O}/{rel}"; os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_parquet(p, index=False, compression="zstd")
    man[rel] = dict(rows=len(df), cols=list(df.columns), bytes=os.path.getsize(p), sha256=sha(p))
    print(rel, len(df), man[rel]["bytes"], flush=True)
dt = ["datetime", "date", "origin_date", "forecast_date", "origin_datetime", "forecast_datetime", "prediction_datetime"]
def rd(name, **kw):
    cols = pd.read_csv(f"{D}/{name}", nrows=0).columns
    return pd.read_csv(f"{D}/{name}", parse_dates=[c for c in cols if c in dt], **kw)
for n in ["train", "client", "electricity_prices", "gas_prices", "weather_station_to_county_mapping"]:
    put(rd(f"{n}.csv"), f"{n}.parquet")
hw = rd("historical_weather.csv"); hw["_m"] = hw.datetime.dt.strftime("%Y-%m")
for m, g in hw.groupby("_m"): put(g.drop(columns="_m"), f"historical_weather/{m}.parquet")
del hw
f32 = {c: "float32" for c in pd.read_csv(f"{D}/forecast_weather.csv", nrows=0).columns
       if c not in ("origin_datetime", "forecast_datetime", "data_block_id", "hours_ahead")}
fw = rd("forecast_weather.csv", dtype=f32); fw["_m"] = fw.forecast_datetime.dt.strftime("%Y-%m")
for m, g in fw.groupby("_m"): put(g.drop(columns="_m"), f"forecast_weather/{m}.parquet")
for n in ["client", "electricity_prices", "forecast_weather", "gas_prices", "historical_weather", "revealed_targets", "sample_submission", "test"]:
    put(rd(f"example_test_files/{n}.csv"), f"example_test_files/{n}.parquet")
shutil.copy(f"{D}/county_id_to_name_map.json", f"{O}/county_id_to_name_map.json")
man["county_id_to_name_map.json"] = dict(bytes=os.path.getsize(f"{O}/county_id_to_name_map.json"), sha256=sha(f"{O}/county_id_to_name_map.json"))
json.dump(man, open(f"{O}/MANIFEST.json", "w"), indent=1)
open(f"{O}/README.md", "w").write("""# Enefit prosumers (private working copy)
Competition data, CC BY-NC-SA 4.0, competition use only. Parquet per table; weather tables sharded by month
(`forecast_weather/YYYY-MM.parquet`, keyed by forecast_datetime; `historical_weather/YYYY-MM.parquet`).
Load lazily with `hf_hub_download(repo_id, "<path>", repo_type="dataset")`; verify against MANIFEST.json sha256.
""")
print("files", len(man), "total MB", sum(v["bytes"] for v in man.values()) / 1e6)
