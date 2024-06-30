import pandas as pd

df = pd.read_json('data/annotations-raw.json')

# list[{ "text": str }] -> list[str]
df["ingredients"] = df["ingredients"].map(lambda x: list(map(lambda y: y["text"], x)))
df["ingredients"] = df.apply(lambda x: [v.strip().lower() for v, m in zip(x["ingredients"], x["valid"]) if m and len(v) < 20], axis=1)

df = df[["id", "ingredients"]]

df.to_json(path_or_buf="data/annotations.json", index=False)

