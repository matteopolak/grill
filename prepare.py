import pandas as pd
import pickle
import re

l1 = pd.read_json('data/layer1.json')
l2 = pd.read_json('data/layer2.json')

# match l1.id to l2.id, then replace l1.id with l2.images[0].id
df = l1.merge(l2, on="id", indicator=True, how="inner")
df["id"] = df.apply(lambda x: x["images"][0]["id"][:-4], axis=1)

# list[{ "text": str }] -> list[str]
df["ingredients"] = df["ingredients"].map(lambda x: list(map(lambda y: y["text"], x)))
df["ingredients"] = df.apply(lambda x: [v.strip().lower() for v in x["ingredients"] if len(v) < 20], axis=1)

# remove text between parentheses, and (if the ingredient starts with a number),
# remove the number and the next word following it (if there are multiple numbers in a row, keep removing
# until there's a word). e.g. "1 1/2 cup sugar" -> "sugar"
parentheses_regex = re.compile(r"\([^)]*\)")
number_regex = re.compile(r"^\s*[-\d][^\s]*\s+(?:[-\d][^\s]+\s+)*[^\s]*\s*(?=[^-\d\s])")
after_comma_regex = re.compile(r",.*")

def clean_ingredient(ingredient):
    ingredient = re.sub(parentheses_regex, "", ingredient)
    ingredient = re.sub(number_regex, "", ingredient)
    ingredient = re.sub(after_comma_regex, "", ingredient)
    ingredient = ingredient.replace("\"", "").replace("\'", "").strip()

    return ingredient

df["ingredients"] = df["ingredients"].map(lambda x: list(set(filter(lambda i: 2 < len(i) < 15 and not i.endswith("ed"), map(clean_ingredient, x)))))

classes = df.explode("ingredients")["ingredients"].value_counts().to_dict()
classes = { k: v for k, v in sorted(classes.items(), key=lambda item: item[1], reverse=True)[:500] }

# remove ingredients that are not in the top 500
df["ingredients"] = df["ingredients"].map(lambda x: list(filter(lambda i: i in classes, x)))

# remove rows with no ingredients
df = df.query("ingredients.str.len() > 0")

ann = df[["id", "ingredients", "partition"]]
ann.to_json(path_or_buf="data/annotations.json", index=False, orient="records")

with open("data/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

