import polars as pl
import pickle

l1 = pl.read_json("data/layer1.json")
l2 = pl.read_json("data/layer2.json")

df = l1.join(l2, on="id")

df.replace_column(
    df.get_column_index("id"),
    df.select(pl.col("images")
        .list.first().struct.field("id")
        .str.head(-4)
        .alias("id")).to_series())

df.drop_in_place("url")
df.drop_in_place("title")
df.drop_in_place("instructions")
df.drop_in_place("images")

df.replace_column(
    df.get_column_index("ingredients"),
    df.select(pl.col("ingredients")
        .list.eval(pl.element()
            .struct.field("text")
            .str.to_lowercase()
            # remove text between parentheses, and (if the ingredient starts with a number),
            # remove the number and the next word following it (if there are multiple numbers in a row, keep removing
            # until there's a word). e.g. "1 1/2 cup sugar" -> "sugar"
            .str.replace_all(r"\([^()]*\)|,.*|\"|\'|^\w\b", "")
            .str.replace_all(
                r"^\s*[-\d/][^\s]*\s+(?:[-\d/][^\s]+\s+)*[^\s]*\s*([^\-\d\s/])",
                "${1}"
            )
            .str.replace_all(r"\.", "")
            .str.replace_all(r"\bnull\b", "")
            .str.strip_chars()
            .str.strip_prefix("cup ")
            .str.strip_prefix("cups ")
            .str.strip_prefix("tbsp ")
            .str.strip_prefix("tablespoon ")
            .str.strip_prefix("tablespoons ")
            .str.strip_prefix("tsp ")
            .str.strip_prefix("lbs ")
            .str.strip_prefix("lb ")
            .str.strip_prefix("dash ")
            .str.strip_chars()
            .str.strip_prefix("of ")
            .str.replace_all(r"\s{2,}", " ")
            .str.strip_chars())
        .list.eval(pl.element()
            .filter(pl.element().str.len_bytes().is_between(2, 15) & ~pl.element().str.ends_with("ed")))
        .list.unique()
        .alias("ingredients")).to_series())

classes = (df.get_column("ingredients")
    .explode().alias("ingredient")
    .value_counts()
    .filter(pl.col("count") >= 80))

# remove ingredients that are not in the classes
df.replace_column(
    df.get_column_index("ingredients"),
    df.select(pl.col("ingredients")
        .list.eval(pl.element().filter(pl.element().is_in(classes.get_column("ingredient"))))).to_series())

df = df.filter(pl.col("ingredients").list.len() > 6)

df.write_parquet("data/annotations.parquet")

# remove unused ingredients that did not pass the above filter
# and divide the number of recipes by the number of occurrences of each ingredient
# so it can be used as a weight in the loss function
classes = (df.get_column("ingredients")
    .explode().alias("ingredient")
    .value_counts(sort=True)
    .filter(pl.col("count") > 80)
    .drop_nulls())

n_recipes = len(df)
classes = classes.replace_column(
    classes.get_column_index("count"),
    classes.select((pl.lit(n_recipes) / pl.col("count")).alias("count")).to_series())

classes = classes.to_dict()
classes = dict(zip(classes["ingredient"], classes["count"]))

with open("data/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

