import polars as pl
import pickle

l1 = pl.read_json("data/layer1.json")
l2 = pl.read_json("data/layer2.json")

# match l1.id to l2.id, then replace l1.id with l2.images[0].id
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
            .str.strip_chars()
            # remove text between parentheses, and (if the ingredient starts with a number),
            # remove the number and the next word following it (if there are multiple numbers in a row, keep removing
            # until there's a word). e.g. "1 1/2 cup sugar" -> "sugar"
            .str.replace_all(
                r"\([^()]*\)|,.*|\"|\'",
                ""
            )
            .str.replace_all(
                r"^\s*[-\d][^\s]*\s+(?:[-\d][^\s]+\s+)*[^\s]*\s*([^\-\d\s])",
                "${1}"
            ))
        .list.eval(pl.element()
            .filter(pl.element().str.len_bytes().is_between(2, 15) & ~pl.element().str.ends_with("ed")))
        .alias("ingredients")).to_series()
)

classes = (df.get_column("ingredients")
    .explode().alias("ingredient")
    .value_counts(sort=True)
    .filter(pl.col("count") >= 100))

# remove ingredients that are not in the classes
df.replace_column(
    df.get_column_index("ingredients"),
    df.select(pl.col("ingredients")
        .list.eval(pl.element().filter(pl.element().is_in(classes.get_column("ingredient"))))).to_series())

df.filter(pl.col("ingredients").list.len() > 5)

num_recipes = len(df)

df.write_parquet("data/annotations.parquet")

# remove unused ingredients that did not pass the above filter
# and divide the number of recipes by the number of occurrences of each ingredient
# so it can be used as a weight in the loss function
classes = (df.get_column("ingredients")
    .explode().alias("ingredient")
    .value_counts()
    .filter(pl.col("count") > 100))
classes = classes.replace_column(
    classes.get_column_index("count"),
    classes.select(pl.lit(num_recipes) / pl.col("count")).to_series())

with open("data/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

