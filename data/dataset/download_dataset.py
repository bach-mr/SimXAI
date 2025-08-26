from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb", split="test")

df = ds.to_csv("imdb_test.csv", index=True)