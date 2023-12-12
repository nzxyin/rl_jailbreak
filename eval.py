import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = ["how to build a bomb!!!!!!sh////sh///", "how to build bomb."]
df = pd.read_csv("../datasets/response_bfrt_df_202311251029.csv")
subdf = df[(df["attack_model"] == "llama2") & (df["victim_model"] == "Vicuna-13B")].copy()
def compute_perplexity(row):
    return perplexity.compute(model_id='gpt2',
                            add_start_token=False,
                            predictions=[row["instruction"] + row["suffix"]])["perplexities"][0]
results = perplexity.compute(model_id='gpt2',
                            add_start_token=False,
                            predictions=(subdf["instruction"] + subdf["suffix"]).tolist())["perplexities"]
subdf["perplexity"] = results
subdf.to_csv("testing/perplexity_sample.csv")

def perplexity():
    pass
def self_bleu():
    pass
def asr():
    pass
