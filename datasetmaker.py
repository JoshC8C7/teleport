import json
from datasets import load_dataset


def repldict(in_dict):
    out_dict = {}
    for k, v in in_dict.items():
        out_dict[k] = v.tolist()

    out_dict = json.dumps(out_dict)

    return out_dict


ds = load_dataset("squad_v2")

for spli in ds:
    df = ds[spli].to_pandas()
    df["answers"] = df["answers"].apply(repldict)
    df.to_csv("squadv2/" + spli + ".csv")
