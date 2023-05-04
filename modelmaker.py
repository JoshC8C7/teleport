from transformers import AutoModel, AutoTokenizer

import safetensors


models = [          'albert-base-v2',
    'facebook/bart-large-mnli',
    'microsoft/deberta-v2-xlarge-mnli',
    'xlm-roberta-base',
          'distilbert-base-uncased',
          'distilbert-base-cased-distilled-squad',
          'google/electra-base-discriminator',
          'roberta-base',
          'dmis-lab/biobert-v1.1',
          ]

for k in models:
    print("######################################", k, "############################")
    model = AutoModel.from_pretrained(k)
    tokenizer = AutoTokenizer.from_pretrained(k)

    outname = k.replace('/','')
    model.save_pretrained(outname,safe_serialization=False)
    tokenizer.save_pretrained(outname,safe_serialization=False)


#tinybert 5cbacbb1-fea0-412d-a226-06007b152681


#iris post --model 5cbacbb1-fea0-412d-a226-06007b152681 --dataset 2c588e5b-0b93-4723-a5ed-a8960d6f8140 --task sequence_classification --name josh-test-cdo -s --text-fields text -nl 4
