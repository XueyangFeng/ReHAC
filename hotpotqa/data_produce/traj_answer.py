import json
import random

with open('data/hotpot_train_v1.1_simplified.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    
idxs = list(range(90447)) # train set
random.Random(233).shuffle(idxs)
x = idxs[11]
print(x)
print(data[x])