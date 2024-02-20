import jsonlines
with open('diverse_1000_seed42.jsonl', 'r', encoding='utf-8') as f1:
    temp = f1.readlines()
with open('diverse_1000_merge.jsonl', 'a', encoding='utf-8') as f2:
    for i in temp:
        f2.write(i)