import datasets
delta = 0.1

train_dataset = datasets.load_dataset("json", data_files=f'data/advantage/advantage_sample_count_{delta}.jsonl', cache_dir="./json2hf_dataset")

train_dataset.save_to_disk(f'advantage_sample_count_{delta}_hf')