from datasets import load_dataset


def get_loaders(name, tokenizer, n_samples=128, seq_len=2048):
    if name == "wikitext2":
        return get_wikitext2(tokenizer, n_samples, seq_len)
    elif name == "c4":
        return get_c4(tokenizer, n_samples, seq_len)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_wikitext2(tokenizer, n_samples, seq_len):
    # add randomization to sequence selection
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    encodings = tokenizer(text, return_tensors="pt")

    input_ids_list = []
    for i in range(n_samples):
        start = i * seq_len
        end = start + seq_len
        if end > encodings.input_ids.shape[1]:
            break
        input_ids_list.append(encodings.input_ids[:, start:end].clone())
    return input_ids_list



def get_c4(tokenizer, n_samples, seq_len):
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    data = data.shuffle(seed=42, buffer_size=10000)

    input_ids_list = []
    for i, batch in enumerate(data):
        if len(input_ids_list) >= n_samples:
            break
        tokens = tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        if tokens.shape[1] >= seq_len:
            input_ids_list.append(tokens[:, :seq_len])

    return input_ids_list
