import json
from datasets import load_dataset

def create_conversation(sample):
    return {
        "prompt": sample["input"],
        "completion": sample["output"]
    }

def main():

    # Read the source JSON file
    # data from https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2025/1/WR.json
    with open('data/WR.json', 'r') as src_data_file:
        data = json.load(src_data_file)

    output = []

    for entry in data:
        player_name = entry.get("PlayerName")
        player_recyds = entry.get("ReceivingYDS")
        if player_recyds is None or player_recyds == "":
            player_recyds = 0  # Default to 0 if ReceivingYDS is missing or null
        if player_name:  # Only process entries with a PlayerName
            question = f"How many receiving yards did {player_name} have in week 1 of the 2025 NFL season?"
            answer = f"{player_name} had {player_recyds} receiving yards in week 1 of the 2025 NFL season."
            qa_pair = {
                "input": question,
                "output": answer
            }
            output.append(qa_pair)

    with open('data/finetune_data.json', 'w') as outfile:
        json.dump(output, outfile, indent=2)

    folder="data/" 

    dataset = load_dataset("json", data_files=folder + "finetune_data.json", split="train")
    dataset = dataset.shuffle().select(range(150))
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

    # dataset.to_json(folder + "all.jsonl", orient="records")

    dataset = dataset.train_test_split(test_size=0.2)
    dataset_test_valid = dataset['test'].train_test_split(0.5)
    print(dataset["train"][0])
    print(dataset["train"][-1])
    dataset["train"].to_json(folder + "train.jsonl", orient="records")
    dataset_test_valid["train"].to_json(folder + "test.jsonl", orient="records")
    dataset_test_valid["test"].to_json(folder + "valid.jsonl", orient="records")

if __name__ == "__main__":
    main()