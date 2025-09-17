# LLM LoRA Fine-Tuning using MLX LM

This is an exploratory and demonstration project that uses Low-Rank Adaptation (LoRA) fine-tuning to improve Large Language Model (LLM) responses related to the fine-tuning training data.

## Built With

* Locally-hosted microsoft/Phi-3.5-mini-instruct LLM
* [MLX LM](https://github.com/ml-explore/mlx-lm) python package used for a variety of tasks with LLMs on Apple Silicon
* NFL player data from https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2025/1/WR.json

## Overview

LoRA fine-tuning is one of several techniques that improves LLM performance by providing additional data to the model.  This additional data may be something that the LLM was not originally trained on and is unable to consider in its response.  Examples could include private company data, highly specialized data, or data that is otherwise unavailable to general-purpose LLMs.  Because of the effort involved, repeated LoRA fine-tuning is likely not a solution to improve LLMs with data that frequently changes.  However, because LoRA fine-tuning is performed once, the development effort is only incurred once.  This is an advantage compared to other LLM response improvement techniques such as RAG or large context windows, which incur processing penalties with each prompt.

The goal of this project is to ask the LLM three questions related to NFL player performance (discussed below) and see if the responses improve if supporting data is provided to the LLM via fine-tuning.  For each question, the following process was used:

* The question is provided to the LLM with no supporting data (intended to validate that the LLM does not have the data and cannot answer the question properly)
* The question is provided to an LLM that was fine-tuned with the data needed to answer the question

Overall the LLM without fine-tuning was unable to answer the questions, but the fine-tuned LLM was able to answer the questions with mixed accuracy.

## Discussion

### Input Data

The source data used is available [here](https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2025/1/WR.json).  This data is for week 1 of the 2025 NFL season.  Each element represents a single player with statistics about their performance in that week's game.

To prepare the fine-tuning data, the ```prep_finetune_data.py``` code was created to load the source data json into a json object.  This data was used to create a list of question prompts and answer responses, e.g.

```
question = "How many receiving yards did {player_name} have in week 1 of the 2025 NFL season?"
answer = "{player_name} had {player_recyds} receiving yards in week 1 of the 2025 NFL season."
```

This data was then transformed into ```prompt``` and ```completion``` pairs.  The secondary transformation was done to support trial-and-error exploration of different final fine-tuning data formats, as discussed in the 
[Early Iterations section](https://github.com/azlarry/model_fine_tuning_mlx_lm?tab=readme-ov-file#early-iterations).

Finally, the data was split into train, test, and validation files using the expected MLX LM data naming convention:

```
train.jsonl
test.jsonl
valid.jsonl
```

### Process

The MLX LM package conveniently includes several command line tools to perform the fine-tuning and response generation.

### Questions

The questions asked were:

***how many receiving yards did Nico Collins have in week 1 of the 2025 NFL season?***

***how many receiving yards did Ladd McConkey have in week 1 of the 2025 NFL season?***

***how many receiving yards did Rome Odunze have in week 1 of the 2025 NFL season?***

Providing the questions to the LLM without fine-tuning, the response were as follows:

```
mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --prompt "how many receiving yards did Nico Collins have in week 1 of the 2025 NFL season?" --max-tokens 4096 --temp 0                        
==========
I'm sorry, but as of my knowledge cutoff in early 2023, I don't have real-time or updated data access, including specific statistics for the 2025 NFL season. To find out how many receiving yards Nico Collins had in Week 1 of the 2025 NFL season, I recommend checking the latest statistics on a reliable sports news website, the official NFL website, or a sports statistics database like Pro Football Reference or ESPN. They regularly update player statistics for the current season.<|end|>

mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --prompt "how many receiving yards did Ladd McConkey have in week 1 of the 2025 NFL season?" --max-tokens 4096 --temp 0                        
==========
I'm sorry, but as of my knowledge cutoff in early 2023, I don't have real-time or updated data access, including specific statistics for the 2025 NFL season. To find out how many receiving yards Ladd McConkey had in Week 1 of the 2025 NFL season, I recommend checking the latest statistics on a reliable sports news website, the official NFL website, or a sports statistics database like Pro Football Reference or ESPN. They provide up-to-date player stats for each season.<|end|>

mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --prompt "how many receiving yards did Rome Odunze have in week 1 of the 2025 NFL season?" --max-tokens 4096 --temp 0                        
==========
I'm sorry, but as of my knowledge cutoff in early 2023, I don't have real-time or updated data access, including specific statistics for the 2025 NFL season. To find out how many receiving yards Rome Odunze had in Week 1 of the 2025 NFL season, I recommend checking the latest statistics on a reliable sports news website, the official NFL website, or a sports statistics database like Pro Football Reference or ESPN. They regularly update player statistics for the current season.<|end|>
```

This validates that the LLM was not trained on, and otherwise does not have access to the recent NFL player data.

Providing the same questions to the fine-tuned LLM, the responses were:

```
mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --prompt "how many receiving yards did Nico Collins have in week 1 of the 2025 NFL season?" --max-tokens 4096 --adapter-path adapters --temp 0
==========
Nico Collins had 25 receiving yards in week 1 of the 2025 NFL season.<|end|>

mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --prompt "how many receiving yards did Ladd McConkey have in week 1 of the 2025 NFL season?" --max-tokens 4096 --adapter-path adapters --temp 0
==========
Ladd McConkey had 74 receiving yards in week 1 of the 2025 NFL season.<|end|>

mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --prompt "how many receiving yards did Rome Odunze have in week 1 of the 2025 NFL season?" --max-tokens 4096 --adapter-path adapters --temp 0
==========
Rome Odunze had 0 receiving yards in week 1 of the 2025 NFL season.<|end|>
```

These answers are not all correct, as verified by manual inspection of the source ```train.jsonl``` data:

```
{"prompt":"How many receiving yards did Nico Collins have in week 1 of the 2025 NFL season?","completion":"Nico Collins had 25 receiving yards in week 1 of the 2025 NFL season."}

{"prompt":"How many receiving yards did Ladd McConkey have in week 1 of the 2025 NFL season?","completion":"Ladd McConkey had 74 receiving yards in week 1 of the 2025 NFL season."}

{"prompt":"How many receiving yards did Rome Odunze have in week 1 of the 2025 NFL season?","completion":"Rome Odunze had 37 receiving yards in week 1 of the 2025 NFL season."}
```

The Nico Collins (25 yards) and Ladd McConkey (74 yards) responses are both correct, but the Rome Odunze (0 yards) response is incorrect (the correct value is 37 yards).

The data for all three players was verified to be present in the ```train.jsonl``` file.  Due to the fact-based nature of the training data, it is assumed that if the above data was not in the ```train.jsonl``` file, the correct response would not be provided.  Ad hoc testing of players not present in the ```train.jsonl``` '0 receiving yards' responses, as assumed.

## Early Iterations

This project was initially attempted with the mistralai/Mistral-7B-Instruct-v0.2 model, but it was unable to correctly answer the questions.  Suspecting an issue with the training data format, several variations were attempted.  To aid in debugging and provide more visibility into the process, the MLX LM [```lora.py``` source code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py) was used directly instead of the command-line tool.  Unfortunately no successful outcomes were achieved, resulting in the change to the microsoft/Phi-3.5-mini-instruct model.  The training data format continues to be the primary suspected cause of the issue. 

## Conclusion and Future Work

This was an interesting project that I think clearly demonstrates the capability of LoRA fine-tuning to add new data to an LLM.

Follow-on efforts related to this work include:

* attempt the mistralai/Mistral-7B-Instruct-v0.2 model again or an updated variant, as initial attempts to use the model failed to provide any correct responses
* explore differences and trade-offs in Quantized Low-Rank Adaptation (QLoRA) vs LoRA fine-tuning approaches
* explore the many different LoRA options and the impact on performance, including # iterations, learning rate, rank, scale, etc.
* explore different types of data and larger datasets, e.g. facts vs stylistic adaptations
* investigate whether train/test/validation data sets are appropriate for fact-based training goals, and if not what approach is best
