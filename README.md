# general
The benchmark framework is divided into 3 parts; dataset construction, document ranking, and LLM generation. They are all standalone files that can be ran, but this repo has the constructed splits, and ranked files if you to go ahead in the pipeline. Refer to data/dataset_template.ipynb for an example how the data is made to ensure product, neighbor, and user size distribution. Refer to notebook/ranking.ipynb for how the files are ranked. This framework ISNT set up to run everything at once. master_generation.py was converted to CLI to run files but will require your own API key or endpoint.

# data
Includes files to construct a dataset to a PGraph Framework
GraphConstruction takes a data split JSON and forms the graph network. The data split is the needed file to run document ranking, and the graph construction is handled internally in the ranking script.

# data_split
Contains file for Ranking

# data_rank 
Returns the profile in a dictionary to run generations on based on tuned settings.

# master_generation.py
Script for creating generations on a dataset-task, using LLAMA or GPT.

## Usage
To run the script:
```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt
```
This uses GPT to generate review text on the dev split of Amazon reviews (User Product Review Generation), ranked by BM25 on all modes, on all k.


## Arguments

- `--input`: File path to the ranking file. **Required.**
  
- `--model`: Model to use for generation. **Required.**
  - Valid options:
    - `llama`: Llama-3.1-8B-Instruct
    - `gpt`: gpt-4o-mini-20240718

- `--mode`: Mode(s) to generate on. **Optional**, default performs **all** modes.
  - Valid options:
    - `none`: Retrieves nothing for the prompt.
    - `random`: Retrieves a random review from the dataset for the prompt.
    - `user`: Retrieves "user_ratings" for the prompt.
    - `neighbor`: Retrieves "neighbor_ratings" for the prompt.
    - `both`: Retrieves both "user_ratings" and "neighbor_ratings" for the prompt.


- `--k`: K-value(s) (top k retrieved reviews) to generate on. **Optional**, default performs **all** k. (`1, 2, 4`)
    - Valid options:
    - `1`
    - `2` 
    - `4`
  

## Examples

Full dataset-task generation:
```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt
```

Generation on a subset of modes (both, neighbor), (all k):
```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt --mode both neighbor 
```

Generation on a subset of k (1), (all modes):
```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt --k 1
```

Generation on a subset of modes and subset of k (none_k2, none_k4, both_k2, both_k4):
```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt --mode none both --k 2 4
```

# master_eval.py
Script for evaluating batches of OUTPUT files.

## Usage
To run the script:
```bash
python master_eval.py --ranking ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --results ./results/amazon_dev_reviewText_GPT_bm25
```
Evaluates all OUTPUT files in the given results directory against gold labels taken from given ranking file.
