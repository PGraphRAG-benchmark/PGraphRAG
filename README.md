<div align=center>
<h1>Personalized Graph-Based Retrieval for LLMs Benchmark</h1>

 [![arXiv](https://img.shields.io/badge/arXiv-2501.02157-b31b1b.svg)](https://arxiv.org/abs/2501.02157)
 [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Farxiv.org%2Fabs%2F2501.02157&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<div>
      <b>Steven Au</b><sup>1</sup>,
      <b>Cameron J. Dimacali</b><sup>1</sup>,
      <b>Ojasmitha Pedirappagari</b><sup>1</sup>,
      <b>Namyong Park</b><sup>2</sup>,
      <b>Franck Dernoncourt</b><sup>3</sup>,
      <b>Yu Wang</b><sup>4</sup>,
      <b>Nikos Kanakaris</b><sup>5</sup>,
      <b>Hanieh Deilamsalehy</b><sup>3</sup>
      <b>Ryan A. Rossi</b><sup>3</sup>,
      <b>Nesreen K. Ahmed</b><sup>6</sup>
    <div>
    <sup>1</sup>University of California Santa Cruz, <sup>2</sup>Meta AI, <sup>3</sup>Adobe Research,
<sup>4</sup>University of Oregon, <sup>5</sup>University of Southern California, <sup>6</sup>Cisco AI Research
    </div>
</div>
</div>


Personalized Graph-Based Retrieval for Large Language Models: https://arxiv.org/abs/2501.02157

As large language models (LLMs) evolve, their ability to deliver personalized and context-aware responses offers transformative potential for improving user experiences. We propose Personalized Graph-based Retrieval-Augmented Generation (PGraphRAG), a framework and benchmark that leverages user-centric knowledge graphs to enrich personalization. By directly integrating structured user knowledge into the retrieval process and augmenting prompts with user-relevant context, PGraphRAG enhances contextual understanding and output quality. This benchmark is designed to evaluate personalized text generation tasks in real-world settings where user history is sparse or unavailable. 



![ ](/pgraphrag-fig.png)



The benchmark framework is divided into three parts; dataset construction, document ranking, and LLM generation. They are all standalone files that can be executed, but this repo has the constructed splits, and ranked files if you to go ahead in the pipeline. Please refer to data/dataset_template.ipynb for an example how the data is made to ensure product, neighbor, and user size distribution. Please refer to notebook/ranking.ipynb for how the files are ranked. This framework is not set up to run everything at once. master_generation.py was converted to CLI to run files but will require your own API key or endpoint.

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

## Reference

For reference please cite the following:

```bibtex
@misc{au2025personalizedgraphbasedretrievallarge,
      title={Personalized Graph-Based Retrieval for Large Language Models}, 
      author={S. Au, C.J. Dimacali, O. Pedirappagari, N. Park, F. Dernoncourt, Y. Wang, N. Kanakaris, H. Deilamsalehy, R.A. Rossi, N.K. Ahmed},
      year={2025},
      eprint={2501.02157},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.02157}, 
}
```
