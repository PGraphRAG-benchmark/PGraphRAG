<div align=center>
<h1>Personalized Graph-Based Retrieval Benchmark for LLMs </h1>



 [![website](https://img.shields.io/badge/website-blue)](https://pgraphrag-benchmark.github.io)
[![Hugging Face Paper](https://img.shields.io/badge/HuggingFace-Paper-blue)](https://huggingface.co/papers/2501.02157)
 [![arXiv](https://img.shields.io/badge/arXiv-2501.02157-b31b1b.svg)](https://arxiv.org/abs/2501.02157)
 [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Farxiv.org%2Fabs%2F2501.02157&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<div>
      <b>Steven Au</b>,
      <b>Cameron J. Dimacali</b>,
      <b>Ojasmitha Pedirappagari</b>,
      <b>Namyong Park</b>,
      <b>Franck Dernoncourt</b>,
      <b>Yu Wang</b>,
      <b>Nikos Kanakaris</b>,
      <b>Hanieh Deilamsalehy</b>,
      <b>Ryan A. Rossi</b>,
      <b>Nesreen K. Ahmed</b>
    
</div>
</div>

<br>
<div>
Check our paper on Personalized Graph-Based Retrieval for Large Language Models at https://arxiv.org/abs/2501.02157
</div>
<br>

 <div>
As large language models (LLMs) evolve, their ability to deliver personalized and context-aware responses offers transformative potential for improving user experiences. We propose Personalized Graph-based Retrieval-Augmented Generation (PGraphRAG), a framework and benchmark that leverages user-centric knowledge graphs to enrich personalization. By directly integrating structured user knowledge into the retrieval process and augmenting prompts with user-relevant context, PGraphRAG enhances contextual understanding and output quality. This benchmark is designed to evaluate personalized text generation tasks in real-world settings where user history is sparse or unavailable. 
</div>


![ ](/pgraphrag-fig.png)

---

The benchmark framework is divided into three parts; dataset construction, document ranking, and LLM generation. They are all standalone files that can be executed, but this repo has the constructed splits, and ranked files if you to go ahead in the pipeline. Please refer to data/dataset_template.ipynb for an example of how the data is made to ensure product, neighbor, and user size distribution. Please refer to notebook/ranking.ipynb for how the files are ranked. This framework is not set up to run everything at once. master_generation.py was converted to CLI to run files but will require your own API key or endpoint.

## Table of Contents

- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
- [Data Structure](#data-structure)
  - [Data](#data)
  - [Data Split](#data-split)
  - [Data Rank](#data-rank)
- [Usage](#usage)
  - [Master Generation Script](#master-generation-script)
  - [Arguments](#arguments)
  - [Examples](#examples)
  - [Master Evaluation Script](#master-evaluation-script)
- [Reference](#reference)


---

## Getting Started

### Clone the Repository

```bash
gh repo clone PGraphRAG-benchmark/PGraphRAG
cd PGraphRAG-benchmark/PGraphRAG

### Install Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.tx
```
Note this is not necessary to run you own LLM models, we ran Llama-3.1-8b-instruct on our own hardware and GPT-4o-mini through Azure cloud services.

---

## Data Structure

### Data

Includes files to construct a dataset for the PGraph Framework. The `GraphConstruction` script processes a data split JSON and forms the graph network. This is a required step to run document ranking, and the graph construction is handled internally in the ranking script.

### Data Split

Contains files for ranking.

### Data Rank

Returns the profile in a dictionary to run generations on based on tuned settings.

---

## Usage

### Master Generation Script

The `master_generation.py` script is used to generate outputs for dataset tasks using LLMs like Llama-3.1-8B-Instruct or GPT.

#### To run the script:

```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt
```

This example uses GPT to generate review text on the dev split of Amazon reviews (User Product Review Generation), ranked by BM25 on all modes and all k values.

---
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


- `--k`: K-value(s) (top k retrieved reviews) to generate on. **Optional**, default performs **all** k (`1, 2, 4`).

### Examples

#### Full dataset-task generation:

```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt
```

#### Generation on a subset of modes (`both`, `neighbor`), (all k):

```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt --mode both neighbor
```

#### Generation on a subset of k (`1`), (all modes):

```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt --k 1
```

#### Generation on a subset of modes and subset of k (`none_k2`, `none_k4`, `both_k2`, `both_k4`):

```bash
python master_generation.py --input ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --model gpt --mode none both --k 2 4
```

---

### Master Evaluation Script

The `master_eval.py` script evaluates batches of output files.

#### To run the script:

```bash
python master_eval.py --ranking ./data/Rankings/Amazon/amazon_dev_reviewText_bm25.json --results ./results/amazon_dev_reviewText_GPT_bm25
```

This evaluates all output files in the given results directory against gold labels taken from the specified ranking file.

---
## Reference

For reference please cite the following:

```bibtex
@misc{pgraphrag,
      title={Personalized Graph-Based Retrieval for Large Language Models}, 
      author={S. Au, C.J. Dimacali, O. Pedirappagari, N. Park, F. Dernoncourt, Y. Wang, N. Kanakaris, H. Deilamsalehy, R.A. Rossi, N.K. Ahmed},
      year={2025},
      eprint={2501.02157},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.02157}, 
}
```
