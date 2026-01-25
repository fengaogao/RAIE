# RAIE

## 1. Introduction
In this work, we aim to make continual recommendation updates more stable and scalable by moving from global fine-tuning to region-level editing. We propose Region-Aware Incremental Editing (RAIE), which freezes the backbone and updates semantically coherent regions with dedicated adapters for confident routing and precise adaptation.

## 2. Environment Requirement
Below are the key packages used across the RAIE codebase with versions from your environment:

- Python 3.9.23
- torch 2.8.0
- numpy 2.0.2
- tqdm 4.67.1
- transformers 4.56.0
- peft 0.17.1
- scikit-learn 1.6.1
- faiss-cpu 1.12.0
- pandas 2.3.2

## 3. File descriptions and run examples (using RAIEmodel)

- `SASRec_RAIE.py`: RAIE training and incremental editing flow for SASRec, including region construction and routing.
  - Example: `python SASRec_RAIE.py --mode raie --data_dir ./data/ --output_dir ./runs/RAIEmodel_sasrec --maxlen 50 --hidden_units 128`

- `TiSASRec_RAIE.py`: RAIE implementation for TiSASRec with time-interval modeling in sequential recommendation.
  - Example: `python TiSASRec_RAIE.py --mode raie --data_dir ./data/ --output_dir ./runs/RAIEmodel_tisasrec --maxlen 50 --hidden_units 128`

- `Bert4Rec_RAIE.py`: RAIE pipeline for Bert4Rec using an MLM objective for sequential recommendation and region updates.
  - Example: `python Bert4Rec_RAIE.py --mode raie --data_dir ./data/ --output_dir ./runs/RAIEmodel_bert4rec --maxlen 50 --hidden_units 128`

- `openP5_RAIE.py`: RAIE for LLM-based recommendation with openP5-style prompt/target data.
  - Example: `python openP5_RAIE.py --mode raie --data_dir ./data/ --output_dir ./runs/RAIEmodel_openp5 --model_name_or_path RAIEmodel`

- `E-BPR.py`: E-BPR editing/retraining script based on SASRec representations for mining or applying edit pairs.
  - Example: `python E-BPR.py --data_dir ./data/ --model_dir ./runs/RAIEmodel_sasrec --output_dir ./runs/RAIEmodel_ebpr --model_py SASRec_RAIE.py --edit_source from_topk`

- `ml_data_load.py`: Utility script for loading data and constructing sequences.
  - Example: `python ml_data_load.py --data_dir ./data/ --output_dir ./runs/RAIEmodel_dataprep --max_len 50`

## 4. Dataset

### 4.1 `ml_data_load.py` outputs

- `original.jsonl`: Sliding-window prompt/target pairs from the original (O) time block.
- `original_stride1.jsonl`: Same as `original.jsonl`, but generated with stride 1 for dense routing.
- `finetune.jsonl`: Sliding-window prompt/target pairs from the finetune (F) time block.
- `test.jsonl`: Sliding-window prompt/target pairs from the test (T) time block.
- `item_ids.json`: All item IDs observed in the generated splits.
- `meta.json`: Counts, time boundaries (`t^s`, `t^F`), and schema metadata.

### 4.2 Download links

- MovieLens 10M: https://grouplens.org/datasets/movielens/10m/
- Yelp Open Dataset: https://www.yelp.com/dataset
