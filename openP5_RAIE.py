import sys
import os
import gc
import math
import json
import random
import argparse
import re
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
from collections import defaultdict

# ---------- Environment flags: avoid tokenizers fork deadlock and CUDA fragmentation ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Optional PEFT LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception as e:
    PEFT_AVAILABLE = False
    _PEFT_ERR = e

# Optional spectral clustering (CID)
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

_TOKEN_RE = re.compile(r"<item_(\d+)>")

# ------------------------------
# Utils & DDP
# ------------------------------
def is_distributed_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1

def init_distributed():
    if is_distributed_env():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://",
                                timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

_GLOO_GROUP = None
def _get_gloo_group():
    global _GLOO_GROUP
    if _GLOO_GROUP is None:
        if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
            _GLOO_GROUP = dist.new_group(backend='gloo')
        else:
            _GLOO_GROUP = None
    return _GLOO_GROUP


def bcast_object(obj, src=0, use_gloo=True):
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    group = _get_gloo_group() if (use_gloo and dist.get_backend() == "nccl") else None
    lst = [obj]
    dist.broadcast_object_list(lst, src=src, group=group)
    return lst[0]

def cleanup_distributed():
    if is_distributed_env() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def set_seed(seed: int = 42, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def aggressive_gc():
    """Perform aggressive garbage collection to reduce GPU fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _free_cuda(*objs):
    """Recursively release objects (including .module) and clear caches."""
    for o in objs:
        try:
            if o is None:
                continue
            if hasattr(o, "module"):
                try:
                    delattr(o, "module")
                except Exception:
                    pass
            del o
        except Exception:
            pass
    aggressive_gc()

# ==============================
# Load original/test/finetune JSONL + item_ids.json
# ==============================
def _tok_to_mid(tok: str) -> int:
    tok = tok.strip()
    m = _TOKEN_RE.fullmatch(tok)
    if m:
        return int(m.group(1))
    tok = tok.lstrip("<").rstrip(">")
    if tok.startswith("item_"):
        tok = tok[len("item_"):]
    return int(tok)

def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def _build_user_pairs_from_prompt_target(rows: List[dict]) -> Dict[int, List[Tuple[int, int]]]:
    per_user: Dict[int, List[Tuple[int, int]]] = {}
    user_next_ts: Dict[int, int] = {}
    for r in rows:
        try:
            uid = int(r.get("user_id", 0))
        except Exception:
            uid = 0
        pr = (r.get("prompt") or "").strip()
        tg = (r.get("target") or "").strip()
        if not pr or not tg:
            continue
        try:
            mids_ctx = [_tok_to_mid(t) for t in pr.split()]
            mid_tgt = _tok_to_mid(tg)
        except Exception:
            continue
        seq = mids_ctx + [mid_tgt]
        user_next_ts.setdefault(uid, 0)
        for m in seq:
            per_user.setdefault(uid, []).append((m, user_next_ts[uid]))
            user_next_ts[uid] += 1

    for u in per_user:
        seq = sorted(per_user[u], key=lambda x: x[1])
        dedup, seen = [], set()
        for mid, ts in seq:
            if (mid, ts) in seen:
                continue
            seen.add((mid, ts)); dedup.append((mid, ts))
        per_user[u] = dedup
    return per_user

def _read_vocab(item_ids_path: str) -> List[int]:
    with open(item_ids_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "item_ids" in obj:
        item_ids = obj["item_ids"]
    else:
        item_ids = obj
    return [int(x) for x in item_ids]

# ---------- Collaborative item indexing ----------
def _build_base_map_collaborative(item_ids: List[int],
                                  train_pairs: Dict[int, List[Tuple[int, int]]]
                                  ) -> Dict[int, int]:
    """
    Collaborative indexing: sort by descending training frequency, then by earliest
    first-seen time, then by original item_id for tie-breaking. This only changes
    the mid -> [0..N-1] mapping order and keeps the rest of the logic intact.
    """
    pop: Dict[int, int] = defaultdict(int)
    first_ts: Dict[int, int] = {}
    for pairs in train_pairs.values():
        for m, ts in pairs:
            pop[m] += 1
            if m not in first_ts:
                first_ts[m] = ts
    # Items missing from training (only in test) use zero popularity and max timestamp.
    big_ts = 1 << 62
    def sort_key(m):
        return (-pop.get(m, 0), first_ts.get(m, big_ts), m)
    items = sorted(item_ids, key=sort_key)
    return {m: i for i, m in enumerate(items)}

def _build_base_map(item_ids: List[int],
                    train_pairs: Dict[int, List[Tuple[int, int]]],
                    method: str = "sequential",
                    seed: int = 42) -> Dict[int, int]:
    if method == "random":
        items = list(item_ids)
        rng = random.Random(seed)
        rng.shuffle(items)
        return {m: i for i, m in enumerate(items)}

    if method == "collaborative":
        return _build_base_map_collaborative(item_ids, train_pairs)

    # Sequential: sort by first appearance time in training.
    first_ts: Dict[int, int] = {}
    for pairs in train_pairs.values():
        for m, ts in pairs:
            if m not in first_ts:
                first_ts[m] = ts
    def sort_key(m):
        if m in first_ts:
            return (0, first_ts[m], m)
        else:
            return (1, 1 << 62, m)
    items = sorted(item_ids, key=sort_key)
    return {m: i for i, m in enumerate(items)}

def _pairs_to_index_sequences(pairs: Dict[int, List[Tuple[int, int]]],
                              mid2idx: Dict[int, int]) -> Dict[int, List[int]]:
    u2s: Dict[int, List[int]] = {}
    for u, lst in pairs.items():
        idx_seq = []
        for mid, _ts in lst:
            if mid in mid2idx:
                idx_seq.append(mid2idx[mid])
        u2s[u] = idx_seq
    return u2s

def load_preprocessed_splits(data_dir: str,
                             item_indexing: str = "sequential",
                             seed: int = 42,
                             min_user_len: int = 5,
                             train_jsonl_path: str = None,
                             item_ids_path: str = None,
                             test_jsonl_path: str = None):
    path_train = train_jsonl_path or os.path.join(data_dir, "original_stride1.jsonl")
    path_vocab = item_ids_path or os.path.join(data_dir, "item_ids.json")
    if not os.path.exists(path_train): raise FileNotFoundError(path_train)
    if not os.path.exists(path_vocab): raise FileNotFoundError(path_vocab)

    rows_train = _load_jsonl(path_train)
    train_pairs = _build_user_pairs_from_prompt_target(rows_train)

    if test_jsonl_path and os.path.exists(test_jsonl_path):
        rows_test = _load_jsonl(test_jsonl_path)
        test_pairs = _build_user_pairs_from_prompt_target(rows_test)
    else:
        test_pairs = {}

    item_ids = _read_vocab(path_vocab)
    if item_indexing not in ("sequential", "random", "collaborative"):
        raise ValueError(f"unknown item_indexing: {item_indexing}")

    method = "random" if item_indexing == "random" else ("collaborative" if item_indexing == "collaborative" else "sequential")
    base_map = _build_base_map(item_ids, train_pairs, method=method, seed=seed)

    train_seq = _pairs_to_index_sequences(train_pairs, base_map)
    test_seq = _pairs_to_index_sequences(test_pairs, base_map) if test_pairs else {}

    def _filter(u2s: Dict[int, List[int]], k: int):
        return {u: s for u, s in u2s.items() if len(s) >= k}
    train_seq = _filter(train_seq, min_user_len)
    test_seq  = _filter(test_seq,  min_user_len) if test_seq else {}
    val_seq   = {u: s[:] for u, s in train_seq.items() if len(s) >= 2}

    n_items = len(item_ids)
    return train_seq, val_seq, test_seq, n_items, base_map

# ------------------------------
# Item tokenization (OpenP5-style)
# ------------------------------
def make_item_final_tokens(n_items: int) -> List[str]:
    return [f"<item_{i}>" for i in range(n_items)]

# ------------------------------
# Dataset / Collate
# ------------------------------
def collate_batch(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, val):
        return seq + [val] * (max_len - len(seq))
    input_ids = torch.tensor([pad(x["input_ids"].tolist(), pad_id) for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long)
    labels = torch.tensor([pad(x["labels"].tolist(), -100) for x in batch], dtype=torch.long)
    target_item_idx = torch.stack([x["target_item_idx"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "target_item_idx": target_item_idx}

class SeqRecDatasetFromPairs(Dataset):
    def __init__(self, jsonl_path: str, mid2idx: Dict[int,int],
                 tokenizer, item_token_seq: List[List[str]], max_history: int = 50,
                 wrap_mode: str = 'none',
                 final_token_only_loss: bool = True,
                 deterministic_template: bool = False):
        assert wrap_mode in ('none',)
        self.examples = []
        self.tok = tokenizer
        self.item_token_seq = item_token_seq
        self.max_history = max_history
        self.final_token_only_loss = final_token_only_loss

        rows = _load_jsonl(jsonl_path)
        for r in rows:
            pr = (r.get("prompt") or "").strip()
            tg = (r.get("target") or "").strip()
            if not pr or not tg:
                continue
            try:
                mids_ctx = [_tok_to_mid(t) for t in pr.split()]
                mid_tgt  = _tok_to_mid(tg)
            except Exception:
                continue
            idx_ctx = [mid2idx[m] for m in mids_ctx if m in mid2idx]
            if not idx_ctx or (mid_tgt not in mid2idx):
                continue
            idx_ctx = idx_ctx[-max_history:]
            idx_tgt = mid2idx[mid_tgt]

            hist_tokens = []
            for i in idx_ctx:
                hist_tokens.extend(item_token_seq[i])
            history_text = " ".join(hist_tokens)
            prompt_text = history_text
            answer_text = " ".join(item_token_seq[idx_tgt])

            enc_prompt = self.tok(prompt_text, add_special_tokens=True)
            enc_answer = self.tok(answer_text, add_special_tokens=False)
            if len(enc_answer["input_ids"]) == 0:
                continue

            input_ids = enc_prompt["input_ids"] + enc_answer["input_ids"]
            attention_mask = enc_prompt["attention_mask"] + enc_answer["attention_mask"]

            if self.final_token_only_loss:
                lab = [-100] * len(enc_prompt["input_ids"])
                lab += [-100] * (len(enc_answer["input_ids"]) - 1)
                lab += [enc_answer["input_ids"][-1]]
            else:
                lab = [-100] * len(enc_prompt["input_ids"]) + enc_answer["input_ids"]

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "target_item_idx": torch.tensor(idx_tgt, dtype=torch.long),
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

# ------------------------------
# Metrics & CE on items
# ------------------------------
def recall_ndcg_from_scores(scores: torch.Tensor, gold_idx: torch.Tensor, k_list=(5,10,20)):
    max_k = max(k_list)
    topk = torch.topk(scores, k=max_k, dim=1).indices  # [B, maxK]
    labels_exp = gold_idx.view(-1,1).expand(-1, max_k)
    match = (topk == labels_exp).float()
    out = {}
    for k in k_list:
        m = match[:, :k]
        hit = m.max(1).values
        pos = torch.argmax(m, 1)
        dcg = hit * (1.0 / torch.log2(pos.float() + 2.0))
        out[f"Recall@{k}"] = float(hit.mean().item())
        out[f"NDCG@{k}"] = float(dcg.mean().item())
    return out

def _last_label_pos(labels: torch.Tensor) -> torch.Tensor:
    mask = (labels != -100).to(torch.int64)
    rev  = torch.flip(mask, dims=[1])
    last_from_right = rev.argmax(dim=1)
    return labels.size(1) - 1 - last_from_right

@torch.no_grad()
def evaluate_causal(model, loader, device, item_final_token_ids: List[int], k_list=(5,10,20), fp16=False, desc="Eval"):
    model.eval()
    item_ids = torch.tensor(item_final_token_ids, device=device)
    sums = {f"Recall@{k}": 0.0 for k in k_list} | {f"NDCG@{k}": 0.0 for k in k_list}
    N = 0
    amp_ctx = torch.amp.autocast("cuda", enabled=fp16) if torch.cuda.is_available() else torch.no_grad()
    for batch in tqdm(loader, desc=desc):
        inp = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["target_item_idx"].to(device, non_blocking=True)
        with amp_ctx:
            out = model(input_ids=inp, attention_mask=attn)
            logits = out.logits  # [B, L, V]
        last_label_pos = _last_label_pos(batch["labels"].to(device))
        query_pos = (last_label_pos - 1).clamp(min=0)
        rows = torch.arange(inp.size(0), device=device)
        logits_q = logits[rows, query_pos, :]              # [B, V]
        scores = logits_q.index_select(dim=1, index=item_ids)  # [B, N_items]
        m = recall_ndcg_from_scores(scores, lab, k_list)
        for k in k_list:
            sums[f"Recall@{k}"] += m[f"Recall@{k}"] * inp.size(0)
            sums[f"NDCG@{k}"] += m[f"NDCG@{k}"] * inp.size(0)
        N += int(inp.size(0))
    return {k: (v / max(1, N)) for k, v in sums.items()}


def _adapter_final_logits_causal(model, adapter_name: str, input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor, labels: torch.Tensor,
                                 item_final_token_ids: List[int]):
    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter(adapter_name)
        except Exception:
            pass
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, L, V]
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    last_label_pos = _last_label_pos(labels)
    query_pos = (last_label_pos - 1).clamp(min=0)
    logits_q = logits[rows, query_pos, :]
    V = logits_q.size(-1)
    mask_vec = torch.full((V,), float('-inf'), device=input_ids.device)
    mask_vec[item_final_token_ids] = 0.0
    return logits_q + mask_vec


@torch.no_grad()
def evaluate_lsat_causal(
    model,
    loader: DataLoader,
    device,
    item_final_token_ids: List[int],
    long_adapter: str,
    short_adapter: str,
    alpha: float = 0.5,
    k_list=(5, 10, 20),
):
    model.eval()
    alpha = float(max(0.0, min(1.0, alpha)))
    sums = {f"Recall@{k}": 0.0 for k in k_list} | {f"NDCG@{k}": 0.0 for k in k_list}
    N = 0
    for batch in tqdm(loader, desc='Eval-LSAT'):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['target_item_idx'].to(device, non_blocking=True)

        logits_long = _adapter_final_logits_causal(model, long_adapter, input_ids, attention_mask, batch['labels'].to(device), item_final_token_ids)
        logits_short = _adapter_final_logits_causal(model, short_adapter, input_ids, attention_mask, batch['labels'].to(device), item_final_token_ids)
        logits_final = alpha * logits_long + (1.0 - alpha) * logits_short

        item_ids = torch.tensor(item_final_token_ids, device=device)
        scores = logits_final.index_select(dim=1, index=item_ids)
        m = recall_ndcg_from_scores(scores, labels, k_list)
        for k in k_list:
            sums[f"Recall@{k}"] += m[f"Recall@{k}"] * input_ids.size(0)
            sums[f"NDCG@{k}"] += m[f"NDCG@{k}"] * input_ids.size(0)
        N += int(input_ids.size(0))
    return {k: (v / max(1, N)) for k, v in sums.items()}

def ce_loss_on_items(model, batch, device, item_final_token_ids: List[int], fp16=False):
    inp = batch["input_ids"].to(device, non_blocking=True)
    attn = batch["attention_mask"].to(device, non_blocking=True)
    lab = batch["target_item_idx"].to(device, non_blocking=True)
    amp_ctx = torch.amp.autocast("cuda", enabled=fp16) if torch.cuda.is_available() else torch.no_grad()
    with amp_ctx:
        out = model(input_ids=inp, attention_mask=attn)
        logits = out.logits
    last_label_pos = _last_label_pos(batch["labels"].to(device))
    query_pos = (last_label_pos - 1).clamp(min=0)
    rows = torch.arange(inp.size(0), device=device)
    logits_q = logits[rows, query_pos, :]
    item_ids = torch.tensor(item_final_token_ids, device=device)
    scores = logits_q.index_select(dim=1, index=item_ids)
    loss = F.cross_entropy(scores.float(), lab)
    return loss, scores

# ------------------------------
# Training (O stage)
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, device,
                    item_final_token_ids: List[int],
                    fp16=False, is_main=True, grad_clip: float = 1.0,
                    label_smoothing: float = 0.0):
    model.train()
    total = 0.0
    use_amp = fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    item_ids = torch.tensor(item_final_token_ids, device=device)

    for batch in tqdm(dataloader, desc="Train(O)", disable=not is_main):
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits
            last_label_pos = _last_label_pos(batch["labels"])
            query_pos = (last_label_pos - 1).clamp(min=0)
            rows = torch.arange(logits.size(0), device=device)
            logits_q = logits[rows, query_pos, :]
            scores = logits_q.index_select(dim=1, index=item_ids)
            gold   = batch["target_item_idx"]
            try:
                loss = F.cross_entropy(scores.float(), gold, label_smoothing=label_smoothing)
            except TypeError:
                loss = F.cross_entropy(scores.float(), gold)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
    return total / max(1, len(dataloader))

# ------------------------------
# LoRA F-stage plugin: Replay
# ------------------------------
class ReplayBuffer:
    def __init__(self, dataset: Dataset, collate_fn, batch_size: int, sampler=None):
        self.sampler = sampler
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=(sampler is None), sampler=sampler,
                                 num_workers=0, collate_fn=collate_fn, pin_memory=True)
        self.it = iter(self.loader)
    def set_epoch(self, epoch: int):
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
    def next_batch(self):
        try: return next(self.it)
        except StopIteration:
            self.it = iter(self.loader); return next(self.it)

@torch.no_grad()
def sample_negatives(labels: torch.Tensor, n_items: int, device=None):
    B = labels.size(0)
    out = torch.empty(B, dtype=torch.long, device=device or labels.device)
    for i in range(B):
        tgt = int(labels[i].item())
        while True:
            cand = random.randrange(0, n_items)
            if cand != tgt: break
        out[i] = cand
    return out

def lwf_kd_loss_causal(
    student,
    teacher,
    batch: dict,
    device,
    item_final_token_ids: List[int],
    T: float = 2.0,
    alpha: float = 0.5,
    fp16: bool = False,
):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["target_item_idx"].to(device, non_blocking=True)
    neg_idx = sample_negatives(labels, len(item_final_token_ids), device=device)
    item_tokens = torch.tensor(item_final_token_ids, device=device)
    cand_tokens = torch.stack([
        item_tokens[labels],
        item_tokens[neg_idx],
    ], dim=1)

    rows = torch.arange(input_ids.size(0), device=device)
    last_label_pos = _last_label_pos(batch["labels"].to(device))
    query_pos = (last_label_pos - 1).clamp(min=0)

    with torch.amp.autocast("cuda", enabled=fp16 and torch.cuda.is_available()):
        with torch.no_grad():
            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            t_logits = t_out.logits  # [B, L, V]
            t_final = t_logits[rows, query_pos, :] / T
            t_cand = t_final.gather(1, cand_tokens)
            t_prob = F.softmax(t_cand, dim=-1)

        s_out = student(input_ids=input_ids, attention_mask=attention_mask)
        s_logits = s_out.logits
        s_final = s_logits[rows, query_pos, :] / T
        s_cand = s_final.gather(1, cand_tokens)
        s_log_prob = F.log_softmax(s_cand, dim=-1)

    kd = F.kl_div(s_log_prob, t_prob, reduction='batchmean') * (T * T)
    return alpha * kd

def finetune_one_epoch_lora(
    lora_model, finetune_loader, device,
    item_final_token_ids: List[int],
    optimizer, scheduler=None, fp16=False,
    plugin="none", replay_buf: Optional[ReplayBuffer]=None,
    replay_ratio=0.3, grad_clip=1.0, log_every=100,
    teacher=None, lwf_T: float = 2.0, lwf_alpha: float = 0.5,
):
    lora_model.train()
    use_amp = fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total, nstep = 0.0, 0
    for step, batch in enumerate(tqdm(finetune_loader, desc=f"Finetune(F)-{plugin}"), start=1):
        loss, _ = ce_loss_on_items(lora_model, batch, device, item_final_token_ids, fp16=use_amp)
        tot = loss
        if (plugin.find("replay") >= 0) and (replay_buf is not None) and (replay_ratio > 0):
            rb = replay_buf.next_batch()
            rloss, _ = ce_loss_on_items(lora_model, rb, device, item_final_token_ids, fp16=use_amp)
            tot = (1.0 - replay_ratio) * tot + replay_ratio * rloss

        kd_loss = torch.tensor(0.0, device=device)
        if (plugin.find("lwf") >= 0) and (teacher is not None):
            kd_loss = lwf_kd_loss_causal(
                lora_model, teacher, batch, device, item_final_token_ids,
                T=lwf_T, alpha=lwf_alpha, fp16=use_amp
            )
            tot = tot + kd_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(tot).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            tot.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None: scheduler.step()

        total += float(tot.detach().item()); nstep += 1
        if (log_every > 0) and (step % log_every == 0):
            print(f"[F] step={step} loss={float(tot.detach().item()):.4f} kd={float(kd_loss.detach().item()):.4f}")
    return total / max(1, nstep)


class _MoLEOutput:
    def __init__(self, logits, loss=None, gate=None):
        self.logits = logits
        self.loss = loss
        self.gate = gate


class MoLEMixtureCausal(nn.Module):
    def __init__(
        self,
        base_model,
        adapter_names: List[str],
        item_final_token_ids: List[int],
        gating_hidden: int = 128,
        temperature: float = 0.7,
        balance_coef: float = 0.01,
    ):
        super().__init__()
        self.model = base_model.module if hasattr(base_model, "module") else base_model
        self.adapter_names = adapter_names
        self.item_final_token_ids = item_final_token_ids
        self.temperature = temperature
        self.balance_coef = balance_coef

        config = getattr(self.model, "config", None)
        H = getattr(config, "hidden_size", None) if config is not None else None
        if H is None:
            H = getattr(config, "d_model", 768) if config is not None else 768
        self.gate = nn.Sequential(
            nn.Linear(H, gating_hidden),
            nn.Tanh(),
            nn.Linear(gating_hidden, len(adapter_names))
        ).to(dtype=torch.float32)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.set_grad_enabled(self.training):
            emb = self.model.get_input_embeddings()(input_ids)
            mask = attention_mask.unsqueeze(-1)
            mask_sum = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            pooled = (emb * mask).sum(dim=1) / mask_sum  # [B, D]
            gate_logits = self.gate(pooled) / self.temperature
            gate_weights = F.softmax(gate_logits, dim=-1)

        logits_list = []
        for name in self.adapter_names:
            if hasattr(self.model, "set_adapter"):
                try:
                    self.model.set_adapter(name)
                except Exception:
                    pass
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if not torch.isfinite(out.logits).all():
                bad = (~torch.isfinite(out.logits)).any().item()
                if bad:
                    print(f"[DBG-X][NON-FINITE LOGITS] adapter={name}")
            logits_list.append(out.logits)

        logits_stack = torch.stack(logits_list, dim=1).to(dtype=gate_weights.dtype)  # [B, E, L, V]
        mix_logits = torch.einsum('be,belv->blv', gate_weights, logits_stack)


        loss = None
        if labels is not None:
            rows = torch.arange(input_ids.size(0), device=input_ids.device)
            last_label_pos = _last_label_pos(labels)
            query_pos = (last_label_pos - 1).clamp(min=0)
            logits_q = mix_logits[rows, query_pos, :]

            target_labels = labels[rows, last_label_pos]

            V = logits_q.size(-1)
            mask_vec = torch.full((V,), float('-inf'), device=input_ids.device)
            mask_vec[self.item_final_token_ids] = 0.0
            logits_q = logits_q + mask_vec

            base_loss = F.cross_entropy(logits_q.float(), target_labels)
            balance_loss = torch.tensor(0.0, device=input_ids.device)
            if self.balance_coef > 0:
                avg_gate = gate_weights.mean(dim=0)
                uniform = torch.full_like(avg_gate, 1.0 / len(self.adapter_names))
                balance_loss = self.balance_coef * F.kl_div((avg_gate + 1e-8).log(), uniform, reduction='batchmean')
            loss = base_loss + balance_loss

        return _MoLEOutput(logits=mix_logits, loss=loss, gate=gate_weights)


def finetune_one_epoch_mole(
    mole_model: MoLEMixtureCausal,
    finetune_loader: DataLoader,
    optimizer,
    scheduler,
    device,
    grad_clip: float = 1.0,
    fp16: bool = False,
    log_every: int = 100,
):
    mole_model.train()
    use_amp = fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss, n_steps = 0.0, 0
    for step, batch in enumerate(tqdm(finetune_loader, desc="Finetune(F)-MoLE"), start=1):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = mole_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            # ===== [DBG-Y] find the first step that corrupts default adapter =====
            if (loss is None) or (not torch.isfinite(loss)):
                print(f"[DBG-Y][NON-FINITE LOSS] step={step} loss={loss}")

            # Extra safety: log if mix_logits becomes non-finite.
            if not torch.isfinite(out.logits).all():
                print(f"[DBG-Y][NON-FINITE MIX_LOGITS] step={step}")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mole_model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mole_model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach().item())
        n_steps += 1
        if (log_every > 0) and (step % log_every == 0):
            print(f"[MoLE] step={step} loss={float(loss.detach().item()):.4f}")

    return total_loss / max(1, n_steps)

# =========================================================
# ================ RAIE ===================================
# =========================================================
def l2n(X: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return (X / n).astype(np.float32)

def spherical_kmeans(X: np.ndarray, K: int, niter=30, seed=42):
    try:
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=K, niter=niter, nredo=2, verbose=False, spherical=True, seed=seed)
        km.train(X.astype(np.float32))
        C = km.centroids.astype(np.float32)
        index = faiss.IndexFlatIP(X.shape[1]); index.add(C)
        _, lab = index.search(X, 1)
        return C, lab.reshape(-1).astype(np.int32)
    except Exception:
        skm = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=seed)
        lab = skm.fit_predict(X)
        C = l2n(skm.cluster_centers_)
        return C.astype(np.float32), lab.astype(np.int32)

@torch.no_grad()
def encode_prompts_to_vecs_causallm(model, ds: Dataset, device, tok: AutoTokenizer, batch_size=256, use_mean_pool=True):
    model.eval()
    vecs = []
    for i in tqdm(range(0, len(ds), batch_size), desc="Enc(prompts)"):
        sub = [ds[j] for j in range(i, min(i+batch_size, len(ds)))]
        if len(sub) == 0: break
        maxL = max(int(r["input_ids"].size(0)) for r in sub)
        pad_id = tok.pad_token_id or tok.eos_token_id
        input_ids = torch.full((len(sub), maxL), pad_id, dtype=torch.long)
        attn = torch.zeros((len(sub), maxL), dtype=torch.long)
        for j, r in enumerate(sub):
            L = int(r["input_ids"].size(0))
            input_ids[j, :L] = r["input_ids"]
            attn[j, :L] = r["attention_mask"]
        input_ids = input_ids.to(device); attn = attn.to(device)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]  # [B, L, H]
        if use_mean_pool:
            mask = attn.unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            v = h[:, -1, :]
        vecs.append(v.detach().float().cpu().numpy())
    if len(vecs) == 0:
        H = getattr(model.config, "hidden_size", 768)
        return np.zeros((0, H), np.float32)
    X = np.concatenate(vecs, 0).astype(np.float32)
    return l2n(X)

def per_cluster_radius(X, C, labels, q=0.9):
    K = C.shape[0]; R = np.zeros(K, np.float32)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0: continue
        dots = X[idx] @ C[k]
        ang = np.arccos(np.clip(dots, -1, 1))
        R[k] = np.quantile(ang, q).astype(np.float32)
    return R

def per_cluster_ang_std(X, C, labels):
    K = C.shape[0]; sig = np.zeros(K, np.float32)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0: continue
        dots = X[idx] @ C[k]
        ang = np.arccos(np.clip(dots, -1, 1))
        sig[k] = ang.std().astype(np.float32)
    return sig

def finetune_one_epoch_causal_anchor(
    model, finetune_loader, device, item_final_token_ids: List[int],
    optimizer, scheduler=None, grad_clip=1.0,
    E0=None, tune_token_ids: Optional[torch.Tensor]=None, lambda_anchor: float=0.0, fp16=False,
    embed_model=None
):
    """
    DDP-safe anchor finetuning: forward/backward uses the (possibly DDP-wrapped)
    model, while embedding regularization/hooks use the unwrapped embed_model.
    """
    model.train()
    use_amp = fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total, nstep = 0.0, 0
    for batch in tqdm(finetune_loader, desc="Finetune(F)-Anchor"):
        loss, _ = ce_loss_on_items(model, batch, device, item_final_token_ids, fp16=use_amp)
        if lambda_anchor > 0.0 and (E0 is not None) and (tune_token_ids is not None) and tune_token_ids.numel()>0:
            m_embed = embed_model if embed_model is not None else model
            Ein = m_embed.get_input_embeddings().weight
            reg = (Ein.index_select(0, tune_token_ids) - E0.index_select(0, tune_token_ids)).pow(2).mean()
            loss = loss + lambda_anchor * reg

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None: scheduler.step()

        total += float(loss.detach().item()); nstep += 1
    return total / max(1, nstep)

def save_clean_base_with_new_tokens(peft_wrapped, tokenizer, out_dir, orig_base_path):
    ensure_dir(out_dir)
    with torch.no_grad():
        trained_E = peft_wrapped.get_input_embeddings().weight.detach().cpu().clone()

    base_clean = AutoModelForCausalLM.from_pretrained(orig_base_path, low_cpu_mem_usage=True, torch_dtype=trained_E.dtype)
    base_clean.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    with torch.no_grad():
        Ein = base_clean.get_input_embeddings().weight
        if Ein.shape != trained_E.shape:
            raise RuntimeError(f"Embedding shape mismatch: clean={tuple(Ein.shape)} vs trained={tuple(trained_E.shape)}")
        Ein.copy_(trained_E)

        Eout = base_clean.get_output_embeddings()
        if (Eout is not None) and (Eout is not base_clean.get_input_embeddings()):
            if Eout.weight.shape == trained_E.shape:
                Eout.weight.copy_(trained_E)

    base_clean.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

# =========================================================
# ================ RAIE: RegionBank (modified here) =======
# =========================================================
class RegionBank:
    def __init__(self, model, tok: AutoTokenizer, item_final_token_ids: List[int],
                 original_ds: Optional[Dataset],
                 device, out_dir: str,
                 K=3, q=0.9, T_low=0.7, T_high=0.9, tau=0.05, gamma=0.5, gap_thr=0.02,
                 lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")):
        self.model = model
        self.tok = tok
        self.item_final_token_ids = item_final_token_ids
        self.original_ds = original_ds
        self.device = device
        self.out_dir = out_dir

        self.S = None; self.n = None
        self.kappa = None; self.pi = None
        self.C0 = None
        self.K = K; self.q = q
        self.beta_post = 8.0
        self.delta_min = 0.05
        self.T_low = T_low; self.T_high = T_high
        self.tau = tau; self.gamma = gamma; self.gap_thr = gap_thr
        self.n0 = 5.0
        self.kappa_ema = 0.2

        self.C = None; self.R = None; self.sig = None
        self.orig_idx_by_k = {}

        self.lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM, inference_mode=False,
            target_modules=list(target_modules)
        )

    # ---------- Utility: sync stats shape (supports dynamic K) ----------
    def _sync_priors_shape(self):
        K, H = self.S.shape
        if self.C0 is None:
            self.C0 = self.C.copy().astype(np.float32)
        if self.C0.shape[0] < K:
            S_ex = self.S[self.C0.shape[0]:K, :]
            norms = np.linalg.norm(S_ex, axis=1, keepdims=True)
            extra = np.zeros_like(S_ex, dtype=np.float32)
            ok = (norms.squeeze(-1) > 1e-12)
            if ok.any():
                extra[ok] = (S_ex[ok] / (norms[ok] + 1e-12)).astype(np.float32)
            mean_mu = self.C.mean(0, keepdims=True) if (self.C is not None and self.C.shape[0]>0) else np.zeros((1, S_ex.shape[1]), np.float32)
            mean_mu = mean_mu / (np.linalg.norm(mean_mu, axis=1, keepdims=True) + 1e-12)
            for r in range(extra.shape[0]):
                if not np.isfinite(extra[r]).all() or np.linalg.norm(extra[r]) < 1e-9:
                    extra[r] = mean_mu
            self.C0 = np.vstack([self.C0, extra.astype(np.float32)])
        elif self.C0.shape[0] > K:
            self.C0 = self.C0[:K, :]

    # ---------- Recompute μ, κ, π from sufficient statistics ----------
    def _recompute_mu_kappa_pi(self):
        eps = 1e-12
        self._sync_priors_shape()
        S_hat = self.S + self.n0 * self.C0
        n_hat = self.n + self.n0
        normS = np.linalg.norm(S_hat, axis=1, keepdims=True) + eps
        C_new = (S_hat / normS).astype(np.float32)
        d = C_new.shape[1]
        r = (normS.squeeze(-1)) / np.maximum(n_hat, 1e-6)
        r = np.clip(r, 1e-6, 1 - 1e-6)
        kappa_new = (r * (d - r ** 2)) / np.clip(1.0 - r ** 2, 1e-6, None)
        kappa_new = np.clip(kappa_new, 1.0, 1e6).astype(np.float32)

        nsum = float(np.maximum(n_hat.sum(), 1.0))
        pi_new = (n_hat / nsum).astype(np.float32)

        if self.kappa is None:
            self.kappa = kappa_new
        else:
            K_old, K_new = int(self.kappa.shape[0]), int(kappa_new.shape[0])
            if K_new == K_old:
                self.kappa = (1.0 - self.kappa_ema) * self.kappa + self.kappa_ema * kappa_new
            elif K_new > K_old:
                out = kappa_new.copy()
                out[:K_old] = (1.0 - self.kappa_ema) * self.kappa + self.kappa_ema * kappa_new[:K_old]
                self.kappa = out
            else:
                self.kappa = kappa_new
        self.pi = pi_new
        self.C = C_new

    # ---------- Scoring / actions ----------
    def _scores_post(self, x: np.ndarray):
        sims = self.C @ x  # [K]
        scores = np.log(np.clip(self.pi, 1e-8, None)) + self.kappa * sims
        top2 = np.argsort(scores)[-2:]
        k2, k1 = int(top2[0]), int(top2[1])
        s1, s2 = float(scores[k1]), float(scores[k2])
        z = scores - scores.max()
        p = np.exp(self.beta_post * z); p = p / (p.sum() + 1e-8)
        return k1, k2, s1, s2, float(p[k1]), float(p[k2])

    def _tau_assign(self, k: int, a0: float = 0.5, a1: float = 0.1):
        return 1.0 / (1.0 + np.exp(-(a0 + a1 * np.log(self.kappa[k] + 1e-6))))

    def _decide_action(self, x: np.ndarray):
        k1, k2, s1, s2, p1, p2 = self._scores_post(x)
        tau1 = self._tau_assign(k1)
        if s1 < -1.0 and p1 < tau1 * 0.7:
            return ("add", k1, k2, p1, p2)
        if p1 >= tau1 and (p1 - p2) >= self.delta_min:
            return ("update", k1, k2, p1, p2)
        if p1 >= tau1:
            return ("expand", k1, k2, p1, p2)
        return ("add", k1, k2, p1, p2)

    # ---------- Initialization (fit on original domain) ----------
    def fit_regions_on_original(self, seed=42):
        if self.original_ds is None or len(self.original_ds) == 0:
            raise RuntimeError("[RAIE] original dataset is required.")
        X = encode_prompts_to_vecs_causallm(self.model, self.original_ds, self.device, self.tok, batch_size=256)
        C, labels = spherical_kmeans(X, self.K, niter=30, seed=seed)
        R = per_cluster_radius(X, C, labels, q=self.q)
        sig = per_cluster_ang_std(X, C, labels)
        self.C, self.R, self.sig = C, R, sig
        self.C0 = C.copy()

        K, H = C.shape
        self.S = np.zeros((K, H), np.float32)
        self.n = np.zeros((K,), np.float32)
        for i, k in enumerate(labels.astype(int)):
            self.S[k] += X[i]
            self.n[k] += 1.0
        self._recompute_mu_kappa_pi()

        self.orig_idx_by_k = {int(k): [] for k in range(self.C.shape[0])}
        for i, k in enumerate(labels.tolist()):
            self.orig_idx_by_k[int(k)].append(i)

        np.savez(os.path.join(self.out_dir, "raie_regions_init.npz"),
                 centroids=C, radii=R, sig=sig, K=self.C.shape[0], dim=self.C.shape[1])
        return labels

    # =====================================================
    # ### [RAIE-CHANGE] Dynamic K: BIC eval + new cluster proposal + merge
    # =====================================================
    def _bic_score_vmf(self, X: np.ndarray, mu: np.ndarray):
        """
        Approximate vMF BIC using angular variance:
        BIC ~ N * log(sigma^2) + p * log(N). Used only for relative comparison.
        """
        if X.shape[0] == 0:
            return 0.0
        dots = np.clip(X @ mu, -1, 1)
        ang = np.arccos(dots)
        sig2 = float(np.maximum(ang.var(), 1e-12))
        p = X.shape[1]
        N = X.shape[0]
        return N * np.log(sig2) + p * np.log(max(N, 2))

    def _bic_gain_for_split(self, X: np.ndarray, mu_parent: np.ndarray, seed=42):
        """
        Run spherical KMeans with K=2 on X and return: gain, (mu_a, mu_b), (idx_a, idx_b).
        """
        if X.shape[0] < 4:
            return -np.inf, None, None
        # Split into two clusters.
        C2, lab2 = spherical_kmeans(X, K=2, niter=30, seed=seed)
        idx_a = np.where(lab2 == 0)[0]; idx_b = np.where(lab2 == 1)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            return -np.inf, None, None
        X_a, X_b = X[idx_a], X[idx_b]
        mu_a = l2n(np.mean(X_a, axis=0, keepdims=True)).reshape(-1)
        mu_b = l2n(np.mean(X_b, axis=0, keepdims=True)).reshape(-1)

        bic_parent = self._bic_score_vmf(X, mu_parent)
        bic_a = self._bic_score_vmf(X_a, mu_a)
        bic_b = self._bic_score_vmf(X_b, mu_b)
        gain = bic_parent - (bic_a + bic_b)
        return float(gain), (mu_a, mu_b), (idx_a, idx_b)

    def _propose_new_cluster_from_pool(self, X_all: np.ndarray, add_pool: List[Tuple[int,int]], bic_gain=1e4, seed=42):
        """
        Attempt to generate new clusters from add_pool entries (idx_in_Xn, nearest_k1):
          1) Group pooled samples by nearest parent cluster.
          2) For each group, compute BIC split gain; if gain >= bic_gain, create a new cluster.
          3) Update global C/R/sig/S/n/kappa/pi with dynamic expansion.
        Returns whether at least one cluster was added and cleans consumed add_pool entries.
        """
        if len(add_pool) == 0:
            return False

        used = set()
        changed = False
        # Group by parent cluster.
        by_parent: Dict[int, List[int]] = defaultdict(list)
        for idx, k1 in add_pool:
            by_parent[int(k1)].append(int(idx))

        for parent_k, idx_list in by_parent.items():
            if len(idx_list) < 4:
                continue
            Xp = X_all[idx_list]  # [m, H]
            mu_parent = self.C[parent_k].copy()
            gain, mubu, split_idx = self._bic_gain_for_split(Xp, mu_parent, seed=seed)
            if not np.isfinite(gain):
                continue
            if gain < bic_gain:
                continue  # Not enough evidence to create a new cluster.

            # Choose the more deviated child cluster (larger angular distance to parent).
            (mu_a, mu_b) = mubu
            (idx_a, idx_b) = split_idx
            dot_a = float(np.clip(mu_a @ mu_parent, -1, 1))
            dot_b = float(np.clip(mu_b @ mu_parent, -1, 1))
            # Larger angle means smaller dot product.
            choose_b = (dot_b < dot_a)
            new_mu = mu_b if choose_b else mu_a
            new_idx_local = idx_b if choose_b else idx_a
            new_idx_global = [idx_list[i] for i in new_idx_local]

            # --- Dynamic expansion ---
            K_old, H = self.C.shape
            K_new = K_old + 1

            self.C = np.vstack([self.C, new_mu.reshape(1, -1).astype(np.float32)])
            self.R = np.concatenate([self.R, np.array([0.0], np.float32)], axis=0)
            self.sig = np.concatenate([self.sig, np.array([0.0], np.float32)], axis=0)

            # Sufficient statistics for new cluster (S, n).
            vec_sum = X_all[new_idx_global].sum(0)
            S_new = vec_sum.astype(np.float32)
            n_new = float(len(new_idx_global))

            self.S = np.vstack([self.S, S_new.reshape(1, -1)])
            self.n = np.concatenate([self.n, np.array([n_new], np.float32)], axis=0)

            # priors
            if self.C0 is None:
                self.C0 = self.C.copy().astype(np.float32)
            else:
                self.C0 = np.vstack([self.C0, new_mu.reshape(1, -1).astype(np.float32)])

            # Track original indices.
            self.orig_idx_by_k[int(K_old)] = []

            # Update radius and variance (new cluster only).
            dots = np.clip(X_all[new_idx_global] @ self.C[-1], -1, 1)
            ang = np.arccos(dots)
            self.R[-1] = np.quantile(ang, self.q).astype(np.float32)
            self.sig[-1] = ang.std().astype(np.float32)

            # Recompute μ/κ/π.
            self._recompute_mu_kappa_pi()
            self.K = int(self.C.shape[0])

            # These samples are consumed from add_pool.
            for gi in new_idx_global:
                used.add((gi, parent_k))

            changed = True

        # Clean add_pool (retain unconsumed entries for nearest-cluster fallback).
        if changed:
            add_pool[:] = [(i, k1) for (i, k1) in add_pool if (i, k1) not in used]

        return changed

    def _try_merge_close_clusters(self, angle_thr: float = 0.15):
        """
        Try to merge the closest pair using angular distance and BIC gain.
        Returns whether a merge occurred.
        """
        if self.C.shape[0] <= 1:
            return False
        K = self.C.shape[0]
        # Find the closest pair.
        sims = self.C @ self.C.T
        np.fill_diagonal(sims, -1.0)
        k_flat = np.argmax(sims)
        i = int(k_flat // K); j = int(k_flat % K)
        i, j = min(i, j), max(i, j)

        cos_ij = float(np.clip(self.C[i] @ self.C[j], -1, 1))
        ang_ij = math.acos(cos_ij)
        if ang_ij > angle_thr:
            return False  # Not close enough.

        # Estimate whether merging improves BIC using S,n heuristics.
        mu_i = self.C[i]; mu_j = self.C[j]
        ni = max(float(self.n[i]), 1.0); nj = max(float(self.n[j]), 1.0)
        # Approximate BIC (relative comparison only).
        bic_i = ni * np.log(max(self.sig[i]**2, 1e-12))
        bic_j = nj * np.log(max(self.sig[j]**2, 1e-12))

        S_merge = self.S[i] + self.S[j]
        n_merge = ni + nj
        mu_merge = l2n(S_merge.reshape(1, -1)).reshape(-1)
        # Approximate post-merge variance via weighted average.
        sig_merge = max( (ni*self.sig[i] + nj*self.sig[j]) / max(n_merge,1.0), 1e-6)
        bic_merge = n_merge * np.log(max(sig_merge**2, 1e-12))

        gain = (bic_i + bic_j) - bic_merge
        if gain < 0:
            return False

        # Merge by collapsing j into i.
        self.S[i] = S_merge.astype(np.float32)
        self.n[i] = n_merge
        self.C[i] = mu_merge.astype(np.float32)
        self.sig[i] = float(sig_merge)
        # Re-estimate radius with a stable heuristic (keep the larger radius).
        self.R[i] = float(max(self.R[i], self.R[j]))

        # Drop j.
        keep = [k for k in range(self.C.shape[0]) if k != j]
        self.C = self.C[keep]
        self.R = self.R[keep]
        self.sig = self.sig[keep]
        self.S = self.S[keep]
        self.n = self.n[keep]
        if self.C0 is not None:
            self.C0 = self.C0[keep]
        # Remap orig_idx_by_k.
        new_map = {}
        t = 0
        for k in range(len(keep)):
            new_map[t] = self.orig_idx_by_k.get(keep[k], [])
            t += 1
        self.orig_idx_by_k = new_map
        self._recompute_mu_kappa_pi()
        self.K = int(self.C.shape[0])
        return True

    # ---------- Map & Finetune ----------
    def map_finetune(self, finetune_ds: Dataset, pool_min=200, bic_gain=1e4):
        Xn = encode_prompts_to_vecs_causallm(self.model, finetune_ds, self.device, self.tok, batch_size=256)
        buckets = defaultdict(list)
        add_pool = []  # (idx, nearest_k1)
        soft_pairs = []

        for i in range(Xn.shape[0]):
            x = Xn[i]
            act, k1, k2, p1, p2 = self._decide_action(x)
            if act == "update":
                buckets[k1].append(i)
                self.S[k1] += x; self.n[k1] += 1.0
            elif act == "expand":
                buckets[k1].append(i)
                s = p1 + p2 + 1e-8
                w1, w2 = p1 / s, p2 / s
                soft_pairs.append((i, k1, k2, float(w1), float(w2)))
                self.S[k1] += 0.25 * x; self.n[k1] += 0.25
            else:
                add_pool.append((i, k1))
                soft_pairs.append((i, k1, k2, 0.3, 0.0))

            # Light online updates (same idea as the BERT4Rec variant).
            if (i+1) % 64 == 0:
                self._recompute_mu_kappa_pi()

        # Recompute once before pool handling.
        self._recompute_mu_kappa_pi()

        # ### [RAIE-CHANGE]: propose new clusters and optional merge when pool is large enough.
        if len(add_pool) >= pool_min:
            changed = self._propose_new_cluster_from_pool(Xn, add_pool, bic_gain=bic_gain, seed=42)
            if changed:
                # Optional: perform a nearest-cluster merge (same idea as BERT4Rec variant).
                _ = self._try_merge_close_clusters(angle_thr=max(self.gap_thr, 0.05))
                # Recompute statistics after changes.
                self._recompute_mu_kappa_pi()

        # Assign remaining pooled samples to their nearest cluster.
        for idx, near_k1 in add_pool:
            buckets[near_k1].append(idx)

        np.savez(os.path.join(self.out_dir, "raie_regions_after_map.npz"),
                 centroids=self.C, radii=self.R, sig=self.sig,
                 kappa=self.kappa, pi=self.pi, S=self.S, n=self.n)
        return buckets, soft_pairs

    # ---------- Adapter setup ----------
    def _ensure_adapter(self, adapter_name: str):
        if adapter_name not in getattr(self.model, "peft_config", {}):
            dir_anchor = os.path.join(self.out_dir, "default_for_raie")
            dir_o = os.path.join(self.out_dir, "default")
            if os.path.isdir(dir_anchor):
                self.model.load_adapter(dir_anchor, adapter_name=adapter_name, is_trainable=True)
            elif os.path.isdir(dir_o):
                self.model.load_adapter(dir_o, adapter_name=adapter_name, is_trainable=True)
            else:
                self.model.add_adapter(adapter_name, self.lora_cfg)

    # ---------- Region training (including new regions) ----------
    def train_regions(self, finetune_ds: Dataset, collate_fn,
                      optimizer_ctor, scheduler_ctor,
                      epochs_per_region=2, batch_size=256, orig_mix_ratio=1.0, seed=42,
                      region_buckets: Dict[int, List[int]] = None,
                      soft_pairs: Optional[List[Tuple[int,int,int,float,float]]] = None,
                      soft_rep_factor: int = 3, fp16=False):
        assert region_buckets is not None
        soft_pairs = soft_pairs or []

        soft_idx_by_k = defaultdict(list)
        for (i, k1, k2, w1, w2) in soft_pairs:
            if w1 > 0: soft_idx_by_k[k1].extend([i] * max(0, int(round(w1 * soft_rep_factor))))
            if w2 > 0: soft_idx_by_k[k2].extend([i] * max(0, int(round(w2 * soft_rep_factor))))

        all_regions = sorted(set(region_buckets.keys()))
        for k in all_regions:
            idx_list = region_buckets.get(k, [])
            if len(idx_list) == 0:
                continue
            name = f"region_{k}"
            self._ensure_adapter(name)
            self.model.set_adapter(name)
            if hasattr(self.model, "train_adapter"): self.model.train_adapter(name)
            self.model.train()

            ft_subset = Subset(finetune_ds, idx_list)
            if (k in self.orig_idx_by_k) and (orig_mix_ratio > 0) and (self.original_ds is not None):
                orig_idx = self.orig_idx_by_k[k]
                n_ft = len(ft_subset)
                n_orig = int(round(n_ft * orig_mix_ratio / max(1e-8, 1.0 - orig_mix_ratio)))
                if n_orig > 0 and len(orig_idx) > 0:
                    rng = np.random.RandomState(seed)
                    sel = rng.choice(orig_idx, size=min(n_orig, len(orig_idx)), replace=False)
                    orig_subset = Subset(self.original_ds, sel)
                    base_ds = ConcatDataset([ft_subset, orig_subset])
                else:
                    base_ds = ft_subset
            else:
                base_ds = ft_subset

            if len(soft_idx_by_k[k]) > 0:
                soft_subset = Subset(finetune_ds, soft_idx_by_k[k])
                train_ds = ConcatDataset([base_ds, soft_subset])
            else:
                train_ds = base_ds

            dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
            optim = optimizer_ctor(self.model)
            sched = scheduler_ctor(optim, len(dl), epochs_per_region)

            for ep in range(epochs_per_region):
                pbar = tqdm(dl, desc=f"[RAIE] Region {k} Ep {ep+1}/{epochs_per_region}")
                for batch in pbar:
                    loss, _ = ce_loss_on_items(self.model, batch, self.device, self.item_final_token_ids, fp16=fp16)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()
                    if sched is not None: sched.step()
                    pbar.set_postfix({"loss": float(loss.detach().cpu())})

            save_dir = os.path.join(self.out_dir, f"adapter_region_{k}")
            ensure_dir(save_dir)
            self.model.save_pretrained(save_dir, selected_adapters=[name])

    @torch.no_grad()
    def route_and_eval(self, test_ds: Dataset, collate_fn,
                       k_list=(5,10,20), batch_size=256, root=None, fp16=False):
        if hasattr(self.model, "set_adapter"):
            try: self.model.set_adapter("default")
            except: pass

        X = encode_prompts_to_vecs_causallm(self.model, test_ds, self.device, self.tok, batch_size=batch_size)
        sims = X @ self.C.T
        scores = np.log(np.clip(self.pi, 1e-8, None))[None, :] + sims * self.kappa[None, :]
        route_k = np.argmax(scores, axis=1)

        idx_by_k = defaultdict(list)
        for i, k in enumerate(route_k.tolist()):
            idx_by_k[int(k)].append(i)

        sums = {f"Recall@{k}": 0.0 for k in k_list} | {f"NDCG@{k}": 0.0 for k in k_list}
        N = 0
        for k, idxs in idx_by_k.items():
            name = f"region_{k}"
            ad_dir = os.path.join(root or self.out_dir, f"adapter_region_{k}", name)
            if name not in getattr(self.model, "peft_config", {}):
                if os.path.isdir(ad_dir):
                    self.model.load_adapter(ad_dir, adapter_name=name, is_trainable=False)
                else:
                    continue
            self.model.set_adapter(name)

            sub = Subset(test_ds, idxs)
            dl = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)
            m = evaluate_causal(self.model, dl, self.device, self.item_final_token_ids, k_list=k_list, fp16=fp16, desc=f"Eval region {k}")
            n_sub = len(sub)
            for kk in k_list:
                sums[f"Recall@{kk}"] += m[f"Recall@{kk}"] * n_sub
                sums[f"NDCG@{kk}"] += m[f"NDCG@{kk}"] * n_sub
            N += n_sub
        return {k: (v / max(1, N)) for k, v in sums.items()}

# ------------------------------
# Shared utilities: save/load data and mid2idx
# ------------------------------
def save_mid2idx(path: str, mid2idx: Dict[int, int]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in mid2idx.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] save mid2idx failed: {e}")

def load_mid2idx(path: str) -> Optional[Dict[int, int]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {int(k): int(v) for k, v in obj.items()}
    except Exception as e:
        print(f"[WARN] load mid2idx failed: {e}")
        return None

def build_or_load_mid2idx(args, is_main=True) -> Tuple[Dict[int,int], int]:
    """
    Load mid2idx from output_dir/mid2idx.json when available, otherwise rebuild
    using the same logic. Returns mid2idx and n_items.
    """
    mid2idx_path = os.path.join(args.output_dir, "mid2idx.json")
    mid2idx = load_mid2idx(mid2idx_path)
    item_ids = _read_vocab(args.item_ids_path or os.path.join(args.data_dir, "item_ids.json"))
    n_items = len(item_ids)
    if mid2idx is not None:
        if is_main:
            print(f"[INFO] Loaded mid2idx from {mid2idx_path} (|mid|={len(mid2idx)})")
        return mid2idx, n_items

    # Fallback: rebuild mapping.
    if is_main:
        print("[INFO] mid2idx.json not found -> rebuild mapping from training split.")
    _, _, _, n_items2, mid2idx = load_preprocessed_splits(
        data_dir=args.data_dir,
        item_indexing=args.item_indexing,
        seed=args.seed,
        min_user_len=args.min_user_len,
        train_jsonl_path=args.train_jsonl_path or os.path.join(args.data_dir, "original_stride1.jsonl"),
        item_ids_path=args.item_ids_path or os.path.join(args.data_dir, "item_ids.json"),
        test_jsonl_path=None
    )
    # Save for reuse in later stages.
    if (mid2idx is not None) and (len(mid2idx) > 0):
        if is_main:
            save_mid2idx(mid2idx_path, mid2idx)
            print(f"[SAVE] mid2idx saved to {mid2idx_path}")
    assert n_items == n_items2, "item_ids size mismatch after rebuild; check data consistency."
    return mid2idx, n_items

def make_datasets_for_stage(args, tok, mid2idx, n_items):
    """
    Build datasets for each stage without relying on pre-stage model state.
    """
    final_item_tokens = make_item_final_tokens(n_items)
    item_token_seq = [[t] for t in final_item_tokens]

    def make_ds(jsonl_path):
        return SeqRecDatasetFromPairs(
            jsonl_path, mid2idx, tok, item_token_seq,
            max_history=args.max_history, wrap_mode="none",
            final_token_only_loss=args.final_token_only_loss,
            deterministic_template=True
        )

    train_jsonl_path = args.train_jsonl_path or os.path.join(args.data_dir, "original_stride1.jsonl")
    original_jsonl_path = args.original_jsonl_path or os.path.join(args.data_dir, "original.jsonl")
    test_jsonl_path     = args.test_jsonl_path or os.path.join(args.data_dir, "test.jsonl")
    finetune_jsonl_path = args.finetune_jsonl_path or os.path.join(args.data_dir, "finetune.jsonl")
    if not os.path.exists(finetune_jsonl_path):
        raise FileNotFoundError(f"finetune_jsonl_path not found: {finetune_jsonl_path}")

    ds_trainO = make_ds(train_jsonl_path)
    ds_O = make_ds(original_jsonl_path)
    ds_T = make_ds(test_jsonl_path)
    ds_F = make_ds(finetune_jsonl_path)
    return ds_trainO, ds_O, ds_T, ds_F, final_item_tokens

# ------------------------------
# Stage execution functions
# ------------------------------
def stage_pre(args, device, is_distributed, local_rank, is_main, load_dtype):
    """
    Run everything before LoRA, producing:
    - output_dir/default (O-stage LoRA adapter)
    - output_dir/base_with_new_tokens (clean base with new tokens)
    - output_dir/mid2idx.json
    - original/test metrics json
    """
    # 1) Data / Indexing
    train_seq, val_seq, test_seq, n_items, mid2idx = load_preprocessed_splits(
        data_dir=args.data_dir,
        item_indexing=args.item_indexing,
        seed=args.seed,
        min_user_len=args.min_user_len,
        train_jsonl_path=args.train_jsonl_path or os.path.join(args.data_dir, "original_stride1.jsonl"),
        item_ids_path=args.item_ids_path or os.path.join(args.data_dir, "item_ids.json"),
        test_jsonl_path=None
    )
    if is_main:
        print(f"[Data] items={n_items} | train_users={len(train_seq)} | val_users={len(val_seq)}")

    # 2) Tokenizer & model (extend vocab)
    if is_main:
        print("[Model] Loading tokenizer and model ...")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, torch_dtype=load_dtype)
    if args.grad_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    final_item_tokens = make_item_final_tokens(n_items)
    num_added = tok.add_tokens(final_item_tokens, special_tokens=False)
    prev_vocab_size = len(tok) - num_added if num_added > 0 else len(tok)
    if num_added > 0:
        model.resize_token_embeddings(len(tok), mean_resizing=False)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            std = emb[:prev_vocab_size].std().item()
            new_token_ids = []
            for t in final_item_tokens:
                tid = tok.convert_tokens_to_ids(t)
                if tid >= prev_vocab_size:
                    new_token_ids.append(tid)
            for tid in new_token_ids:
                torch.nn.init.normal_(emb[tid], mean=0.0, std=std)
        emb_mod = model.get_input_embeddings()
        emb_mod.weight.requires_grad = True
        base_rows = prev_vocab_size
        def _zero_base_rows_grad(grad):
            if grad is not None and grad.dim() == 2 and grad.size(0) >= base_rows:
                grad[:base_rows] = 0
            return grad
        emb_mod.weight.register_hook(_zero_base_rows_grad)

    item_final_token_ids: List[int] = [tok.convert_tokens_to_ids(t) for t in final_item_tokens]

    # LoRA wrapping (train O stage)
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError(f"PEFT not available: {_PEFT_ERR}")
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias='none', task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        )
        model = get_peft_model(model, peft_cfg)
    model.to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )

    item_token_seq = [[tok_str] for tok_str in final_item_tokens]

    def make_ds(jsonl_path):
        return SeqRecDatasetFromPairs(
            jsonl_path, mid2idx, tok, item_token_seq,
            max_history=args.max_history, wrap_mode="none",
            final_token_only_loss=args.final_token_only_loss,
            deterministic_template=True
        )

    ds_trainO = make_ds(args.train_jsonl_path or os.path.join(args.data_dir, "original_stride1.jsonl"))
    ds_O = make_ds(args.original_jsonl_path or os.path.join(args.data_dir, "original.jsonl"))
    ds_T = make_ds(args.test_jsonl_path or os.path.join(args.data_dir, "test.jsonl"))
    finetune_jsonl_path = args.finetune_jsonl_path or os.path.join(args.data_dir, "finetune.jsonl")
    if not os.path.exists(finetune_jsonl_path):
        raise FileNotFoundError(f"finetune_jsonl_path not found: {finetune_jsonl_path}")
    ds_F = make_ds(finetune_jsonl_path)

    pad_id = tok.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)

    train_sampler = DistributedSampler(ds_trainO, num_replicas=dist.get_world_size() if is_distributed else 1,
                                       rank=dist.get_rank() if is_distributed else 0, shuffle=True) if is_distributed else None
    train_loader = DataLoader(ds_trainO, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=2, pin_memory=True, collate_fn=collate)

    ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    # Optimizer (O stage)
    lora_params, emb_params, other_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)
        elif "embed_tokens" in n or "wte" in n or "embedding" in n:
            emb_params.append(p)
        else:
            other_params.append(p)
    optim_groups = []
    if lora_params: optim_groups.append({"params": lora_params, "weight_decay": 0.0})
    if emb_params:  optim_groups.append({"params": emb_params, "weight_decay": 0.0})
    if other_params: optim_groups.append({"params": other_params, "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)
    num_train_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * num_train_steps), num_train_steps)

    k_list = sorted(set(args.topk))

    # 5) Training (O stage only, no per-epoch eval)
    if is_main:
        print(f"[Train] O-stage epochs={args.epochs}")
    for ep in range(1, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(ep)
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                               item_final_token_ids=item_final_token_ids,
                               fp16=args.fp16, is_main=is_main, grad_clip=args.grad_clip)
        if is_main:
            print(f"[Train][O] epoch={ep} loss={loss:.4f}")

    if is_distributed:
        dist.barrier()
    if is_main:
        base_dir = os.path.join(args.output_dir, 'base_model')
        os.makedirs(base_dir, exist_ok=True)
        base_eval_model = model.module if hasattr(model, "module") else model
        base_eval_model.save_pretrained(base_dir)
        mO = evaluate_causal(model, ld_O, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-O")
        mT = evaluate_causal(model, ld_T, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-T")
        with open(os.path.join(args.output_dir, "metrics_original.json"), "w", encoding="utf-8") as f:
            json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
            json.dump(mT, f, ensure_ascii=False, indent=2)
    if is_distributed:
        dist.barrier()

    # Save default adapter + clean base (with new vocab) + mid2idx.json
    if is_main and args.use_lora:
        default_dir = os.path.join(args.output_dir, "default")
        ensure_dir(default_dir)
        peft_wrapped = model.module if hasattr(model, "module") else model
        peft_wrapped.save_pretrained(default_dir, selected_adapters=["default"])
        tok.save_pretrained(default_dir)

    # Generate clean base.
    if is_main and args.use_lora:
        base_dir = os.path.join(args.output_dir, "base_with_new_tokens")
        save_clean_base_with_new_tokens(
            peft_wrapped=peft_wrapped,
            tokenizer=tok,
            out_dir=base_dir,
            orig_base_path=args.model_name_or_path
        )
        print(f"[SAVE] clean base saved to {base_dir}")
        # Save mid2idx.
        save_mid2idx(os.path.join(args.output_dir, "mid2idx.json"), mid2idx)
    if is_distributed:
        dist.barrier()
    try:
        del optimizer, scheduler
    except Exception:
        pass
    try:
        peft_wrapped = locals().get("peft_wrapped", None)
        _free_cuda(model, peft_wrapped)
    except Exception:
        pass
    aggressive_gc()

def fresh_lora_model_from_base(base_dir, device, lora_r, lora_alpha, lora_dropout, load_dtype):
    tok2 = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    if tok2.pad_token is None: tok2.pad_token = tok2.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base_dir, low_cpu_mem_usage=True, torch_dtype=load_dtype,
    )
    if hasattr(m.config, "use_cache"): m.config.use_cache = False
    lcfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    m = get_peft_model(m, lcfg)
    m.to(device)
    return tok2, m

def stage_lora(args, device, is_distributed, local_rank, is_main, load_dtype):
    base_dir = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[stage lora] base_with_new_tokens not found at {base_dir}")
    default_dir = os.path.join(args.output_dir, "default")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(f"[stage lora] default adapter dir not found at {default_dir}")

    tok_lora, lora_model = fresh_lora_model_from_base(
        base_dir, device, args.lora_r, args.lora_alpha, args.lora_dropout, load_dtype
    )
    lora_model.load_adapter(default_dir, adapter_name="default", is_trainable=True)
    lora_model.set_adapter("default")

    # Prepare data (using saved/rebuilt mid2idx).
    mid2idx, n_items = build_or_load_mid2idx(args, is_main=is_main)
    ds_trainO, ds_O, ds_T, ds_F, final_item_tokens = make_datasets_for_stage(args, tok_lora, mid2idx, n_items)
    pad_id = tok_lora.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)
    item_final_token_ids = [tok_lora.convert_tokens_to_ids(t) for t in final_item_tokens]

    if is_distributed:
        lora_model = torch.nn.parallel.DistributedDataParallel(
            lora_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )

    ft_sampler = DistributedSampler(ds_F, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                    rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    ld_F_ddp = DataLoader(ds_F, batch_size=args.finetune_batch_size,
                          shuffle=(ft_sampler is None), sampler=ft_sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    FT_optim = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=args.lr)
    FT_sched = get_linear_schedule_with_warmup(FT_optim, int(args.warmup_ratio*len(ld_F_ddp)*max(1,args.finetune_epochs)),
                                               len(ld_F_ddp)*max(1,args.finetune_epochs))

    for ep in range(1, args.finetune_epochs + 1):
        if is_distributed and ft_sampler is not None: ft_sampler.set_epoch(ep)
        loss = finetune_one_epoch_lora(
            lora_model, ld_F_ddp, device, item_final_token_ids,
            optimizer=FT_optim, scheduler=FT_sched, fp16=args.fp16,
            plugin="none", replay_buf=None,
            replay_ratio=args.replay_ratio, grad_clip=1.0, log_every=100
        )
        if is_main: print(f"[Method: LoRA] F epoch={ep} loss={loss:.4f}")

    if is_main:
        base_eval_model = lora_model.module if hasattr(lora_model, "module") else lora_model
        k_list = sorted(set(args.topk))
        mO = evaluate_causal(base_eval_model, ld_O, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-O(lora)")
        mT = evaluate_causal(base_eval_model, ld_T, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-T(lora)")
        with open(os.path.join(args.output_dir, "metrics_original_lora.json"), "w", encoding="utf-8") as f: json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test_lora.json"), "w", encoding="utf-8") as f: json.dump(mT, f, ensure_ascii=False, indent=2)
        save_lora = os.path.join(args.output_dir, "lora_adapter_lora"); ensure_dir(save_lora); base_eval_model.save_pretrained(save_lora)
        print("[Done] LoRA:", mO, mT)

    try:
        del FT_optim, FT_sched
    except Exception:
        pass
    _free_cuda(lora_model)
    if is_distributed: dist.barrier()

def stage_replay(args, device, is_distributed, local_rank, is_main, load_dtype):
    base_dir = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[stage replay] base_with_new_tokens not found at {base_dir}")
    default_dir = os.path.join(args.output_dir, "default")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(f"[stage replay] default adapter dir not found at {default_dir}")

    tok_rep, rep_model = fresh_lora_model_from_base(
        base_dir, device, args.lora_r, args.lora_alpha, args.lora_dropout, load_dtype
    )
    rep_model.load_adapter(default_dir, adapter_name="default", is_trainable=True)
    rep_model.set_adapter("default")

    mid2idx, n_items = build_or_load_mid2idx(args, is_main=is_main)
    ds_trainO, ds_O, ds_T, ds_F, final_item_tokens = make_datasets_for_stage(args, tok_rep, mid2idx, n_items)
    pad_id = tok_rep.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)
    item_final_token_ids = [tok_rep.convert_tokens_to_ids(t) for t in final_item_tokens]

    if is_distributed:
        rep_model = torch.nn.parallel.DistributedDataParallel(
            rep_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )

    ft_sampler = DistributedSampler(ds_F, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                    rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    ld_F_ddp = DataLoader(ds_F, batch_size=args.finetune_batch_size,
                          shuffle=(ft_sampler is None), sampler=ft_sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    rb_sampler = DistributedSampler(ds_O, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                    rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    replay_buf = ReplayBuffer(ds_O, collate, batch_size=args.finetune_batch_size, sampler=rb_sampler)

    ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    FT_optim = torch.optim.AdamW([p for p in rep_model.parameters() if p.requires_grad], lr=args.lr)
    FT_sched = get_linear_schedule_with_warmup(FT_optim, int(args.warmup_ratio*len(ld_F_ddp)*max(1,args.finetune_epochs)),
                                               len(ld_F_ddp)*max(1,args.finetune_epochs))

    for ep in range(1, args.finetune_epochs + 1):
        if is_distributed and ft_sampler is not None: ft_sampler.set_epoch(ep)
        if is_distributed: replay_buf.set_epoch(ep)
        loss = finetune_one_epoch_lora(
            rep_model, ld_F_ddp, device, item_final_token_ids,
            optimizer=FT_optim, scheduler=FT_sched, fp16=args.fp16,
            plugin="replay", replay_buf=replay_buf,
            replay_ratio=args.replay_ratio, grad_clip=1.0, log_every=100
        )
        if is_main: print(f"[Method: LoRA+Replay] F epoch={ep} loss={loss:.4f}")

    if is_main:
        base_eval_model = rep_model.module if hasattr(rep_model, "module") else rep_model
        k_list = sorted(set(args.topk))
        mO = evaluate_causal(base_eval_model, ld_O, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-O(lora_replay)")
        mT = evaluate_causal(base_eval_model, ld_T, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-T(lora_replay)")
        with open(os.path.join(args.output_dir, "metrics_original_lora_replay.json"), "w", encoding="utf-8") as f: json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test_lora_replay.json"), "w", encoding="utf-8") as f: json.dump(mT, f, ensure_ascii=False, indent=2)
        save_lora = os.path.join(args.output_dir, "lora_adapter_lora_replay"); ensure_dir(save_lora); base_eval_model.save_pretrained(save_lora)
        print("[Done] LoRA+Replay:", mO, mT)

    try:
        del FT_optim, FT_sched, replay_buf
    except Exception:
        pass
    _free_cuda(rep_model)
    if is_distributed: dist.barrier()

def stage_lwf(args, device, is_distributed, local_rank, is_main, load_dtype):
    base_dir = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[stage lwf] base_with_new_tokens not found at {base_dir}")
    default_dir = os.path.join(args.output_dir, "default")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(f"[stage lwf] default adapter dir not found at {default_dir}")

    tok_lwf, lwf_model = fresh_lora_model_from_base(
        base_dir, device, args.lora_r, args.lora_alpha, args.lora_dropout, load_dtype
    )
    lwf_model.load_adapter(default_dir, adapter_name="default", is_trainable=True)
    lwf_model.set_adapter("default")

    mid2idx, n_items = build_or_load_mid2idx(args, is_main=is_main)
    ds_trainO, ds_O, ds_T, ds_F, final_item_tokens = make_datasets_for_stage(args, tok_lwf, mid2idx, n_items)
    pad_id = tok_lwf.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)
    item_final_token_ids = [tok_lwf.convert_tokens_to_ids(t) for t in final_item_tokens]

    if is_distributed:
        lwf_model = torch.nn.parallel.DistributedDataParallel(
            lwf_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )
    base_eval_model = lwf_model.module if hasattr(lwf_model, "module") else lwf_model

    ft_sampler = DistributedSampler(ds_F, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                    rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    ld_F_ddp = DataLoader(ds_F, batch_size=args.finetune_batch_size,
                          shuffle=(ft_sampler is None), sampler=ft_sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    teacher = AutoModelForCausalLM.from_pretrained(base_dir, low_cpu_mem_usage=True, torch_dtype=load_dtype)
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    FT_optim = torch.optim.AdamW([p for p in lwf_model.parameters() if p.requires_grad], lr=args.lr)
    FT_sched = get_linear_schedule_with_warmup(FT_optim, int(args.warmup_ratio*len(ld_F_ddp)*max(1,args.finetune_epochs)),
                                               len(ld_F_ddp)*max(1,args.finetune_epochs))

    for ep in range(1, args.finetune_epochs + 1):
        if is_distributed and ft_sampler is not None: ft_sampler.set_epoch(ep)
        loss = finetune_one_epoch_lora(
            lwf_model, ld_F_ddp, device, item_final_token_ids,
            optimizer=FT_optim, scheduler=FT_sched, fp16=args.fp16,
            plugin="lwf", replay_buf=None, replay_ratio=0.0, grad_clip=1.0, log_every=100,
            teacher=teacher, lwf_T=args.lwf_T, lwf_alpha=args.lwf_alpha,
        )
        if is_main: print(f"[Method: LoRA+LwF] F epoch={ep} loss={loss:.4f}")

    if is_main:
        k_list = sorted(set(args.topk))
        mO = evaluate_causal(base_eval_model, ld_O, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-O(lora_lwf)")
        mT = evaluate_causal(base_eval_model, ld_T, device, item_final_token_ids, k_list=k_list, fp16=args.fp16, desc="Eval-T(lora_lwf)")
        with open(os.path.join(args.output_dir, "metrics_original_lora_lwf.json"), "w", encoding="utf-8") as f: json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test_lora_lwf.json"), "w", encoding="utf-8") as f: json.dump(mT, f, ensure_ascii=False, indent=2)
        save_lora = os.path.join(args.output_dir, "lora_adapter_lora_lwf"); ensure_dir(save_lora); base_eval_model.save_pretrained(save_lora)
        print("[Done] LoRA+LwF:", mO, mT)

    try:
        del FT_optim, FT_sched
    except Exception:
        pass
    _free_cuda(lwf_model)
    _free_cuda(teacher)
    if is_distributed: dist.barrier()

def stage_lsat(args, device, is_distributed, local_rank, is_main, load_dtype):
    base_dir = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[stage lsat] base_with_new_tokens not found at {base_dir}")
    default_dir = os.path.join(args.output_dir, "default")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(f"[stage lsat] default adapter dir not found at {default_dir}")

    tok_lsat, lsat_model = fresh_lora_model_from_base(
        base_dir, device, args.lora_r, args.lora_alpha, args.lora_dropout, load_dtype
    )
    lsat_model.load_adapter(default_dir, adapter_name="default", is_trainable=True)
    lsat_model.set_adapter("default")

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    if "short_term" not in getattr(lsat_model, "peft_config", {}):
        lsat_model.add_adapter("short_term", lcfg)

    mid2idx, n_items = build_or_load_mid2idx(args, is_main=is_main)
    ds_trainO, ds_O, ds_T, ds_F, final_item_tokens = make_datasets_for_stage(args, tok_lsat, mid2idx, n_items)
    pad_id = tok_lsat.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)
    item_final_token_ids = [tok_lsat.convert_tokens_to_ids(t) for t in final_item_tokens]

    if is_distributed:
        lsat_model = torch.nn.parallel.DistributedDataParallel(
            lsat_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )
    base_eval_model = lsat_model.module if hasattr(lsat_model, "module") else lsat_model

    train_sampler = DistributedSampler(ds_trainO, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                       rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    train_loader = DataLoader(ds_trainO, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=2, pin_memory=True, collate_fn=collate)

    ft_sampler = DistributedSampler(ds_F, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                    rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    ld_F_ddp = DataLoader(ds_F, batch_size=args.finetune_batch_size,
                          shuffle=(ft_sampler is None), sampler=ft_sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    if hasattr(lsat_model, "set_adapter"):
        try:
            lsat_model.set_adapter("default")
            if hasattr(lsat_model, "train_adapter"):
                lsat_model.train_adapter("default")
        except Exception:
            pass

    long_optim = torch.optim.AdamW([p for p in lsat_model.parameters() if p.requires_grad], lr=args.lr)
    long_sched = get_linear_schedule_with_warmup(long_optim, int(args.warmup_ratio*len(train_loader)*max(1,args.lsat_long_epochs)),
                                                 len(train_loader)*max(1,args.lsat_long_epochs))
    for ep in range(1, args.lsat_long_epochs + 1):
        if is_distributed and train_sampler is not None: train_sampler.set_epoch(ep)
        loss = train_one_epoch(lsat_model, train_loader, long_optim, long_sched, device,
                               item_final_token_ids=item_final_token_ids, fp16=args.fp16, is_main=is_main, grad_clip=args.grad_clip)
        if is_main: print(f"[LSAT][Long] epoch={ep} loss={loss:.4f}")

    if hasattr(lsat_model, "set_adapter"):
        try:
            lsat_model.set_adapter("short_term")
            if hasattr(lsat_model, "train_adapter"):
                lsat_model.train_adapter("short_term")
        except Exception:
            pass

    short_optim = torch.optim.AdamW([p for p in lsat_model.parameters() if p.requires_grad], lr=args.lr)
    short_sched = get_linear_schedule_with_warmup(short_optim, int(args.warmup_ratio*len(ld_F_ddp)*max(1,args.lsat_short_epochs)),
                                                  len(ld_F_ddp)*max(1,args.lsat_short_epochs))
    for ep in range(1, args.lsat_short_epochs + 1):
        if is_distributed and ft_sampler is not None: ft_sampler.set_epoch(ep)
        loss = finetune_one_epoch_lora(
            lsat_model, ld_F_ddp, device, item_final_token_ids,
            optimizer=short_optim, scheduler=short_sched, fp16=args.fp16,
            plugin="none", replay_buf=None, replay_ratio=0.0, grad_clip=1.0, log_every=100
        )
        if is_main: print(f"[LSAT][Short] epoch={ep} loss={loss:.4f}")

    if is_main:
        k_list = sorted(set(args.topk))
        mO = evaluate_lsat_causal(base_eval_model, ld_O, device, item_final_token_ids, long_adapter="default", short_adapter="short_term", alpha=args.lsat_alpha, k_list=k_list)
        mT = evaluate_lsat_causal(base_eval_model, ld_T, device, item_final_token_ids, long_adapter="default", short_adapter="short_term", alpha=args.lsat_alpha, k_list=k_list)
        with open(os.path.join(args.output_dir, "metrics_original_lsat.json"), "w", encoding="utf-8") as f: json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test_lsat.json"), "w", encoding="utf-8") as f: json.dump(mT, f, ensure_ascii=False, indent=2)
        long_dir = os.path.join(args.output_dir, "lora_adapter_lsat_long"); ensure_dir(long_dir); base_eval_model.save_pretrained(long_dir, selected_adapters=["default"])
        short_dir = os.path.join(args.output_dir, "lora_adapter_lsat_short"); ensure_dir(short_dir); base_eval_model.save_pretrained(short_dir, selected_adapters=["short_term"])
        print("[Done] LSAT:", mO, mT)

    try:
        del long_optim, long_sched, short_optim, short_sched
    except Exception:
        pass
    _free_cuda(lsat_model)
    if is_distributed: dist.barrier()

def _cast_trainable_mole_params_to_fp32(peft_model, adapter_names: List[str]):
    """Keep base weights in FP16 while moving trainable MoLE adapters to FP32."""
    base_peft = peft_model.module if hasattr(peft_model, "module") else peft_model
    for n, p in base_peft.named_parameters():
        if not p.requires_grad:
            continue
        if ("lora_" not in n) and (".lora_" not in n):
            continue
        if any((name != "default") and (f".{name}." in n) for name in adapter_names):
            p.data = p.data.float()

def stage_mole(args, device, is_distributed, local_rank, is_main, load_dtype):
    base_dir = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[stage mole] base_with_new_tokens not found at {base_dir}")
    default_dir = os.path.join(args.output_dir, "default")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(f"[stage mole] default adapter dir not found at {default_dir}")

    tok_mole, lora_model = fresh_lora_model_from_base(
        base_dir, device, args.lora_r, args.lora_alpha, args.lora_dropout, load_dtype
    )
    lora_model.load_adapter(default_dir, adapter_name="default", is_trainable=False)

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    adapter_names = []
    num_exp = max(1, args.mole_num_experts)
    for i in range(num_exp):
        name = "default" if i == 0 else f"expert_{i}"
        adapter_names.append(name)
        if name not in getattr(lora_model, "peft_config", {}):
            lora_model.add_adapter(name, lcfg)
        if (name != "default") and hasattr(lora_model, "train_adapter"):
            try:
                lora_model.train_adapter(name)
            except Exception:
                pass
    base_peft = lora_model.module if hasattr(lora_model, "module") else lora_model

    def _is_lora_param(n: str) -> bool:
        return ("lora_" in n) or (".lora_" in n)

    for n, p in base_peft.named_parameters():
        if _is_lora_param(n) and ".default." in n:
            p.requires_grad_(False)

    for n, p in base_peft.named_parameters():
        if not _is_lora_param(n):
            continue
        for name in adapter_names:
            if name == "default":
                continue
            if f".{name}." in n:
                p.requires_grad_(True)
                break
    _cast_trainable_mole_params_to_fp32(lora_model, adapter_names)

    mid2idx, n_items = build_or_load_mid2idx(args, is_main=is_main)
    ds_trainO, ds_O, ds_T, ds_F, final_item_tokens = make_datasets_for_stage(args, tok_mole, mid2idx, n_items)
    pad_id = tok_mole.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)
    item_final_token_ids = [tok_mole.convert_tokens_to_ids(t) for t in final_item_tokens]

    if is_distributed:
        lora_model = torch.nn.parallel.DistributedDataParallel(
            lora_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )

    ft_sampler = DistributedSampler(ds_F, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                    rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    ld_F_ddp = DataLoader(ds_F, batch_size=args.finetune_batch_size,
                          shuffle=(ft_sampler is None), sampler=ft_sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate)

    ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    mole_model = MoLEMixtureCausal(
        base_model=lora_model,
        adapter_names=adapter_names,
        item_final_token_ids=item_final_token_ids,
        gating_hidden=args.mole_gating_hidden,
        temperature=args.mole_temp,
        balance_coef=args.mole_balance,
    ).to(device)
    mole_model.gate.float()

    if is_distributed:
        mole_model = torch.nn.parallel.DistributedDataParallel(
            mole_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )
    base_eval_model = mole_model.module if hasattr(mole_model, "module") else mole_model

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {"params": [p for n, p in mole_model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in mole_model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    FT_optim = torch.optim.AdamW(params, lr=args.lr)
    FT_sched = get_linear_schedule_with_warmup(FT_optim, int(args.warmup_ratio*len(ld_F_ddp)*max(1,args.finetune_epochs)),
                                               len(ld_F_ddp)*max(1,args.finetune_epochs))

    for ep in range(1, args.finetune_epochs + 1):
        if is_distributed and ft_sampler is not None: ft_sampler.set_epoch(ep)
        loss = finetune_one_epoch_mole(
            mole_model, ld_F_ddp, FT_optim, FT_sched, device,
            grad_clip=1.0, fp16=args.fp16, log_every=100
        )
        if is_main: print(f"[MoLE] epoch={ep} loss={loss:.4f}")

    if is_main:
        k_list = sorted(set(args.topk))
        mO = evaluate_causal(base_eval_model, ld_O, device, item_final_token_ids, k_list=k_list, fp16=args.fp16,
                             desc="Eval-O(mole)")
        mT = evaluate_causal(base_eval_model, ld_T, device, item_final_token_ids, k_list=k_list, fp16=args.fp16,
                             desc="Eval-T(mole)")
        with open(os.path.join(args.output_dir, "metrics_original_mole.json"), "w", encoding="utf-8") as f:
            json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test_mole.json"), "w", encoding="utf-8") as f:
            json.dump(mT, f, ensure_ascii=False, indent=2)

        adapter_dir = os.path.join(args.output_dir, "mole_adapters")
        ensure_dir(adapter_dir)

        # === FIX: unwrap DDP ===
        peft_to_save = lora_model.module if hasattr(lora_model, "module") else lora_model
        peft_to_save.save_pretrained(adapter_dir, selected_adapters=adapter_names)

        torch.save({
            'gate_state': base_eval_model.gate.state_dict(),
            'adapter_names': adapter_names,
            'temperature': args.mole_temp,
            'balance_coef': args.mole_balance,
        }, os.path.join(adapter_dir, 'mole_gate.pt'))

        print("[Done] MoLE:", mO, mT)

    try:
        del FT_optim, FT_sched
    except Exception:
        pass
    _free_cuda(mole_model)
    if is_distributed: dist.barrier()

def stage_raie(args, device, is_distributed, local_rank, is_main, load_dtype):
    base_dir = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[stage raie] base_with_new_tokens not found at {base_dir}")
    default_dir = os.path.join(args.output_dir, "default")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError(f"[stage raie] default adapter dir not found at {default_dir}")

    tok_raie, raie_model = fresh_lora_model_from_base(
        base_dir, device, args.lora_r, args.lora_alpha, args.lora_dropout, load_dtype
    )
    raie_model.load_adapter(default_dir, adapter_name="default", is_trainable=True); raie_model.set_adapter("default")

    # Anchor finetuning (DDP wrapped, regularization on unwrapped module).
    if is_distributed:
        raie_model = torch.nn.parallel.DistributedDataParallel(
            raie_model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False
        )
    rm = raie_model.module if hasattr(raie_model, "module") else raie_model

    # Data
    mid2idx, n_items = build_or_load_mid2idx(args, is_main=is_main)
    ds_trainO, ds_O, ds_T, ds_F, final_item_tokens = make_datasets_for_stage(args, tok_raie, mid2idx, n_items)
    pad_id = tok_raie.pad_token_id
    collate = lambda batch: collate_batch(batch, pad_id)
    item_final_token_ids = [tok_raie.convert_tokens_to_ids(t) for t in final_item_tokens]

    ft_sampler_raie = DistributedSampler(ds_F, num_replicas=(dist.get_world_size() if is_distributed else 1),
                                         rank=(dist.get_rank() if is_distributed else 0), shuffle=True) if is_distributed else None
    ld_F_raie = DataLoader(ds_F, batch_size=args.finetune_batch_size,
                           shuffle=(ft_sampler_raie is None), sampler=ft_sampler_raie,
                           num_workers=2, pin_memory=True, collate_fn=collate)

    FT_optim = torch.optim.AdamW([p for p in raie_model.parameters() if p.requires_grad], lr=args.lr)
    FT_sched = get_linear_schedule_with_warmup(FT_optim, int(args.warmup_ratio*len(ld_F_raie)*max(1,args.finetune_epochs)),
                                               len(ld_F_raie)*max(1,args.finetune_epochs))

    Ein = rm.get_input_embeddings()
    E0 = Ein.weight.detach().clone()
    f_item_token_ids = set()
    for r in ds_F.examples:
        tgt_idx = int(r["target_item_idx"].item())
        tid = item_final_token_ids[tgt_idx]
        f_item_token_ids.add(int(tid))
    tune_token_ids = torch.tensor(sorted(list(f_item_token_ids)), device=Ein.weight.device, dtype=torch.long) \
                      if len(f_item_token_ids) > 0 else None
    Ein.weight.requires_grad_(True)
    if tune_token_ids is not None and tune_token_ids.numel() > 0:
        grad_mask = torch.zeros_like(Ein.weight, dtype=torch.bool); grad_mask[tune_token_ids] = True
        def _hook(g): return g.masked_fill(~grad_mask, 0)
        Ein.weight.register_hook(_hook)
        Eout = rm.get_output_embeddings()
        if Eout is not None and Eout.weight is not Ein.weight:
            Eout.weight.requires_grad_(True)
            Eout.weight.register_hook(_hook)

    for ep in range(args.finetune_epochs):
        if is_distributed and ft_sampler_raie is not None:
            ft_sampler_raie.set_epoch(ep)
        loss = finetune_one_epoch_causal_anchor(
            raie_model, ld_F_raie, device, item_final_token_ids,
            optimizer=FT_optim, scheduler=FT_sched, grad_clip=1.0,
            E0=E0, tune_token_ids=tune_token_ids, lambda_anchor=args.lambda_anchor, fp16=args.fp16,
            embed_model=rm
        )
        if is_main: print(f"[RAIE][F-anchor] epoch={ep+1} loss={loss:.4f}")

    # Save default adapter (for region adapter init/fallback).
    if is_main:
        default_dir2 = os.path.join(args.output_dir, "default_for_raie")
        ensure_dir(default_dir2)
        rm.save_pretrained(default_dir2, selected_adapters=["default"])

    # Fit regions + per-region training (only on rank0, not DDP).
    bank = RegionBank(
        model=rm, tok=tok_raie,
        item_final_token_ids=item_final_token_ids,
        original_ds=ds_trainO, device=device, out_dir=args.output_dir,
        K=args.K, q=args.q, tau=args.tau, gamma=args.gamma, gap_thr=args.gap_thr,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )

    if is_main:
        _ = bank.fit_regions_on_original(seed=args.seed)
        region_buckets, soft_pairs = bank.map_finetune(ds_F, pool_min=120, bic_gain=1e4)
    else:
        bank.C = bank.R = bank.sig = bank.kappa = bank.pi = None
        bank.S = bank.n = None
        bank.orig_idx_by_k = {}
        region_buckets = None; soft_pairs = None
    # ==== Barrier here to sync all ranks before creating/using the gloo group ====
    if is_distributed:
        dist.barrier()

    bank.C = bcast_object(bank.C, src=0)
    bank.R = bcast_object(bank.R, src=0)
    bank.sig = bcast_object(bank.sig, src=0)
    bank.kappa = bcast_object(bank.kappa, src=0)
    bank.pi = bcast_object(bank.pi, src=0)
    bank.S = bcast_object(bank.S, src=0)
    bank.n = bcast_object(bank.n, src=0)
    bank.orig_idx_by_k = bcast_object(bank.orig_idx_by_k, src=0)
    region_buckets = bcast_object(region_buckets, src=0)
    soft_pairs = bcast_object(soft_pairs, src=0)

    # -- Region training runs only on rank0 to avoid cross-rank synchronization --
    if is_main:
        bank.train_regions(
            finetune_ds=ds_F, collate_fn=collate,
            optimizer_ctor=lambda m: torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr),
            scheduler_ctor=lambda optim, steps_per_epoch, epochs: get_linear_schedule_with_warmup(
                optim, int(args.warmup_ratio * steps_per_epoch * max(1, epochs)), steps_per_epoch * max(1, epochs)
            ),
            epochs_per_region=args.raie_epochs_per_region,
            batch_size=args.finetune_batch_size,
            orig_mix_ratio=args.orig_mix_ratio,
            region_buckets=region_buckets,  # Use full buckets (including new regions).
            soft_pairs=soft_pairs,
            soft_rep_factor=args.soft_rep_factor, fp16=args.fp16
        )

    if is_distributed:
        dist.barrier()  # Wait for rank0 to finish training and saving region adapters.

    if is_main:
        k_list = sorted(set(args.topk))
        ld_O = DataLoader(ds_O, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
        ld_T = DataLoader(ds_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
        mO = bank.route_and_eval(ds_O, collate, k_list=k_list, root=args.output_dir, batch_size=256, fp16=args.fp16)
        mT = bank.route_and_eval(ds_T, collate, k_list=k_list, root=args.output_dir, batch_size=256, fp16=args.fp16)
        with open(os.path.join(args.output_dir, "metrics_original_raie.json"), "w", encoding="utf-8") as f: json.dump(mO, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "metrics_test_raie.json"), "w", encoding="utf-8") as f: json.dump(mT, f, ensure_ascii=False, indent=2)
        print("[Done] RAIE:", mO, mT)

    _free_cuda(raie_model, rm)
    if is_distributed: dist.barrier()

# ------------------------------
# Main entry (with --stage routing)
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', type=str, default='/home/zj/model/Llama-2-7b-hf')
    ap.add_argument('--data_dir', type=str, default='/home/zj/code/Amazon_toys/')
    ap.add_argument('--output_dir', type=str, default='./runs/openP5_Amazon_toys')

    ap.add_argument('--train_jsonl_path', type=str, default='')
    ap.add_argument('--original_jsonl_path', type=str, default='')
    ap.add_argument('--test_jsonl_path', type=str, default='')
    ap.add_argument('--item_ids_path', type=str, default='')
    ap.add_argument('--finetune_jsonl_path', type=str, default='')

    # Stage control options.
    ap.add_argument('--stage', type=str, choices=['pre','lora','replay','lwf','lsat','raie','mole','all'], default='raie',
                    help="Select the stage to run")
    ap.add_argument('--resume_base_dir', type=str, default='',
                    help="Path to saved base_with_new_tokens; defaults to output_dir/base_with_new_tokens")
    ap.add_argument('--skip_pre_if_exists', action='store_true', default=True,
                    help="When stage=all and base_with_new_tokens exists, skip pre (default on)")

    ap.add_argument('--item_indexing', type=str, choices=['sequential', 'random', 'collaborative'], default='sequential')
    ap.add_argument('--min_user_len', type=int, default=5)
    ap.add_argument('--max_history', type=int, default=10)

    # Training
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup_ratio', type=float, default=0.05)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--fp16', type=bool, default=False)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--final_token_only_loss', action='store_true', default=True)

    # LoRA
    ap.add_argument('--use_lora', action='store_true', default=True)
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--grad_checkpointing', action='store_true')

    # Evaluation
    ap.add_argument('--topk', type=int, nargs='+', default=[5, 10, 20])

    # Shared for F-stage methods
    ap.add_argument('--finetune_batch_size', type=int, default=32)
    ap.add_argument('--finetune_epochs', type=int, default=3)

    # Replay
    ap.add_argument('--replay_ratio', type=float, default=0.3)

    # LwF
    ap.add_argument('--lwf_T', type=float, default=2.0)
    ap.add_argument('--lwf_alpha', type=float, default=0.5)

    # LSAT
    ap.add_argument('--lsat_alpha', type=float, default=0.6)
    ap.add_argument('--lsat_long_epochs', type=int, default=1)
    ap.add_argument('--lsat_short_epochs', type=int, default=3)

    # RAIE
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--gap_thr", type=float, default=0.02)
    ap.add_argument("--orig_mix_ratio", type=float, default=0.7)
    ap.add_argument("--soft_rep_factor", type=int, default=2)
    ap.add_argument("--raie_epochs_per_region", type=int, default=3)
    ap.add_argument("--lambda_anchor", type=float, default=1e-4)

    # MoLE
    ap.add_argument('--mole_num_experts', type=int, default=3)
    ap.add_argument('--mole_gating_hidden', type=int, default=128)
    ap.add_argument('--mole_temp', type=float, default=0.7)
    ap.add_argument('--mole_balance', type=float, default=0.01)

    args = ap.parse_args()

    # DDP init
    is_distributed, rank, world_size, local_rank = init_distributed()
    is_main = (rank == 0)
    set_seed(args.seed, rank)

    # Dtype control (reduce reload memory).
    load_dtype = torch.float16

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}' if is_distributed else 'cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if is_main:
        ensure_dir(args.output_dir)

    # Stage execution
    if args.stage in ("all", "pre"):
        need_pre = True
        if args.stage == "all" and args.skip_pre_if_exists:
            base_dir_default = args.resume_base_dir or os.path.join(args.output_dir, "base_with_new_tokens")
            if os.path.isdir(base_dir_default):
                need_pre = False
                if is_main:
                    print(f"[SKIP] Detected existing base at {base_dir_default}, skip pre-stage as requested.")
        if need_pre:
            stage_pre(args, device, is_distributed, local_rank, is_main, load_dtype)
        else:
            if is_distributed: dist.barrier()

    if args.stage in ("all", "lora"):
        stage_lora(args, device, is_distributed, local_rank, is_main, load_dtype)

    if args.stage in ("all", "replay"):
        stage_replay(args, device, is_distributed, local_rank, is_main, load_dtype)

    if args.stage in ("all", "lwf"):
        stage_lwf(args, device, is_distributed, local_rank, is_main, load_dtype)

    if args.stage in ("all", "lsat"):
        stage_lsat(args, device, is_distributed, local_rank, is_main, load_dtype)

    if args.stage in ("all", "mole"):
        stage_mole(args, device, is_distributed, local_rank, is_main, load_dtype)

    if args.stage in ("all", "raie"):
        if is_distributed and rank != 0:
            print(f"[INFO] RAIE runs single process on rank0; exiting rank {rank}.", flush=True)
            aggressive_gc()
            sys.exit(0)

        if is_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            is_distributed = False

        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True

        stage_raie(args, device, False, 0, True, load_dtype)

    if is_main:
        print("\n=== SELECTED STAGE(S) FINISHED ===")
        print("Results saved in:", args.output_dir)

    aggressive_gc()
    cleanup_distributed()

if __name__ == '__main__':
    main()
