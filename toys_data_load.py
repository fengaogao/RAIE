import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def format_item_token(mid: int) -> str:
    return f"<item_{mid}>"


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_meta_asins(meta_path: str) -> set:
    asins = set()
    if not os.path.exists(meta_path):
        return asins
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            asin = r.get("asin")
            if asin:
                asins.add(asin)
    return asins


def load_amazon_toys_ratings(
    meta_path: str,
    reviews_path: str,
    min_item_freq: int = 5,
) -> Dict[int, List[Tuple[int, int]]]:
    """
    读取 meta_Toys_and_Games.json 与 Toys_and_Games_5.json（每行一个 JSON）。
    字段：reviewerID(str), asin(str), overall(float), unixReviewTime(int)
    规则：仅保留 overall >= 4.0 的正反馈；删除出现次数 < min_item_freq 的 item。
    reviewerID/asin 会映射为连续整数 uid/mid。
    """
    valid_asins = load_meta_asins(meta_path)
    raw_user2seq: Dict[str, List[Tuple[str, int]]] = {}
    item_counts: Counter[str] = Counter()

    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            overall = r.get("overall")
            if overall is None:
                continue
            try:
                overall = float(overall)
            except Exception:
                continue
            if overall < 4.0:
                continue

            reviewer = r.get("reviewerID")
            asin = r.get("asin")
            ts = r.get("unixReviewTime")
            if reviewer is None or asin is None or ts is None:
                continue
            if valid_asins and asin not in valid_asins:
                continue
            try:
                ts = int(ts)
            except Exception:
                continue

            raw_user2seq.setdefault(reviewer, []).append((asin, ts))
            item_counts[asin] += 1

    keep_asins = {asin for asin, cnt in item_counts.items() if cnt >= min_item_freq}

    uid_map: Dict[str, int] = {}
    asin_map: Dict[str, int] = {}
    next_uid, next_mid = 1, 1

    user2seq: Dict[int, List[Tuple[int, int]]] = {}
    for reviewer, pairs in raw_user2seq.items():
        filtered_pairs = [(asin, ts) for asin, ts in pairs if asin in keep_asins]
        if not filtered_pairs:
            continue
        if reviewer not in uid_map:
            uid_map[reviewer] = next_uid
            next_uid += 1
        uid = uid_map[reviewer]
        for asin, ts in filtered_pairs:
            if asin not in asin_map:
                asin_map[asin] = next_mid
                next_mid += 1
            mid = asin_map[asin]
            user2seq.setdefault(uid, []).append((mid, ts))

    for u in user2seq:
        user2seq[u].sort(key=lambda x: x[1])

    return user2seq


def sliding_windows(seq: List[int], win_len: int, stride: int = 1, align: str = "end"):
    """
    右对齐滑窗：确保最后一次交互（seq[-1]）作为目标被包含。
    返回: (ctx, tgt, tgt_idx)
    """
    n = len(seq)
    if n < win_len:
        return
    if align == "end":
        # 让 range() 的最后一个 i 命中 n - win_len
        start = (n - win_len) % stride
    else:  # "start"
        start = 0

    for i in range(start, n - win_len + 1, stride):
        window = seq[i:i + win_len]
        ctx, tgt = window[:-1], window[-1]
        yield ctx, tgt, (i + win_len - 1)


def build_examples_global_time_split(
    user2seq: Dict[int, List[Tuple[int, int]]],
    win_len: int = 10,
    stride: int = 1,
    q_orig: float = 0.5,
    q_test_start: float = 0.8,
):
    """
    基于全局时间分位（t0, t1）将序列切为 O/F/T 三段，并分别滑窗构造样本。
    保持与原版一致的输出键与文件结构。
    """
    ts_all = np.array([ts for pairs in user2seq.values() for _, ts in pairs], dtype=np.int64)
    assert ts_all.size > 0, "没有可用时间戳。"
    t0 = int(np.quantile(ts_all, q_orig))
    t1 = int(np.quantile(ts_all, q_test_start))
    if t1 <= t0:
        t1 = t0 + 1

    original, finetune, test = [], [], []
    users_o, items_o, inter_o = set(), set(), 0
    users_f, items_f, inter_f = set(), set(), 0
    users_t, items_t, inter_t = set(), set(), 0

    original_stride1 = []
    all_items = set()

    for uid, pairs in user2seq.items():
        mids_o = [m for m, ts in pairs if ts < t0]
        tss_o = [ts for _, ts in [(m, ts) for m, ts in pairs if ts < t0]]

        mids_f = [m for m, ts in pairs if t0 <= ts < t1]
        tss_f = [ts for _, ts in [(m, ts) for m, ts in pairs if t0 <= ts < t1]]

        mids_t = [m for m, ts in pairs if ts >= t1]
        tss_t = [ts for _, ts in [(m, ts) for m, ts in pairs if ts >= t1]]

        if len(mids_o) > 0:
            users_o.add(uid)
            items_o.update(mids_o)
            inter_o += len(mids_o)
        if len(mids_f) > 0:
            users_f.add(uid)
            items_f.update(mids_f)
            inter_f += len(mids_f)
        if len(mids_t) > 0:
            users_t.add(uid)
            items_t.update(mids_t)
            inter_t += len(mids_t)

        if len(mids_o) >= win_len:
            for (ctx, tgt, tgt_idx) in sliding_windows(mids_o, win_len, stride, align="end"):
                ts = tss_o[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                original.append(
                    {
                        "user_id": uid,
                        "timestamp": int(ts),
                        "prompt": prompt,
                        "target": target,
                    }
                )
                all_items.update(ctx)
                all_items.add(tgt)

            for (ctx, tgt, tgt_idx) in sliding_windows(mids_o, win_len, 1, align="end"):
                ts = tss_o[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                original_stride1.append(
                    {
                        "user_id": uid,
                        "timestamp": int(ts),
                        "prompt": prompt,
                        "target": target,
                    }
                )
                all_items.update(ctx)
                all_items.add(tgt)

        if len(mids_f) >= win_len:
            for (ctx, tgt, tgt_idx) in sliding_windows(mids_f, win_len, stride, align="end"):
                ts = tss_f[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                finetune.append(
                    {
                        "user_id": uid,
                        "timestamp": int(ts),
                        "prompt": prompt,
                        "target": target,
                    }
                )
                all_items.update(ctx)
                all_items.add(tgt)

        if len(mids_t) >= win_len:
            for (ctx, tgt, tgt_idx) in sliding_windows(mids_t, win_len, stride, align="end"):
                ts = tss_t[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                test.append(
                    {
                        "user_id": uid,
                        "timestamp": int(ts),
                        "prompt": prompt,
                        "target": target,
                    }
                )
                all_items.update(ctx)
                all_items.add(tgt)

    stats = {
        "O": {"users": len(users_o), "items": len(items_o), "interactions": inter_o},
        "F": {"users": len(users_f), "items": len(items_f), "interactions": inter_f},
        "T": {"users": len(users_t), "items": len(items_t), "interactions": inter_t},
    }

    return {
        "original": original,
        "finetune": finetune,
        "test": test,
        "original_stride1": original_stride1,
        "all_items": sorted(all_items),
        "t0": t0,
        "t1": t1,
        "stats": stats,
    }


def save_stream(
    path,
    user2seq: Dict[int, List[Tuple[int, int]]],
    t0: int,
    t1: int,
    which: str,
):
    """
    which: 'original' | 'finetune'
    original: ts < t0
    finetune: t0 <= ts < t1
    """
    rows = []
    for uid, pairs in user2seq.items():
        if which == "original":
            seg = [(m, ts) for m, ts in pairs if ts < t0]
        elif which == "finetune":
            seg = [(m, ts) for m, ts in pairs if t0 <= ts < t1]
        else:
            raise ValueError("which must be 'original' or 'finetune'")
        if len(seg) < 2:  # 至少要有序列
            continue
        mids = [m for m, _ in seg]
        tss = [ts for _, ts in seg]
        rows.append(
            {
                "user_id": uid,
                "items": " ".join(format_item_token(m) for m in mids),
                "timestamps": " ".join(str(ts) for ts in tss),
            }
        )
    save_jsonl(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zj/code/Amazon_toys/")
    parser.add_argument("--output_dir", type=str, default="/home/zj/code/Amazon_toys/")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5, help="O/F/T 段滑窗步长（原版默认 5）")
    parser.add_argument("--q_orig", type=float, default=0.5, help="t0 分位点（按全局时间）")
    parser.add_argument("--q_test_start", type=float, default=0.8, help="t1 分位点（按全局时间）")
    parser.add_argument("--min_item_freq", type=int, default=5, help="item 出现次数阈值")
    parser.add_argument("--save_stream", action="store_true", default=True)
    args = parser.parse_args()

    meta_path = os.path.join(args.data_dir, "meta_Toys_and_Games.json")
    reviews_path = os.path.join(args.data_dir, "Toys_and_Games_5.json")

    if not os.path.exists(reviews_path):
        raise FileNotFoundError(reviews_path)

    print("Loading Amazon Toys and Games ...")
    user2seq = load_amazon_toys_ratings(
        meta_path,
        reviews_path,
        min_item_freq=args.min_item_freq,
    )

    print("Building examples ...")
    pack = build_examples_global_time_split(
        user2seq,
        win_len=args.max_len + 1,
        stride=args.stride,
        q_orig=args.q_orig,
        q_test_start=args.q_test_start,
    )

    s = pack["stats"]
    print("[Block Stats]")
    print(f"O: users={s['O']['users']}, items={s['O']['items']}, interactions={s['O']['interactions']}")
    print(f"F: users={s['F']['users']}, items={s['F']['items']}, interactions={s['F']['interactions']}")
    print(f"T: users={s['T']['users']}, items={s['T']['items']}, interactions={s['T']['interactions']}")

    data_original_stride1 = pack["original_stride1"]
    data_original = pack["original"]
    data_finetune = pack["finetune"]
    data_test = pack["test"]
    item_ids = pack["all_items"]

    save_jsonl(os.path.join(args.output_dir, "original.jsonl"), data_original)
    save_jsonl(os.path.join(args.output_dir, "original_stride1.jsonl"), data_original_stride1)
    save_jsonl(os.path.join(args.output_dir, "finetune.jsonl"), data_finetune)
    save_jsonl(os.path.join(args.output_dir, "test.jsonl"), data_test)

    if args.save_stream:
        save_stream(
            os.path.join(args.output_dir, "original_stream.jsonl"),
            user2seq,
            pack["t0"],
            pack["t1"],
            which="original",
        )
        save_stream(
            os.path.join(args.output_dir, "finetune_stream.jsonl"),
            user2seq,
            pack["t0"],
            pack["t1"],
            which="finetune",
        )

    with open(os.path.join(args.output_dir, "item_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"item_ids": item_ids}, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": {
                    "original": len(data_original),
                    "finetune": len(data_finetune),
                    "test": len(data_test),
                    "num_items": len(item_ids),
                },
                "t0": pack["t0"],
                "t1": pack["t1"],
                "schema": {"prompt": "str", "target": "str", "user_id": "int", "timestamp": "int"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Done.")


if __name__ == "__main__":
    main()
