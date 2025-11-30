import argparse
import importlib.util
import json
import os
import types
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Tuple
import shutil
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, get_linear_schedule_with_warmup

import numpy as np
from peft import PeftModel


# -----------------------------------------------
# Utilities for loading the original Bert4Rec_RAIE
# -----------------------------------------------

def _load_bert4rec_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("Bert4Rec_RAIE_ALL_NEW.py", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@dataclass
class StageResult:
    finetune_stage: str
    predict_stage: str
    predict_metrics: Dict[str, float]
    original_metrics: Dict[str, float]
    model_dir: str


class MultiStageRunner:
    """Orchestrates sequential fine-tune/eval cycles."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.module = _load_bert4rec_module(os.path.join(os.path.dirname(__file__), "Bert4Rec_RAIE_ALL_NEW.py"))

        # shared vocab + collators
        self.token2id, self.id2token, self.item_token_ids = self.module.load_item_vocab(args.data_dir)
        self.pad_id = self.token2id["[PAD]"]
        self.cls_id = self.token2id["[CLS]"]
        self.mask_id = self.token2id["[MASK]"]

        self.train_ds = self.module.StreamDataset(os.path.join(args.data_dir, "original_stream.jsonl"))
        self.train_collator = self.module.ClozeTrainCollator(
            self.token2id, args.max_len, self.pad_id, self.mask_id, self.cls_id, args.mask_prob
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.train_collator,
        )

        self.eval_collator = self.module.EvalCollator(
            self.token2id, args.max_len, self.mask_id, self.cls_id, self.pad_id
        )
        self.ft_collator = self.module.EvalCollator(
            self.token2id,
            args.max_len,
            self.mask_id,
            self.cls_id,
            self.pad_id,
            do_prompt_mask=args.do_prompt_mask,
            prompt_mask_prob=args.prompt_mask_prob,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Model helpers
    # -------------------------
    def _init_base_model(self) -> BertForMaskedLM:
        cfg = BertConfig(
            vocab_size=len(self.id2token),
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_hidden_layers,
            num_attention_heads=self.args.num_attention_heads,
            intermediate_size=self.args.intermediate_size,
            max_position_embeddings=self.args.max_len + 2,
            hidden_dropout_prob=self.args.dropout,
            attention_probs_dropout_prob=self.args.dropout,
            pad_token_id=self.pad_id,
        )
        return BertForMaskedLM(cfg).to(self.device)

    def _apply_lora(self, base_model: torch.nn.Module):
        target_modules = [s.strip() for s in self.args.lora_target.split(",") if s.strip()]
        lora_cfg = self.module.LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=self.module.TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
        )
        return self.module.get_peft_model(base_model, lora_cfg)

    # -------------------------
    # Training primitives
    # -------------------------
    def _train_on_stream(self, model: torch.nn.Module, epochs: int = None):
        epochs = self.args.epochs if epochs is None else epochs
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = torch.optim.AdamW(params, lr=self.args.lr)
        total_steps = len(self.train_loader) * max(epochs, 1)
        sched = get_linear_schedule_with_warmup(
            optim, int(total_steps * self.args.warmup_ratio), total_steps
        )
        for ep in range(1, epochs + 1):
            tr = self.module.train_one_epoch(
                model,
                self.train_loader,
                optim,
                sched,
                self.device,
                self.item_token_ids,
                objective=self.args.objective,
                grad_clip=self.args.grad_clip,
            )
            print(f"[O][Epoch {ep}] train_loss={tr['train_loss']:.4f}")

    def _finetune_general_lora(
        self, lora_model: torch.nn.Module, ft_loader: DataLoader, replay_buf=None, teacher=None
    ) -> float:
        ft_params = [p for p in lora_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        loss = 0.0
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_lora(
                lora_model,
                ft_loader,
                ft_optim,
                ft_sched,
                self.device,
                self.item_token_ids,
                self.pad_id,
                self.cls_id,
                self.mask_id,
                plugin=self.args.mode,
                replay_buf=replay_buf,
                teacher=teacher,
                lwf_T=self.args.lwf_T,
                lwf_alpha=self.args.lwf_alpha,
                grad_clip=self.args.grad_clip,
                replay_ratio=self.args.replay_ratio,
                log_every=100,
            )
            print(f"[F][{self.args.mode}][Epoch {ep}] loss={loss:.4f}")
        return loss

    # -------------------------
    # Stage execution
    # -------------------------
    def run(self) -> List[StageResult]:
        base_model = self._init_base_model()
        self._train_on_stream(base_model)
        base_dir = os.path.join(self.args.output_dir, "base_model")
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        base_model.save_pretrained(base_dir)
        with open(os.path.join(base_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"id2token": self.id2token}, f, ensure_ascii=False, indent=2)

        current_model_dir = base_dir
        results: List[StageResult] = []

        ft_stages = self.args.stages.split(",")
        pred_stages = self.args.pred_stages.split(",")
        if len(ft_stages) != len(pred_stages):
            raise ValueError("--stages and --pred_stages must align")

        original_eval_ds = self.module.NextItemDataset(os.path.join(self.args.data_dir, "original.jsonl"))
        self.original_route_ds = original_eval_ds
        original_loader = DataLoader(
            original_eval_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self.eval_collator,
        )

        if not ft_stages:
            raise ValueError("--stages must not be empty")

        initial_pred_path = os.path.join(self.args.data_dir, f"{ft_stages[0]}.jsonl")
        if not os.path.exists(initial_pred_path):
            raise FileNotFoundError(initial_pred_path)
        initial_pred_loader = DataLoader(
            self.module.NextItemDataset(initial_pred_path),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self.eval_collator,
        )
        base_eval_fn = self.module.evaluate
        initial_predict_metrics = base_eval_fn(
            base_model,
            initial_pred_loader,
            self.device,
            self.item_token_ids,
            topk=tuple(int(x) for x in self.args.topk.split(",")),
        )
        initial_original_metrics = base_eval_fn(
            base_model,
            original_loader,
            self.device,
            self.item_token_ids,
            topk=tuple(int(x) for x in self.args.topk.split(",")),
        )

        results.append(
            StageResult(
                finetune_stage="original_stream",
                predict_stage=ft_stages[0],
                predict_metrics=initial_predict_metrics,
                original_metrics=initial_original_metrics,
                model_dir=base_dir,
            )
        )
        print(f"[Eval][original_stream->{ft_stages[0]}] {initial_predict_metrics}")
        print(f"[Eval][original_stream->original] {initial_original_metrics}")

        topk_tuple = tuple(int(x) for x in self.args.topk.split(","))

        for idx, (ft_name, pred_name) in enumerate(zip(ft_stages, pred_stages), start=1):
            print(f"===== Stage {idx}: finetune {ft_name} -> predict {pred_name} =====")
            ft_path = os.path.join(self.args.data_dir, f"{ft_name}.jsonl")
            pred_path = os.path.join(self.args.data_dir, f"{pred_name}.jsonl")
            if not os.path.exists(ft_path):
                raise FileNotFoundError(ft_path)
            if not os.path.exists(pred_path):
                raise FileNotFoundError(pred_path)

            ft_ds = self.module.NextItemDataset(ft_path)
            ft_loader = DataLoader(
                ft_ds,
                batch_size=self.args.finetune_batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=self.ft_collator,
            )
            pred_ds = self.module.NextItemDataset(pred_path)
            pred_loader = DataLoader(
                pred_ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=self.eval_collator,
            )

            stage_dir = os.path.join(self.args.output_dir, f"stage_{ft_name}")
            if os.path.isdir(stage_dir):
                shutil.rmtree(stage_dir)
            os.makedirs(stage_dir, exist_ok=True)

            if self.args.mode == "base":
                model = BertForMaskedLM.from_pretrained(self._base_path(current_model_dir)).to(self.device)
                self._finetune_base(model, ft_loader)
                predict_metrics = self.module.evaluate(
                    model, pred_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                original_metrics = self.module.evaluate(
                    model, original_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                model.save_pretrained(stage_dir)
                current_model_dir = stage_dir
            elif self.args.mode in {"lora", "lora_replay", "lora_lwf"}:
                base_model = BertForMaskedLM.from_pretrained(self._base_path(current_model_dir)).to(self.device)
                lora_model = self._apply_lora(base_model)
                eval_model = self._finetune_lora_based(lora_model, ft_loader)
                predict_metrics = self.module.evaluate(
                    eval_model, pred_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                original_metrics = self.module.evaluate(
                    eval_model, original_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                eval_model.save_pretrained(stage_dir)
                current_model_dir = stage_dir
            elif self.args.mode == "lsat":
                lora_model, _ = self._load_lsat_model(current_model_dir)
                eval_model = self._train_lsat_stage(lora_model, ft_loader)
                predict_metrics = self.module.evaluate_lsat(
                    eval_model,
                    pred_loader,
                    self.device,
                    self.item_token_ids,
                    long_adapter="default",
                    short_adapter="short_term",
                    alpha=self.args.lsat_alpha,
                    topk=topk_tuple,
                )
                original_metrics = self.module.evaluate_lsat(
                    eval_model,
                    original_loader,
                    self.device,
                    self.item_token_ids,
                    long_adapter="default",
                    short_adapter="short_term",
                    alpha=self.args.lsat_alpha,
                    topk=topk_tuple,
                )
                self._save_lora_checkpoint(
                    eval_model, stage_dir, ["default", "short_term"], meta={"mode": "lsat"}
                )
                current_model_dir = stage_dir
            elif self.args.mode == "ebpr":
                base_model = BertForMaskedLM.from_pretrained(self._base_path(current_model_dir)).to(self.device)
                edit_pairs = self._mine_edit_pairs_ebpr(base_model, ft_ds)
                eval_model = self._finetune_ebpr(base_model, edit_pairs)
                predict_metrics = self.module.evaluate(
                    eval_model, pred_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                original_metrics = self.module.evaluate(
                    eval_model, original_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                eval_model.save_pretrained(stage_dir)
                current_model_dir = stage_dir
            elif self.args.mode == "mole":
                mole_model, adapter_names = self._load_mole_model(current_model_dir)

                ft_params = [p for p in mole_model.parameters() if p.requires_grad]
                ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
                ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
                ft_sched = get_linear_schedule_with_warmup(
                    ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
                )
                for ep in range(1, self.args.finetune_epochs + 1):
                    loss = self.module.finetune_one_epoch_mole(
                        mole_model, ft_loader, ft_optim, ft_sched, self.device, grad_clip=self.args.grad_clip
                    )
                    print(f"[MoLE][F][Epoch {ep}] loss={loss:.4f}")

                predict_metrics = self.module.evaluate(
                    mole_model, pred_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                original_metrics = self.module.evaluate(
                    mole_model, original_loader, self.device, self.item_token_ids, topk=topk_tuple
                )
                self._save_mole_checkpoint(mole_model, stage_dir, adapter_names)
                current_model_dir = stage_dir
            elif self.args.mode == "raie":
                lora_model, adapter_names, _ = self._load_raie_model(current_model_dir)

                emb_in = lora_model.get_input_embeddings()
                E0 = emb_in.weight.detach().clone()
                tune_ids = sorted(
                    {
                        self.token2id.get(ex.target_token, None)
                        for ex in ft_ds.examples
                        if self.token2id.get(ex.target_token, None) in self.item_token_ids
                    }
                )
                ids_t = (
                    torch.tensor(tune_ids, device=emb_in.weight.device) if len(tune_ids) > 0 else None
                )
                lambda_anchor = 1e-4

                emb_in.weight.requires_grad_(True)
                mask = torch.zeros_like(emb_in.weight, dtype=torch.bool)
                if len(tune_ids) > 0:
                    ids = torch.tensor(sorted(tune_ids), device=emb_in.weight.device, dtype=torch.long)
                    mask[ids] = True

                def grad_mask(g):
                    return g.masked_fill(~mask, 0)

                emb_in.weight.register_hook(grad_mask)
                emb_out = lora_model.get_output_embeddings()
                if emb_out is not None and emb_out.weight is not emb_in.weight:
                    emb_out.weight.requires_grad_(True)
                    emb_out.weight.register_hook(grad_mask)

                ft_params = [p for p in lora_model.parameters() if p.requires_grad]
                ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
                ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
                ft_sched = get_linear_schedule_with_warmup(
                    ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
                )
                for ep in range(self.args.finetune_epochs):
                    ft_loss = self.module.finetune_one_epoch_bert(
                        lora_model,
                        ft_loader,
                        ft_optim,
                        ft_sched,
                        self.device,
                        self.item_token_ids,
                        grad_clip=self.args.grad_clip,
                        E0=E0,
                        ids_t=ids_t,
                        lambda_anchor=lambda_anchor,
                    )
                    print(f"[RAIE][F][Epoch {ep+1}] loss={ft_loss:.4f}")

                emb_in.weight.requires_grad_(False)
                if emb_out is not None and emb_out.weight is not emb_in.weight:
                    emb_out.weight.requires_grad_(False)

                default_dir = os.path.join(stage_dir, "default")
                os.makedirs(default_dir, exist_ok=True)
                lora_model.save_pretrained(default_dir, selected_adapters=["default"])

                bank = self.module.RegionBank(
                    model=lora_model,
                    token2id=self.token2id,
                    item_token_ids=self.item_token_ids,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    max_len=self.args.max_len,
                    device=self.device,
                    out_dir=stage_dir,
                    K=self.args.K,
                    q=self.args.q,
                    T_low=self.args.T_low,
                    T_high=self.args.T_high,
                    tau=self.args.tau,
                    gamma=self.args.gamma,
                    gap_thr=self.args.gap_thr,
                    lora_r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=tuple(t.strip() for t in self.args.lora_target.split(",") if t.strip()),
                    original_ds=self.original_route_ds,
                )

                restored = self._restore_raie_regions(bank, current_model_dir)
                if not restored:
                    _ = bank.fit_regions_on_original(self.original_route_ds)
                if not bank.orig_idx_by_k:
                    self._rebuild_orig_assignments(bank)

                region_buckets, soft_pairs = bank.map_finetune(ft_ds)

                def _optim_ctor(m):
                    no_decay = ["bias", "LayerNorm.weight"]
                    groups = [
                        {
                            "params": [
                                p for n, p in m.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in m.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                    return torch.optim.AdamW(groups, lr=self.args.finetune_lr)

                def _sched_ctor(optim, steps_per_epoch, epochs):
                    total = steps_per_epoch * epochs
                    warm = int(total * self.args.warmup_ratio)
                    return get_linear_schedule_with_warmup(optim, num_warmup_steps=warm, num_training_steps=total)

                bank.train_regions(
                    finetune_ds=ft_ds,
                    finetune_collator=self.ft_collator,
                    optimizer_ctor=_optim_ctor,
                    scheduler_ctor=_sched_ctor,
                    epochs_per_region=self.args.finetune_epochs,
                    batch_size=self.args.finetune_batch_size,
                    orig_mix_ratio=1,
                    region_buckets=region_buckets,
                    soft_pairs=soft_pairs,
                )

                stats_pred = bank.route_and_eval(
                    pred_ds, self.eval_collator, self.item_token_ids, self.module._eval_fn_for_raie, k_list=topk_tuple
                )
                stats_orig = bank.route_and_eval(
                    original_eval_ds, self.eval_collator, self.item_token_ids, self.module._eval_fn_for_raie, k_list=topk_tuple
                )

                def _to_global(stats):
                    import pandas as pd

                    df = pd.DataFrame(stats)
                    w = df["num_samples"]
                    W = max(1, w.sum())
                    return {m: float((df[m] * w).sum() / W) for m in df.columns if m.startswith(("Recall@", "NDCG@"))}

                predict_metrics = _to_global(stats_pred)
                original_metrics = _to_global(stats_orig)

                adapter_names = list(getattr(lora_model, "peft_config", {}).keys())
                self._save_lora_checkpoint(lora_model, stage_dir, adapter_names, meta={"mode": "raie"})
                self._save_raie_regions(bank, stage_dir)
                current_model_dir = stage_dir
            else:  # pragma: no cover
                raise ValueError(f"Unsupported mode {self.args.mode}")

            results.append(
                StageResult(
                    finetune_stage=ft_name,
                    predict_stage=pred_name,
                    predict_metrics=predict_metrics,
                    original_metrics=original_metrics,
                    model_dir=stage_dir,
                )
            )
            print(f"[Eval][{ft_name}->{pred_name}] {predict_metrics}")
            print(f"[Eval][{ft_name}->original] {original_metrics}")

        self._save_summary(results)
        return results

    # -------------------------
    # Mode-specific helpers
    # -------------------------
    def _finetune_base(self, model: torch.nn.Module, ft_loader: DataLoader):
        ft_params = [p for p in model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_bert(
                model,
                ft_loader,
                ft_optim,
                ft_sched,
                self.device,
                self.item_token_ids,
                grad_clip=self.args.grad_clip,
                E0=None,
                ids_t=None,
                lambda_anchor=0.0,
            )
            print(f"[Base][F][Epoch {ep}] loss={loss:.4f}")

    def _finetune_lora_based(self, lora_model: torch.nn.Module, ft_loader: DataLoader):
        if self.args.mode == "lsat":
            long_adapter = "default"
            short_adapter = "short_term"
            if short_adapter not in getattr(lora_model, "peft_config", {}):
                lora_model.add_adapter(short_adapter, lora_model.peft_config[long_adapter])
            if hasattr(lora_model, "set_adapter") and hasattr(lora_model, "train_adapter"):
                lora_model.set_adapter(long_adapter)
                lora_model.train_adapter(long_adapter)
            self._train_on_stream(lora_model)
            if hasattr(lora_model, "set_adapter") and hasattr(lora_model, "train_adapter"):
                lora_model.set_adapter(short_adapter)
                lora_model.train_adapter(short_adapter)
            self._finetune_general_lora(lora_model, ft_loader)
            return lora_model

        replay_buf = None
        teacher = None
        if self.args.mode == "lora_replay":
            replay_buf = self.module.ReplayBuffer(self.train_ds, self.train_collator, self.args.finetune_batch_size)
        if self.args.mode == "lora_lwf":
            teacher = BertForMaskedLM.from_pretrained(os.path.join(self.args.output_dir, "base_model")).to(self.device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        self._finetune_general_lora(lora_model, ft_loader, replay_buf=replay_buf, teacher=teacher)
        return lora_model

    def _finetune_mole(self, base_model: torch.nn.Module, ft_loader: DataLoader):
        lora_model = self._apply_lora(base_model)
        adapters = ["default"] + [f"expert_{i}" for i in range(self.args.mole_num_experts - 1)]
        for name in adapters[1:]:
            lora_model.add_adapter(name, lora_model.peft_config["default"])
        mole_model = self.module.MoLEMixture(
            lora_model,
            adapter_names=tuple(adapters),
            hidden_size=self.args.hidden_size,
            gate_hidden=self.args.mole_gating_hidden,
            temperature=self.args.mole_temp,
            balance_coef=self.args.mole_balance,
        ).to(self.device)

        ft_params = [p for p in mole_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_mole(
                mole_model, ft_loader, ft_optim, ft_sched, self.device, grad_clip=self.args.grad_clip
            )
            print(f"[MoLE][F][Epoch {ep}] loss={loss:.4f}")
        return mole_model

    # -------------------------
    # Checkpoint helpers
    # -------------------------
    def _base_path(self, model_dir: str) -> str:
        cand = os.path.join(model_dir, "base_model")
        return cand if os.path.isdir(cand) else model_dir

    def _adapter_path(self, model_dir: str) -> str:
        cand = os.path.join(model_dir, "adapters")
        return cand if os.path.isdir(cand) else model_dir

    def _load_meta(self, model_dir: str) -> Dict:
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_meta(self, model_dir: str, meta: Dict):
        with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _save_lora_checkpoint(self, lora_model: torch.nn.Module, stage_dir: str, adapter_names: List[str], meta: Dict):
        base_dir = os.path.join(stage_dir, "base_model")
        adapters_dir = os.path.join(stage_dir, "adapters")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(adapters_dir, exist_ok=True)

        base_model = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
        base_model.save_pretrained(base_dir)
        lora_model.save_pretrained(adapters_dir, selected_adapters=adapter_names)

        meta_to_save = {"mode": self.args.mode, "adapter_names": adapter_names} | meta
        self._save_meta(stage_dir, meta_to_save)

    # -------------------------
    # Mode-specific loaders/trainers
    # -------------------------
    def _load_lsat_model(self, model_dir: str):
        base_model = BertForMaskedLM.from_pretrained(self._base_path(model_dir)).to(self.device)
        adapter_dir = self._adapter_path(model_dir)
        meta = self._load_meta(model_dir)

        lora_model: torch.nn.Module
        if os.path.isdir(adapter_dir):
            try:
                lora_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
            except Exception:
                lora_model = self._apply_lora(base_model)
        else:
            lora_model = self._apply_lora(base_model)

        adapter_names = meta.get("adapter_names", ["default", "short_term"])
        for name in adapter_names:
            if name not in getattr(lora_model, "peft_config", {}):
                lora_model.add_adapter(name, lora_model.peft_config["default"])
        return lora_model, adapter_names

    def _train_lsat_stage(self, lora_model: torch.nn.Module, ft_loader: DataLoader):
        if hasattr(lora_model, "set_adapter") and hasattr(lora_model, "train_adapter"):
            lora_model.set_adapter("default")
            lora_model.train_adapter("default")
        self._train_on_stream(lora_model, epochs=self.args.lsat_long_epochs)

        if hasattr(lora_model, "set_adapter") and hasattr(lora_model, "train_adapter"):
            lora_model.set_adapter("short_term")
            lora_model.train_adapter("short_term")

        ft_params = [p for p in lora_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.lsat_short_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        for ep in range(1, self.args.lsat_short_epochs + 1):
            loss = self.module.finetune_one_epoch_lora(
                lora_model,
                ft_loader,
                ft_optim,
                ft_sched,
                self.device,
                self.item_token_ids,
                self.pad_id,
                self.cls_id,
                self.mask_id,
                plugin="none",
                grad_clip=self.args.grad_clip,
                log_every=100,
            )
            print(f"[LSAT][short_term][Epoch {ep}] loss={loss:.4f}")
        return lora_model

    def _mine_edit_pairs_ebpr(self, model: torch.nn.Module, ds) -> List[Dict[str, List[int]]]:
        model.eval()
        pairs: List[Dict[str, List[int]]] = []
        item_mask = torch.full((len(self.id2token),), float("-inf"), device=self.device)
        item_mask[self.item_token_ids] = 0.0

        with torch.no_grad():
            batch_size = self.args.ebpr_batch_size
            for start in range(0, len(ds.examples), batch_size):
                if len(pairs) >= self.args.ebpr_max_pairs:
                    break
                chunk = ds.examples[start : start + batch_size]
                encoded = self.eval_collator(chunk)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                labels = encoded["labels"].to(self.device)

                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                final_pos = attention_mask.sum(dim=1) - 1
                rows = torch.arange(input_ids.size(0), device=self.device)
                logits_final = logits[rows, final_pos, :] + item_mask

                max_k = self.args.ebpr_topk
                top_logits, top_idx = torch.topk(logits_final, k=max_k, dim=1)

                for i, ex in enumerate(chunk):
                    prompt_ids = [self.token2id.get(t, self.token2id["[UNK]"]) for t in ex.prompt_tokens[-self.args.max_len :]]
                    tgt_id = int(labels[i].item())
                    picked = 0
                    for pred_id in top_idx[i].tolist():
                        if pred_id == tgt_id:
                            continue
                        pairs.append({"prompt_ids": prompt_ids, "bad_id": pred_id})
                        picked += 1
                        if picked >= self.args.ebpr_per_user or len(pairs) >= self.args.ebpr_max_pairs:
                            break
        return pairs

    def _finetune_ebpr(self, model: torch.nn.Module, edit_pairs: List[Dict[str, List[int]]]):
        if not edit_pairs:
            print("[EBPR] No edit pairs mined; skipping finetune.")
            return model

        for p in model.parameters():
            p.requires_grad_(False)

        params = []
        emb_in = model.get_input_embeddings()
        if emb_in is not None:
            emb_in.weight.requires_grad_(True)
            params.append(emb_in.weight)
        emb_out = model.get_output_embeddings()
        if emb_out is not None and emb_out is not emb_in:
            emb_out.weight.requires_grad_(True)
            params.append(emb_out.weight)

        if not params:
            print("[EBPR] No trainable parameters found; aborting.")
            return model

        optim = torch.optim.Adam(params, lr=self.args.ebpr_lr)
        item_vocab = torch.tensor(self.item_token_ids, device=self.device, dtype=torch.long)

        def _build_batch(rows: List[Dict[str, List[int]]]):
            B = len(rows)
            seq_len = self.args.max_len + 2
            input_ids = torch.full((B, seq_len), self.pad_id, device=self.device, dtype=torch.long)
            attention_mask = torch.zeros((B, seq_len), device=self.device, dtype=torch.long)
            bad_ids = torch.zeros((B,), device=self.device, dtype=torch.long)
            for i, r in enumerate(rows):
                pos = 0
                input_ids[i, pos] = self.cls_id
                pos += 1
                prompt = r["prompt_ids"][-self.args.max_len :]
                if prompt:
                    plen = len(prompt)
                    input_ids[i, pos : pos + plen] = torch.tensor(prompt, device=self.device)
                    pos += plen
                input_ids[i, pos] = self.mask_id
                attention_mask[i, : pos + 1] = 1
                bad_ids[i] = int(r["bad_id"])
            return input_ids, attention_mask, bad_ids

        steps = max(1, self.args.ebpr_steps)
        neg_per_step = max(1, self.args.ebpr_neg_per_step)
        batch_size = min(self.args.ebpr_batch_size, len(edit_pairs))
        for step in range(steps):
            sample = random.sample(edit_pairs, k=batch_size) if len(edit_pairs) > batch_size else edit_pairs
            input_ids, attention_mask, bad_ids = _build_batch(sample)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            final_pos = attention_mask.sum(dim=1) - 1
            rows = torch.arange(input_ids.size(0), device=self.device)
            logits_final = logits[rows, final_pos, :]

            neg_idx = torch.randint(0, item_vocab.numel(), (batch_size, neg_per_step), device=self.device)
            neg_ids = item_vocab[neg_idx]
            neg_ids = torch.where(neg_ids == bad_ids.view(-1, 1), item_vocab[(neg_idx + 1) % item_vocab.numel()], neg_ids)

            s_bad = logits_final.gather(1, bad_ids.view(-1, 1)).expand(-1, neg_per_step)
            s_neg = logits_final.gather(1, neg_ids)
            loss = -torch.log(torch.sigmoid(s_neg - s_bad) + 1e-8).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.grad_clip and self.args.grad_clip > 0:
                for p in params:
                    torch.nn.utils.clip_grad_norm_(p, self.args.grad_clip)
            optim.step()
            print(f"[EBPR][Step {step+1}/{steps}] loss={float(loss.item()):.4f}")

        for p in params:
            p.requires_grad_(False)
        return model

    def _load_mole_model(self, model_dir: str):
        base_model = BertForMaskedLM.from_pretrained(self._base_path(model_dir)).to(self.device)
        adapter_dir = self._adapter_path(model_dir)
        meta = self._load_meta(model_dir)
        adapter_names = meta.get(
            "adapter_names", ["default"] + [f"expert_{i}" for i in range(self.args.mole_num_experts - 1)]
        )

        if os.path.isdir(adapter_dir):
            try:
                lora_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
            except Exception:
                lora_model = self._apply_lora(base_model)
        else:
            lora_model = self._apply_lora(base_model)

        for name in adapter_names:
            if name not in getattr(lora_model, "peft_config", {}):
                lora_model.add_adapter(name, lora_model.peft_config["default"])
            if hasattr(lora_model, "train_adapter"):
                try:
                    lora_model.train_adapter(name)
                except Exception:
                    pass

        mole_model = self.module.MoLEMixture(
            lora_model,
            adapter_names=tuple(adapter_names),
            item_token_ids=self.item_token_ids,
            gating_hidden=self.args.mole_gating_hidden,
            temperature=meta.get("temperature", self.args.mole_temp),
            balance_coef=meta.get("balance_coef", self.args.mole_balance),
        ).to(self.device)

        gate_path = os.path.join(model_dir, "mole_gate.pt")
        if os.path.isfile(gate_path):
            state = torch.load(gate_path, map_location=self.device)
            if "gate_state" in state:
                mole_model.gate.load_state_dict(state["gate_state"])
        return mole_model, adapter_names

    def _save_mole_checkpoint(self, mole_model: torch.nn.Module, stage_dir: str, adapter_names: List[str]):
        self._save_lora_checkpoint(
            mole_model.model,
            stage_dir,
            adapter_names,
            meta={"temperature": self.args.mole_temp, "balance_coef": self.args.mole_balance},
        )
        torch.save(
            {
                "gate_state": mole_model.gate.state_dict(),
                "adapter_names": adapter_names,
                "temperature": self.args.mole_temp,
                "balance_coef": self.args.mole_balance,
            },
            os.path.join(stage_dir, "mole_gate.pt"),
        )

    def _load_raie_model(self, model_dir: str):
        base_model = BertForMaskedLM.from_pretrained(self._base_path(model_dir)).to(self.device)
        adapter_dir = self._adapter_path(model_dir)
        meta = self._load_meta(model_dir)
        if os.path.isdir(adapter_dir):
            try:
                lora_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
            except Exception:
                lora_model = self._apply_lora(base_model)
        else:
            lora_model = self._apply_lora(base_model)

        adapter_names = meta.get("adapter_names")
        if adapter_names is None:
            adapter_names = list(getattr(lora_model, "peft_config", {}).keys()) or ["default"]
        for name in adapter_names:
            if name not in getattr(lora_model, "peft_config", {}):
                lora_model.add_adapter(name, lora_model.peft_config["default"])
        return lora_model, adapter_names, meta

    def _save_raie_regions(self, bank, stage_dir: str):
        state_path = os.path.join(stage_dir, "raie_regions_state.npz")
        np.savez(
            state_path,
            centroids=bank.C,
            radii=bank.R,
            sig=bank.sig,
            kappa=bank.kappa,
            pi=bank.pi,
            S=bank.S,
            n=bank.n,
            K=bank.K,
            orig_idx_by_k=np.array(bank.orig_idx_by_k, dtype=object),
        )

    def _restore_raie_regions(self, bank, model_dir: str):
        state_path = os.path.join(model_dir, "raie_regions_state.npz")
        fallback = os.path.join(model_dir, "raie_regions_after_map.npz")
        if os.path.isfile(state_path):
            data = np.load(state_path, allow_pickle=True)
        elif os.path.isfile(fallback):
            data = np.load(fallback, allow_pickle=True)
        else:
            return False

        bank.C = data["centroids"]
        bank.R = data["radii"]
        bank.sig = data["sig"]
        bank.kappa = data["kappa"]
        bank.pi = data["pi"]
        bank.S = data["S"]
        bank.n = data["n"]
        bank.K = int(data["K"]) if "K" in data else bank.C.shape[0]
        if "orig_idx_by_k" in data:
            try:
                bank.orig_idx_by_k = dict(data["orig_idx_by_k"].item())
            except Exception:
                pass
        return True

    def _rebuild_orig_assignments(self, bank):
        X = self.module.encode_prompts_to_vecs(
            bank.model,
            self.original_route_ds.examples,
            self.token2id,
            self.args.max_len,
            self.pad_id,
            self.cls_id,
            self.device,
            pbar=False,
        )
        sims = X @ bank.C.T
        scores = np.log(np.clip(bank.pi, 1e-8, None))[None, :] + sims * bank.kappa[None, :]
        route_k = np.argmax(scores, axis=1)
        bank.orig_idx_by_k = {}
        for i, k in enumerate(route_k):
            bank.orig_idx_by_k.setdefault(int(k), []).append(i)

    # -------------------------
    # Summary
    # -------------------------
    def _save_summary(self, results: Iterable[StageResult]):
        summary = []
        for r in results:
            summary.append(
                {
                    "finetune_stage": r.finetune_stage,
                    "predict_stage": r.predict_stage,
                    "predict_metrics": r.predict_metrics,
                    "original_metrics": r.original_metrics,
                    "model_dir": r.model_dir,
                }
            )
        with open(os.path.join(self.args.output_dir, "multistage_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage runner for Bert4Rec_RAIE")
    p.add_argument("--data_dir", type=str, default="/home/zj/code/ml-10M100K_multistage")
    p.add_argument("--output_dir", type=str, default="/home/zj/code/ml-10M100K_multistage")
    p.add_argument("--mode", type=str, default="mole", choices=[
        "base",
        "lora",
        "lora_replay",
        "lora_lwf",
        "lsat",
        "ebpr",
        "raie",
        "mole",
    ])
    p.add_argument("--stages", type=str, default="f1,f2,f3,f4")
    p.add_argument("--pred_stages", type=str, default="f2,f3,f4,test")

    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--objective", type=str, default="cloze", choices=["cloze", "next"])
    p.add_argument("--mask_prob", type=float, default=0.15)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--finetune_epochs", type=int, default=3)
    p.add_argument("--finetune_batch_size", type=int, default=32)
    p.add_argument("--finetune_lr", type=float, default=5e-4)
    p.add_argument("--do_prompt_mask", action="store_true")
    p.add_argument("--prompt_mask_prob", type=float, default=0.15)

    p.add_argument("--replay_ratio", type=float, default=0.3)
    p.add_argument("--lwf_T", type=float, default=2.0)
    p.add_argument("--lwf_alpha", type=float, default=0.5)
    p.add_argument("--lsat_alpha", type=float, default=0.6)
    p.add_argument("--lsat_long_epochs", type=int, default=1)
    p.add_argument("--lsat_short_epochs", type=int, default=3)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target",
        type=str,
        default="query, key, value, intermediate.dense, output.dense, attention.output.dense",
    )

    p.add_argument("--mole_num_experts", type=int, default=3)
    p.add_argument("--mole_gating_hidden", type=int, default=128)
    p.add_argument("--mole_temp", type=float, default=0.7)
    p.add_argument("--mole_balance", type=float, default=0.01)

    p.add_argument("--K", type=int, default=3)
    p.add_argument("--q", type=float, default=0.9)
    p.add_argument("--T_low", type=float, default=0.7)
    p.add_argument("--T_high", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--gap_thr", type=float, default=0.02)

    p.add_argument("--ebpr_steps", type=int, default=20)
    p.add_argument("--ebpr_lr", type=float, default=3e-4)
    p.add_argument("--ebpr_neg_per_step", type=int, default=20)
    p.add_argument("--ebpr_batch_size", type=int, default=256)
    p.add_argument("--ebpr_topk", type=int, default=10)
    p.add_argument("--ebpr_per_user", type=int, default=2)
    p.add_argument("--ebpr_max_pairs", type=int, default=10000)

    p.add_argument("--topk", type=str, default="5,10,20")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    runner = MultiStageRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
