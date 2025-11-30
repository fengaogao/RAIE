import argparse
import importlib.util
import json
import os
import shutil
import types
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


# -------------------------------------------------
# Utilities for loading the original openP5_RAIE
# -------------------------------------------------


def _load_openp5_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("openP5_RAIE", path)
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
    """Multi-round incremental learning pipeline for OpenP5."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.module = _load_openp5_module(os.path.join(os.path.dirname(__file__), "openP5_RAIE"))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # tokenizer + item vocab
        self.mid2idx, self.n_items = self.module.build_or_load_mid2idx(args, is_main=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.final_item_tokens = self.module.make_item_final_tokens(self.n_items)
        self.item_final_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.final_item_tokens]
        self.collate = lambda batch: self.module.collate_batch(batch, self.tokenizer.pad_token_id)

        # loaders for original evaluation
        self.ds_original = self._make_ds(args.original_jsonl_path or os.path.join(args.data_dir, "original.jsonl"))
        self.ds_train_stream = self._make_ds(args.train_jsonl_path or os.path.join(args.data_dir, "original_stride1.jsonl"))
        self.original_loader = DataLoader(
            self.ds_original,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate,
            pin_memory=True,
        )

        os.makedirs(args.output_dir, exist_ok=True)

        self.stream_loader = DataLoader(
            self.ds_train_stream,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate,
            pin_memory=True,
        )

    # -------------------------------------------------
    # Dataset helpers
    # -------------------------------------------------
    def _make_ds(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(jsonl_path)
        item_token_seq = [[t] for t in self.final_item_tokens]
        return self.module.SeqRecDatasetFromPairs(
            jsonl_path,
            self.mid2idx,
            self.tokenizer,
            item_token_seq,
            max_history=self.args.max_history,
            wrap_mode="none",
            final_token_only_loss=self.args.final_token_only_loss,
            deterministic_template=True,
        )

    def _make_loader(self, jsonl_path: str, batch_size: int, shuffle: bool) -> DataLoader:
        ds = self._make_ds(jsonl_path)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.collate,
            pin_memory=True,
        )

    # -------------------------------------------------
    # Model helpers
    # -------------------------------------------------
    def _fresh_lora_model(self):
        """Load (or build) a LoRA-wrapped base model with the expanded vocab.

        Bert4Rec 的多阶段 driver 在内部训练并持久化基础模型；这里保持一致，
        当 `base_with_new_tokens` 不存在时自动从原始基座构建并保存，避免因为缺少
        初始检查点而无法进入后续阶段。
        """

        base_dir = self.args.resume_base_dir or os.path.join(self.args.output_dir, "base_with_new_tokens")
        if os.path.isdir(base_dir):
            tok, model = self.module.fresh_lora_model_from_base(
                base_dir,
                self.device,
                self.args.lora_r,
                self.args.lora_alpha,
                self.args.lora_dropout,
                torch.float16 if self.args.fp16 else torch.float32,
            )
        else:
            # 从原始基座构建，并补全 item tokens 后保存，保证后续阶段可复用
            tok = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            # 追加物品末端 token
            tok.add_tokens([t for t in self.final_item_tokens if t not in tok.get_vocab()])

            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            model.resize_token_embeddings(len(tok))
            model.save_pretrained(base_dir)
            tok.save_pretrained(base_dir)

            # 再包装 LoRA
            lcfg = self.module.LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                bias="none",
                task_type=self.module.TaskType.CAUSAL_LM,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            model = self.module.get_peft_model(model, lcfg)
            model.to(self.device)

        # keep tokenizer aligned with base
        self.tokenizer = tok
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return model

    def _load_base_model(self, model_dir: str):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.to(self.device)
        return model

    def _load_lora_from_dir(self, model_dir: str, adapter_names: List[str] = None):
        model = self._fresh_lora_model()
        adapter_dir = model_dir
        meta = self._load_meta(model_dir)
        names = adapter_names or meta.get("adapter_names") or ["default"]
        for name in names:
            if os.path.isdir(adapter_dir):
                try:
                    model.load_adapter(adapter_dir, adapter_name=name, is_trainable=True)
                except Exception:
                    pass
        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter(names[0])
            except Exception:
                pass
        return model, names, meta

    def _load_mole_model(self, model_dir: str):
        lora_model, adapter_names, meta = self._load_lora_from_dir(model_dir)
        if not adapter_names:
            adapter_names = ["default"] + [f"expert_{i}" for i in range(self.args.mole_num_experts - 1)]

        base_cfg = getattr(lora_model, "peft_config", {})
        for name in adapter_names:
            if name not in base_cfg and "default" in base_cfg:
                try:
                    lora_model.add_adapter(name, base_cfg["default"])
                except Exception:
                    pass

        mole_model = self.module.MoLEMixtureCausal(
            lora_model,
            adapter_names=adapter_names,
            item_final_token_ids=self.item_final_token_ids,
            gating_hidden=self.args.mole_gating_hidden,
            temperature=meta.get("temperature", self.args.mole_temp),
            balance_coef=meta.get("balance_coef", self.args.mole_balance),
        ).to(self.device)
        mole_model.gate.float()

        gate_path = os.path.join(model_dir, "mole_gate.pt")
        if os.path.isfile(gate_path):
            state = torch.load(gate_path, map_location=self.device)
            if "gate_state" in state:
                mole_model.gate.load_state_dict(state["gate_state"])
        return mole_model, adapter_names

    def _save_mole_checkpoint(self, mole_model, stage_dir: str, adapter_names: List[str]):
        self._save_adapter(
            mole_model.model,
            stage_dir,
            adapter_names,
            meta={
                "adapter_names": adapter_names,
                "temperature": self.args.mole_temp,
                "balance_coef": self.args.mole_balance,
            },
        )
        torch.save(
            {
                "gate_state": mole_model.gate.state_dict(),
                "temperature": self.args.mole_temp,
                "balance_coef": self.args.mole_balance,
            },
            os.path.join(stage_dir, "mole_gate.pt"),
        )

    def _save_adapter(self, model, stage_dir: str, adapter_names: Iterable[str], meta: Dict = None):
        os.makedirs(stage_dir, exist_ok=True)
        base_eval_model = model.module if hasattr(model, "module") else model
        base_eval_model.save_pretrained(stage_dir, selected_adapters=list(adapter_names))
        with open(os.path.join(stage_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta or {"adapter_names": list(adapter_names)}, f, ensure_ascii=False, indent=2)

    def _load_meta(self, model_dir: str) -> Dict:
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # -------------------------------------------------
    # Training helpers
    # -------------------------------------------------
    def _train_default_adapter(self) -> Tuple[torch.nn.Module, str]:
        """Train a default adapter on the original stream when none exists.

        This mirrors the Bert4Rec multi-stage driver behaviour where the
        baseline model is established inside the runner, ensuring the
        subsequent multi-stage finetune/predict cycles have a valid starting
        checkpoint.
        """

        print("[Train] default adapter missing; training on original stream ...")
        lora_model = self._fresh_lora_model()
        trainable = [p for p in lora_model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=self.args.lr)
        total_steps = len(self.stream_loader) * max(self.args.epochs, 1)
        sched = get_linear_schedule_with_warmup(
            optim, int(total_steps * self.args.warmup_ratio), total_steps
        )

        for ep in range(1, self.args.epochs + 1):
            loss = self.module.train_one_epoch(
                lora_model,
                self.stream_loader,
                optim,
                sched,
                self.device,
                item_final_token_ids=self.item_final_token_ids,
                fp16=self.args.fp16,
                is_main=True,
                grad_clip=self.args.grad_clip,
            )
            print(f"[Train][default][Epoch {ep}] loss={loss:.4f}")

        adapter_names = getattr(lora_model, "peft_config", {}).keys() or ["default"]
        default_dir = self.args.default_adapter_dir or os.path.join(self.args.output_dir, "default")
        self._save_adapter(lora_model, default_dir, adapter_names, meta={"adapter_names": list(adapter_names)})
        return lora_model, default_dir

    def _finetune_lora(self, lora_model, ft_loader: DataLoader, replay_buf=None, teacher=None):
        ft_params = [p for p in lora_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.lr)
        total_steps = len(ft_loader) * max(1, self.args.finetune_epochs)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(total_steps * self.args.warmup_ratio), total_steps
        )
        plugin = "none"
        if self.args.mode == "lora_replay":
            plugin = "replay"
        elif self.args.mode == "lora_lwf":
            plugin = "lwf"
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_lora(
                lora_model,
                ft_loader,
                self.device,
                self.item_final_token_ids,
                optimizer=ft_optim,
                scheduler=ft_sched,
                fp16=self.args.fp16,
                plugin=plugin,
                replay_buf=replay_buf,
                replay_ratio=self.args.replay_ratio,
                grad_clip=self.args.grad_clip,
                log_every=100,
                teacher=teacher,
                lwf_T=self.args.lwf_T,
                lwf_alpha=self.args.lwf_alpha,
            )
            print(f"[Finetune][{plugin}] Epoch {ep} loss={loss:.4f}")
        return lora_model

    def _finetune_base(self, base_model, ft_loader: DataLoader):
        ft_params = [p for p in base_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.lr)
        total_steps = len(ft_loader) * max(1, self.args.finetune_epochs)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(total_steps * self.args.warmup_ratio), total_steps
        )
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_lora(
                base_model,
                ft_loader,
                self.device,
                self.item_final_token_ids,
                optimizer=ft_optim,
                scheduler=ft_sched,
                fp16=self.args.fp16,
                plugin="none",
                replay_buf=None,
                replay_ratio=0.0,
                grad_clip=self.args.grad_clip,
                log_every=100,
            )
            print(f"[Finetune][base] Epoch {ep} loss={loss:.4f}")
        return base_model

    def _finetune_lsat(self, model, train_loader: DataLoader, ft_loader: DataLoader):
        long_adapter = "default"
        short_adapter = "short_term"
        if short_adapter not in getattr(model, "peft_config", {}):
            model.add_adapter(short_adapter, model.peft_config[long_adapter])
        if hasattr(model, "set_adapter") and hasattr(model, "train_adapter"):
            model.set_adapter(long_adapter)
            model.train_adapter(long_adapter)
        long_optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.args.lr)
        long_sched = get_linear_schedule_with_warmup(
            long_optim,
            int(self.args.warmup_ratio * len(train_loader) * max(1, self.args.lsat_long_epochs)),
            len(train_loader) * max(1, self.args.lsat_long_epochs),
        )
        for ep in range(1, self.args.lsat_long_epochs + 1):
            loss = self.module.train_one_epoch(
                model,
                train_loader,
                long_optim,
                long_sched,
                self.device,
                item_final_token_ids=self.item_final_token_ids,
                fp16=self.args.fp16,
                is_main=True,
                grad_clip=self.args.grad_clip,
            )
            print(f"[LSAT][Long] Epoch {ep} loss={loss:.4f}")

        if hasattr(model, "set_adapter") and hasattr(model, "train_adapter"):
            model.set_adapter(short_adapter)
            model.train_adapter(short_adapter)
        short_optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.args.lr)
        short_sched = get_linear_schedule_with_warmup(
            short_optim,
            int(self.args.warmup_ratio * len(ft_loader) * max(1, self.args.lsat_short_epochs)),
            len(ft_loader) * max(1, self.args.lsat_short_epochs),
        )
        for ep in range(1, self.args.lsat_short_epochs + 1):
            loss = self.module.finetune_one_epoch_lora(
                model,
                ft_loader,
                self.device,
                self.item_final_token_ids,
                optimizer=short_optim,
                scheduler=short_sched,
                fp16=self.args.fp16,
                plugin="none",
                replay_buf=None,
                replay_ratio=0.0,
                grad_clip=1.0,
                log_every=100,
            )
            print(f"[LSAT][Short] Epoch {ep} loss={loss:.4f}")
        return model

    def _finetune_mole(self, mole_model, ft_loader: DataLoader):
        ft_params = [p for p in mole_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.lr)
        total_steps = len(ft_loader) * max(1, self.args.finetune_epochs)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(total_steps * self.args.warmup_ratio), total_steps
        )
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_mole(
                mole_model,
                ft_loader,
                ft_optim,
                ft_sched,
                self.device,
                grad_clip=self.args.grad_clip,
                fp16=self.args.fp16,
            )
            print(f"[MoLE][Finetune] Epoch {ep} loss={loss:.4f}")
        return mole_model

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
            orig_idx_by_k=np.array(list((bank.orig_idx_by_k or {}).items()), dtype=object),
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

        bank.C = data.get("centroids")
        bank.R = data.get("radii")
        bank.sig = data.get("sig")
        bank.kappa = data.get("kappa")
        bank.pi = data.get("pi")
        bank.S = data.get("S")
        bank.n = data.get("n")
        bank.K = int(data.get("K")) if data.get("K") is not None else bank.K
        orig_pairs = data.get("orig_idx_by_k")
        bank.orig_idx_by_k = {}
        if orig_pairs is not None:
            for k, v in orig_pairs:
                bank.orig_idx_by_k[int(k)] = list(v)
        return True

    # -------------------------------------------------
    # Stage execution
    # -------------------------------------------------
    def run(self) -> List[StageResult]:
        ft_stages = self.args.stages.split(",")
        pred_stages = self.args.pred_stages.split(",")
        if len(ft_stages) != len(pred_stages):
            raise ValueError("--stages and --pred_stages must align")
        if not ft_stages:
            raise ValueError("--stages must not be empty")

        topk_tuple = tuple(sorted(set(self.args.topk)))

        if self.args.mode == "base":
            current_model_dir = self.args.resume_base_dir or os.path.join(
                self.args.output_dir, "base_with_new_tokens"
            )
            if not os.path.isdir(current_model_dir):
                raise FileNotFoundError(current_model_dir)
            base_model = self._load_base_model(current_model_dir)
            initial_model = base_model
        elif self.args.mode == "mole":
            current_model_dir = self.args.default_adapter_dir or os.path.join(
                self.args.output_dir, "default"
            )
            if not os.path.isdir(current_model_dir):
                initial_model, current_model_dir = self._train_default_adapter()
            else:
                initial_model, _ = self._load_mole_model(current_model_dir)
        else:
            base_model = self._fresh_lora_model()
            default_dir = self.args.default_adapter_dir or os.path.join(self.args.output_dir, "default")
            if not os.path.isdir(default_dir):
                base_model, default_dir = self._train_default_adapter()
            base_model.load_adapter(default_dir, adapter_name="default", is_trainable=True)
            if hasattr(base_model, "set_adapter"):
                try:
                    base_model.set_adapter("default")
                except Exception:
                    pass
            initial_model = base_model
            current_model_dir = default_dir

        initial_pred_path = os.path.join(self.args.data_dir, f"{ft_stages[0]}.jsonl")
        initial_loader = self._make_loader(initial_pred_path, self.args.batch_size, shuffle=False)
        initial_predict_metrics = self.module.evaluate_causal(
            initial_model,
            initial_loader,
            self.device,
            self.item_final_token_ids,
            k_list=topk_tuple,
            fp16=self.args.fp16,
            desc=f"Eval-{ft_stages[0]}",
        )
        initial_original_metrics = self.module.evaluate_causal(
            initial_model,
            self.original_loader,
            self.device,
            self.item_final_token_ids,
            k_list=topk_tuple,
            fp16=self.args.fp16,
            desc="Eval-original",
        )
        results: List[StageResult] = [
            StageResult(
                finetune_stage="original_stream",
                predict_stage=ft_stages[0],
                predict_metrics=initial_predict_metrics,
                original_metrics=initial_original_metrics,
                model_dir=current_model_dir,
            )
        ]
        print(f"[Eval][original_stream->{ft_stages[0]}] {initial_predict_metrics}")
        print(f"[Eval][original_stream->original] {initial_original_metrics}")

        for idx, (ft_name, pred_name) in enumerate(zip(ft_stages, pred_stages), start=1):
            print(f"===== Stage {idx}: finetune {ft_name} -> predict {pred_name} =====")
            ft_path = os.path.join(self.args.data_dir, f"{ft_name}.jsonl")
            pred_path = os.path.join(self.args.data_dir, f"{pred_name}.jsonl")
            ft_loader = self._make_loader(ft_path, self.args.finetune_batch_size, shuffle=True)
            pred_loader = self._make_loader(pred_path, self.args.batch_size, shuffle=False)

            stage_dir = os.path.join(self.args.output_dir, f"stage_{ft_name}")
            if os.path.isdir(stage_dir):
                shutil.rmtree(stage_dir)
            os.makedirs(stage_dir, exist_ok=True)

            predict_metrics: Dict[str, float]
            original_metrics: Dict[str, float]

            if self.args.mode == "base":
                base_model = self._load_base_model(current_model_dir)
                base_model = self._finetune_base(base_model, ft_loader)
                predict_metrics = self.module.evaluate_causal(
                    base_model,
                    pred_loader,
                    self.device,
                    self.item_final_token_ids,
                    k_list=topk_tuple,
                    fp16=self.args.fp16,
                    desc=f"Eval-{pred_name}",
                )
                original_metrics = self.module.evaluate_causal(
                    base_model,
                    self.original_loader,
                    self.device,
                    self.item_final_token_ids,
                    k_list=topk_tuple,
                    fp16=self.args.fp16,
                    desc="Eval-original",
                )
                base_model.save_pretrained(stage_dir)
                current_model_dir = stage_dir

            elif self.args.mode in {"lora", "lora_replay", "lora_lwf"}:
                replay_buf = None
                teacher = None
                if self.args.mode == "lora_replay":
                    replay_buf = self.module.ReplayBuffer(
                        self.ds_train_stream, self.collate, batch_size=self.args.finetune_batch_size
                    )
                if self.args.mode == "lora_lwf":
                    teacher = AutoModelForCausalLM.from_pretrained(
                        self.args.resume_base_dir or os.path.join(self.args.output_dir, "base_with_new_tokens"),
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
                    )
                    if hasattr(teacher.config, "use_cache"):
                        teacher.config.use_cache = False
                    teacher.to(self.device)
                    teacher.eval()
                    for p in teacher.parameters():
                        p.requires_grad = False

                lora_model, adapter_names, _ = self._load_lora_from_dir(current_model_dir)
                lora_model = self._finetune_lora(lora_model, ft_loader, replay_buf=replay_buf, teacher=teacher)
                predict_metrics = self.module.evaluate_causal(
                    lora_model,
                    pred_loader,
                    self.device,
                    self.item_final_token_ids,
                    k_list=topk_tuple,
                    fp16=self.args.fp16,
                    desc=f"Eval-{pred_name}",
                )
                original_metrics = self.module.evaluate_causal(
                    lora_model,
                    self.original_loader,
                    self.device,
                    self.item_final_token_ids,
                    k_list=topk_tuple,
                    fp16=self.args.fp16,
                    desc="Eval-original",
                )
                self._save_adapter(lora_model, stage_dir, adapter_names, meta={"adapter_names": adapter_names})
                current_model_dir = stage_dir

            elif self.args.mode == "lsat":
                lora_model, adapter_names, _ = self._load_lora_from_dir(current_model_dir)
                train_loader = DataLoader(
                    self.ds_train_stream,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    collate_fn=self.collate,
                    pin_memory=True,
                )
                lora_model = self._finetune_lsat(lora_model, train_loader, ft_loader)
                predict_metrics = self.module.evaluate_lsat_causal(
                    lora_model,
                    pred_loader,
                    self.device,
                    self.item_final_token_ids,
                    long_adapter="default",
                    short_adapter="short_term",
                    alpha=self.args.lsat_alpha,
                    k_list=topk_tuple,
                )
                original_metrics = self.module.evaluate_lsat_causal(
                    lora_model,
                    self.original_loader,
                    self.device,
                    self.item_final_token_ids,
                    long_adapter="default",
                    short_adapter="short_term",
                    alpha=self.args.lsat_alpha,
                    k_list=topk_tuple,
                )
                self._save_adapter(lora_model, stage_dir, ["default", "short_term"], meta={"adapter_names": ["default", "short_term"]})
                current_model_dir = stage_dir

            elif self.args.mode == "mole":
                mole_model, adapter_names = self._load_mole_model(current_model_dir)
                mole_model = self._finetune_mole(mole_model, ft_loader)
                predict_metrics = self.module.evaluate_causal(
                    mole_model,
                    pred_loader,
                    self.device,
                    self.item_final_token_ids,
                    k_list=topk_tuple,
                    fp16=self.args.fp16,
                    desc=f"Eval-{pred_name}",
                )
                original_metrics = self.module.evaluate_causal(
                    mole_model,
                    self.original_loader,
                    self.device,
                    self.item_final_token_ids,
                    k_list=topk_tuple,
                    fp16=self.args.fp16,
                    desc="Eval-original",
                )
                self._save_mole_checkpoint(mole_model, stage_dir, adapter_names)
                current_model_dir = stage_dir

            elif self.args.mode == "raie":
                lora_model, adapter_names, meta = self._load_lora_from_dir(current_model_dir)
                if "default" not in adapter_names:
                    adapter_names = ["default"] + list(adapter_names)
                base_eval_model = lora_model
                bank = self.module.RegionBank(
                    model=base_eval_model,
                    tok=self.tokenizer,
                    item_final_token_ids=self.item_final_token_ids,
                    original_ds=self.ds_original,
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
                )

                restored = self._restore_raie_regions(bank, current_model_dir)

                if not restored:
                    bank.fit_regions_on_original()
                region_buckets, soft_pairs = bank.map_finetune(self._make_ds(ft_path))
                bank.train_regions(
                    finetune_ds=self._make_ds(ft_path),
                    collate_fn=self.collate,
                    optimizer_ctor=lambda m: torch.optim.AdamW(
                        [p for p in m.parameters() if p.requires_grad], lr=self.args.lr
                    ),
                    scheduler_ctor=lambda opt, steps_per_epoch, epochs: get_linear_schedule_with_warmup(
                        opt,
                        int(self.args.warmup_ratio * steps_per_epoch * epochs),
                        steps_per_epoch * epochs,
                    ),
                    epochs_per_region=self.args.finetune_epochs,
                    batch_size=self.args.finetune_batch_size,
                    orig_mix_ratio=1.0,
                    region_buckets=region_buckets,
                    soft_pairs=soft_pairs,
                )

                stats_pred = bank.route_and_eval(
                    self._make_ds(pred_path), self.collate, k_list=topk_tuple, root=stage_dir, fp16=self.args.fp16
                )
                stats_orig = bank.route_and_eval(
                    self.ds_original, self.collate, k_list=topk_tuple, root=stage_dir, fp16=self.args.fp16
                )

                def _to_global(stats):
                    import pandas as pd

                    df = pd.DataFrame(stats)
                    w = df["num_samples"]
                    W = max(1, w.sum())
                    return {m: float((df[m] * w).sum() / W) for m in df.columns if m.startswith(("Recall@", "NDCG@"))}

                predict_metrics = _to_global(stats_pred)
                original_metrics = _to_global(stats_orig)
                self._save_raie_regions(bank, stage_dir)
                self._save_adapter(lora_model, stage_dir, adapter_names, meta=meta or {"adapter_names": adapter_names})
                current_model_dir = stage_dir

            else:  # pragma: no cover - defensive
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

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
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


# -------------------------------------------------
# CLI
# -------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage runner for OpenP5_RAIE")
    p.add_argument("--data_dir", type=str, default="/home/zj/code/yelp/")
    p.add_argument("--output_dir", type=str, default="./runs/openP5_yelp")
    p.add_argument(
        "--mode",
        type=str,
        default="lora",
        choices=["base", "lora", "lora_replay", "lora_lwf", "lsat", "raie", "mole"],
    )
    p.add_argument("--stages", type=str, required=True)
    p.add_argument("--pred_stages", type=str, required=True)

    p.add_argument("--model_name_or_path", type=str, default="/home/zj/model/Llama-2-7b-hf")
    p.add_argument("--resume_base_dir", type=str, default="")
    p.add_argument("--default_adapter_dir", type=str, default="")

    p.add_argument("--max_history", type=int, default=20)
    p.add_argument("--final_token_only_loss", action="store_true", default=True)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--finetune_batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--finetune_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--replay_ratio", type=float, default=0.3)
    p.add_argument("--lwf_T", type=float, default=2.0)
    p.add_argument("--lwf_alpha", type=float, default=0.5)
    p.add_argument("--lsat_alpha", type=float, default=0.6)
    p.add_argument("--lsat_long_epochs", type=int, default=1)
    p.add_argument("--lsat_short_epochs", type=int, default=3)
    p.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--fp16", type=bool, default=True)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

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

    p.add_argument("--train_jsonl_path", type=str, default="")
    p.add_argument("--original_jsonl_path", type=str, default="")
    p.add_argument("--test_jsonl_path", type=str, default="")
    p.add_argument("--item_ids_path", type=str, default="")
    p.add_argument("--finetune_jsonl_path", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    runner = MultiStageRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
