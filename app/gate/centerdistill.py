"""CenterDistill ambiguity gate — learned routing via EAAAI 2026 model.

Loads an XLM-RoBERTa-large checkpoint with a center head (Linear → 5).
Falls back to HeuristicGate if no checkpoint is available.

Checkpoint resolution order (spec §5.1):
    1. Local dir at settings.gate_checkpoint_path
    2. HF Hub via settings.gate_hf_repo → snapshot_download
    3. Absent → HeuristicGate, fallback_used=True

The app MUST NEVER crash on a missing checkpoint (AGENTS.md rule 3).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from app.gate.base import Behaviour, GateDecision
from app.gate.heuristic import HeuristicGate
from app.gate.thresholds import (
    GateThresholds,
    get_thresholds_from_settings,
)
from app.settings import Settings

logger = logging.getLogger(__name__)

# Number of semantic centers from spectral clustering (paper §2)
_K: int = 5

# Tokenizer settings (spec §5.2)
_MAX_LENGTH: int = 384
_TRUNCATION: str = "longest_first"

# Teacher temperature for center similarity (paper §2: τ = 10.0)
_TEACHER_TAU: float = 10.0


def _try_import_torch() -> tuple[Any, bool]:
    """Import torch if available."""
    try:
        import torch
        return torch, True
    except ImportError:
        return None, False


def _try_load_checkpoint(
    checkpoint_path: Path | None,
    hf_repo: str | None,
) -> tuple[Any, Any, npt.NDArray[np.float64] | None, bool]:
    """Try to load model, tokenizer, and centers from checkpoint.

    Returns (model, tokenizer, centers, success). On any failure,
    returns (None, None, None, False) — never raises.
    """
    torch, has_torch = _try_import_torch()
    if not has_torch:
        logger.warning("torch not installed — using heuristic fallback")
        return None, None, None, False

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed — using heuristic fallback")
        return None, None, None, False

    # Resolve checkpoint path
    resolved_path: Path | None = None

    if checkpoint_path and checkpoint_path.exists():
        resolved_path = checkpoint_path
        logger.info("Loading checkpoint from local path: %s", checkpoint_path)
    elif hf_repo:
        try:
            from huggingface_hub import snapshot_download
            local_dir: str = snapshot_download(hf_repo)
            resolved_path = Path(local_dir)
            logger.info("Downloaded checkpoint from HF Hub: %s", hf_repo)
        except Exception as exc:
            logger.warning("Failed to download from HF Hub: %s", exc)
            return None, None, None, False

    if resolved_path is None:
        logger.info("No checkpoint configured — using heuristic fallback")
        return None, None, None, False

    try:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        full_pt = resolved_path / "centerdistill_full.pt"

        if full_pt.exists():
            logger.info("Loading repaired artifact: %s", full_pt)
            ckpt_data = torch.load(full_pt, map_location=device)
            base_model_name = ckpt_data.get("base_model", "deepset/xlm-roberta-large-squad2")
            is_local = base_model_name.startswith("/") or "\\" in base_model_name
            if not Path(base_model_name).exists() and is_local:
                base_model_name = "deepset/xlm-roberta-large-squad2"
            k_val = ckpt_data.get("K", 5)

            # Rebuild model structure matching repair_checkpoint.py
            import torch.nn as nn

            class RepairedCenterDistillModel(nn.Module):
                def __init__(self, base_name: str, num_centers: int) -> None:
                    super().__init__()
                    try:
                        from transformers import AutoConfig, AutoModel
                        config = AutoConfig.from_pretrained(base_name)
                        self.encoder = AutoModel.from_config(config)  # type: ignore[no-untyped-call]
                    except Exception:
                        self.encoder = AutoModel.from_pretrained(base_name)

                    hidden = self.encoder.config.hidden_size
                    self.span_head = nn.Linear(hidden, 2)
                    self.center_head = nn.Linear(hidden, num_centers)

                def forward(
                    self,
                    input_ids: Any = None,
                    attention_mask: Any = None,
                    **kwargs: Any,
                ) -> Any:
                    return self.encoder(
                        input_ids=input_ids, attention_mask=attention_mask, **kwargs
                    )

            model = RepairedCenterDistillModel(base_model_name, k_val)
            model.load_state_dict(ckpt_data["state_dict"], strict=False)
            model.eval().to(device)

            tok_path = (
                resolved_path
                if (resolved_path / "tokenizer_config.json").exists()
                else (
                    resolved_path.parent
                    if (resolved_path.parent / "tokenizer_config.json").exists()
                    else base_model_name
                )
            )
            tokenizer = AutoTokenizer.from_pretrained(str(tok_path))  # type: ignore[no-untyped-call]
            dummy_centers = np.zeros((k_val, 768), dtype=np.float64)

            logger.info("Repaired CenterDistill artifact loaded on %s", device)
            return model, tokenizer, dummy_centers, True

        # Standard HuggingFace / centers.npy format
        centers_path = resolved_path / "centers.npy"
        if not centers_path.exists():
            logger.warning(
                "Neither centerdistill_full.pt nor centers.npy found in %s — using fallback",
                resolved_path,
            )
            return None, None, None, False
        centers: npt.NDArray[np.float64] = np.load(str(centers_path))

        tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))  # type: ignore[no-untyped-call]
        model = AutoModel.from_pretrained(str(resolved_path))
        model.eval()

        if device.type == "cuda":
            model = model.half()
        model = model.to(device)

        logger.info("CenterDistill checkpoint loaded on %s", device)
        return model, tokenizer, centers, True

    except Exception as exc:
        logger.warning("Failed to load checkpoint: %s — using heuristic fallback", exc)
        return None, None, None, False


class CenterDistillGate:
    """Learned ambiguity gate from CenterDistill (EAAAI 2026).

    If no checkpoint is available, delegates to HeuristicGate transparently.
    The fallback_used field in GateDecision distinguishes the two cases.

    The app must never crash on a missing checkpoint — a reviewer without
    weights still gets a working demo (AGENTS.md rule 3).
    """

    def __init__(
        self,
        checkpoint_path: Path | Settings | None = None,
        hf_repo: str | None = None,
        thresholds: GateThresholds | None = None,
    ) -> None:
        if isinstance(checkpoint_path, Settings):
            settings = checkpoint_path
            checkpoint_path = settings.gate_checkpoint_path
            hf_repo = hf_repo or settings.gate_hf_repo
            thresholds = thresholds or get_thresholds_from_settings(settings)

        self._thresholds = thresholds or get_thresholds_from_settings()
        self._model: Any = None
        self._tokenizer: Any = None
        self._centers: npt.NDArray[np.float64] | None = None
        self._device: Any = None
        self._fallback: bool = True
        # Typed to avoid Any return in __call__ when delegating
        self._heuristic_gate: HeuristicGate | None = None
        self._heuristic: Any = None

        model, tokenizer, centers, success = _try_load_checkpoint(
            checkpoint_path, hf_repo
        )

        if success:
            self._model = model
            self._tokenizer = tokenizer
            self._centers = centers
            self._fallback = False
            self._heuristic_gate = None
            self._device = next(model.parameters()).device
        else:
            self._heuristic_gate = HeuristicGate(thresholds=self._thresholds)

    @property
    def using_fallback(self) -> bool:
        """Whether this gate is using the heuristic fallback."""
        return self._fallback

    def __call__(self, question: str, context: str) -> GateDecision:
        """Classify a question-context pair.

        Delegates to heuristic if no checkpoint loaded.
        """
        if self._fallback and self._heuristic_gate is not None:
            return self._heuristic_gate(question, context)
        return self._classify_learned(question, context)

    decide = __call__

    def encode_cls(self, question: str, context: str | None = None) -> npt.NDArray[np.float64]:
        """Extract the CLS token hidden embedding from XLM-RoBERTa (1024-dim)."""
        return self.encode_cls_batch([question], [context])[0]

    def encode_cls_batch(
        self, questions: list[str], contexts: list[str | None]
    ) -> npt.NDArray[np.float64]:
        """Extract CLS embeddings for a batch of question-context pairs (1024-dim)."""
        if self._fallback:
            return np.zeros((len(questions), 1024), dtype=np.float64)

        torch, _ = _try_import_torch()
        out: list[npt.NDArray[np.float64]] = []
        bs = 32
        for i in range(0, len(questions), bs):
            b_q = questions[i:i + bs]
            b_c = contexts[i:i + bs]
            inputs = self._tokenizer(
                b_q,
                b_c,
                max_length=_MAX_LENGTH,
                truncation=_TRUNCATION,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
                cls_hidden = outputs.last_hidden_state[:, 0, :]
                out.append(cls_hidden.cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)

    def decide_from_embedding(
        self, embedding: npt.NDArray[np.float64] | Any
    ) -> GateDecision:
        """Classify a pre-computed CLS embedding vector (1024-dim) directly.

        Passes embedding through the center_head (or teacher similarity formula)
        and applies thresholds.
        """
        if self._fallback and self._heuristic_gate is not None:
            return self._heuristic_gate("", "")

        torch, _ = _try_import_torch()
        start: float = time.perf_counter()

        if isinstance(embedding, np.ndarray):
            emb_tensor = torch.tensor(embedding, dtype=torch.float32, device=self._device)
            if emb_tensor.dim() == 1:
                emb_tensor = emb_tensor.unsqueeze(0)
        else:
            emb_tensor = embedding.to(self._device)
            if emb_tensor.dim() == 1:
                emb_tensor = emb_tensor.unsqueeze(0)

        with torch.inference_mode():
            if hasattr(self._model, "center_head"):
                logits = self._model.center_head(emb_tensor)
                p_s: npt.NDArray[np.float64] = (
                    torch.softmax(logits, dim=-1).cpu().numpy()[0]
                )
            else:
                p_s = self._center_similarity(emb_tensor)

        elapsed_ms: float = (time.perf_counter() - start) * 1000.0
        return self._apply_thresholds(p_s.tolist(), elapsed_ms)

    def _classify_learned(self, question: str, context: str) -> GateDecision:
        """Run CenterDistill inference.

        Pipeline: tokenize → forward → CLS hidden → center head → softmax → thresholds.
        """
        torch, _ = _try_import_torch()

        start: float = time.perf_counter()

        # Tokenize (spec §5.2: max_length=384, truncation="only_second")
        inputs = self._tokenizer(
            question,
            context,
            max_length=_MAX_LENGTH,
            truncation=_TRUNCATION,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass under inference mode (spec §5.2)
        with torch.inference_mode():
            outputs = self._model(**inputs)
            cls_hidden = outputs.last_hidden_state[:, 0, :]  # CLS token

            # Prefer the trained center_head if the checkpoint includes it
            if hasattr(self._model, "center_head"):
                logits = self._model.center_head(cls_hidden)
                p_s: npt.NDArray[np.float64] = (
                    torch.softmax(logits, dim=-1).cpu().numpy()[0]
                )
            else:
                # Fallback: teacher formula P_T(c_k|q) = softmax(τ · µ̃_kᵀ ê_q)
                p_s = self._center_similarity(cls_hidden)

        elapsed_ms: float = (time.perf_counter() - start) * 1000.0
        return self._apply_thresholds(p_s.tolist(), elapsed_ms)

    def _center_similarity(
        self, cls_hidden: Any
    ) -> npt.NDArray[np.float64]:
        """Compute P_S via teacher distribution formula.

        P_T(c_k|q) = softmax(τ · µ̃_kᵀ ê_q), τ = 10.0
        Used when checkpoint doesn't include a trained center_head module.
        """
        assert self._centers is not None  # guaranteed by _fallback=False path

        cls_np: npt.NDArray[np.float64] = (
            cls_hidden.cpu().float().numpy()[0].astype(np.float64)
        )
        # L2-normalise
        cls_norm: npt.NDArray[np.float64] = cls_np / (
            np.linalg.norm(cls_np) + 1e-8
        )
        centers_norm: npt.NDArray[np.float64] = self._centers / (
            np.linalg.norm(self._centers, axis=1, keepdims=True) + 1e-8
        )
        similarities: npt.NDArray[np.float64] = centers_norm @ cls_norm

        # Softmax with teacher temperature
        logits: npt.NDArray[np.float64] = _TEACHER_TAU * similarities
        exp_logits: npt.NDArray[np.float64] = np.exp(logits - logits.max())
        p_s: npt.NDArray[np.float64] = exp_logits / exp_logits.sum()

        return p_s

    def _apply_thresholds(
        self, distribution: list[float], latency_ms: float
    ) -> GateDecision:
        """Apply threshold evaluation in the FIXED order.

        Order (AGENTS.md rule 2):
            1. max_prob > tau_conf        → ANSWER
            2. second_mass > tau_multi    → ALTERNATIVES
            3. entropy > tau_ent          → CLARIFY
            4. else                       → CLARIFY (safe default — asking is cheaper)

        DO NOT reorder — it silently changes results.
        """
        p: npt.NDArray[np.float64] = np.array(distribution, dtype=np.float64)
        max_prob: float = float(p.max())
        sorted_probs: list[float] = sorted(distribution, reverse=True)
        second_mass: float = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        # Entropy in NATS — np.log, NEVER log2 (AGENTS.md rule 1)
        entropy: float = float(-(p * np.log(p)).sum())

        # Threshold evaluation — FIXED ORDER (AGENTS.md rule 2)
        behaviour: Behaviour
        if max_prob > self._thresholds.tau_conf:               # noqa: E501
            behaviour = "ANSWER"
        elif second_mass > self._thresholds.tau_multi:
            behaviour = "ALTERNATIVES"
        elif entropy > self._thresholds.tau_ent:
            behaviour = "CLARIFY"
        else:
            # DECISION: safe default — asking is cheaper than answering wrongly.
            behaviour = "CLARIFY"

        return GateDecision(
            behaviour=behaviour,
            center_distribution=distribution,
            max_prob=max_prob,
            entropy=entropy,
            second_mass=second_mass,
            thresholds=self._thresholds.as_dict(),
            latency_ms=round(latency_ms, 2),
            fallback_used=False,
        )
