"""Object-message product codebook quantization.

This is an object-vector adaptation of CodeFilling's shared multi-codebook
idea. It uses deterministic nearest-codeword assignment at deployment so the
transmitted code IDs have an exact fixed bitrate.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ProductCodebook(nn.Module):
    """Independent codebook per contiguous vector segment."""

    format_version = 1

    def __init__(self, dimension: int, groups: int, entries: int):
        super().__init__()
        if dimension <= 0 or groups <= 0 or entries <= 1:
            raise ValueError("dimension/groups/entries must be positive; entries > 1")
        if dimension % groups:
            raise ValueError("dimension must be divisible by groups")
        self.dimension = int(dimension)
        self.groups = int(groups)
        self.entries = int(entries)
        self.segment_dimension = self.dimension // self.groups
        self.register_buffer(
            "codebook",
            torch.empty(self.groups, self.entries, self.segment_dimension),
        )

    @property
    def bits_per_vector(self) -> int:
        return self.groups * math.ceil(math.log2(self.entries))

    @property
    def bytes_per_vector(self) -> int:
        return math.ceil(self.bits_per_vector / 8)

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        """Return one discrete code index per vector segment."""
        if vectors.ndim != 2 or vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected [count, {self.dimension}] vectors, got {tuple(vectors.shape)}"
            )
        values = vectors.reshape(-1, self.groups, self.segment_dimension)
        distance = (
            values.square().sum(-1, keepdim=True)
            + self.codebook.square().sum(-1).unsqueeze(0)
            - 2 * torch.einsum("ngd,ged->nge", values, self.codebook)
        )
        return distance.argmin(dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[-1] != self.groups:
            raise ValueError(
                f"Expected [count, {self.groups}] codes, got {tuple(codes.shape)}"
            )
        group_index = torch.arange(self.groups, device=codes.device)
        return self.codebook[group_index[None], codes].reshape(-1, self.dimension)

    def roundtrip(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(vectors)
        return self.decode(codes), codes

    @torch.no_grad()
    def fit(
        self,
        vectors: torch.Tensor,
        *,
        iterations: int = 25,
        chunk_size: int = 4096,
        seed: int = 42,
    ) -> dict:
        """Fit k-means centroids independently for every vector segment."""
        if vectors.ndim != 2 or vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected [count, {self.dimension}] vectors, got {tuple(vectors.shape)}"
            )
        if len(vectors) < self.entries:
            raise ValueError(
                f"Need at least {self.entries} vectors to initialize the codebook"
            )
        values = vectors.to(self.codebook.device, dtype=self.codebook.dtype)
        generator = torch.Generator(device=values.device).manual_seed(seed)
        initial = torch.randperm(
            len(values), generator=generator, device=values.device
        )[: self.entries]
        self.codebook.copy_(
            values[initial].reshape(
                self.entries, self.groups, self.segment_dimension
            ).transpose(0, 1)
        )
        for _ in range(iterations):
            sums = torch.zeros_like(self.codebook)
            counts = torch.zeros(
                self.groups, self.entries, device=values.device, dtype=torch.long
            )
            for chunk in values.split(chunk_size):
                codes = self.encode(chunk)
                pieces = chunk.reshape(-1, self.groups, self.segment_dimension)
                for group in range(self.groups):
                    sums[group].index_add_(0, codes[:, group], pieces[:, group])
                    counts[group].scatter_add_(
                        0, codes[:, group], torch.ones_like(codes[:, group])
                    )
            active = counts > 0
            self.codebook[active] = (
                sums[active] / counts[active].to(sums.dtype).unsqueeze(-1)
            )
            for group, entry in (~active).nonzero(as_tuple=False).tolist():
                index = torch.randint(
                    len(values), (), generator=generator, device=values.device
                )
                self.codebook[group, entry] = values[
                    index
                ].reshape(self.groups, self.segment_dimension)[group]
        decoded, codes = self.roundtrip(values)
        usage = torch.stack(
            [
                torch.bincount(codes[:, group], minlength=self.entries)
                for group in range(self.groups)
            ]
        )
        return {
            "reconstruction_mse": float((decoded - values).square().mean()),
            "active_code_fraction": float((usage > 0).float().mean()),
            "bits_per_vector": self.bits_per_vector,
            "bytes_per_vector": self.bytes_per_vector,
        }


class ResidualProductCodebook(nn.Module):
    """Progressive object-message quantizer built from residual codebooks.

    Each stage quantizes the residual left by the preceding stages.  A receiver
    can therefore decode a prefix of stages: later messages refine, rather
    than replace, an earlier reconstruction.  This adapts CodeFilling's
    stacked residual-codebook idea to one 256-D TrackFormer object message.
    """

    format_version = 1

    def __init__(self, dimension: int, stages: list[tuple[int, int]]):
        super().__init__()
        if not stages:
            raise ValueError("At least one residual codebook stage is required")
        self.dimension = int(dimension)
        self.stage_specs = tuple((int(groups), int(entries)) for groups, entries in stages)
        self.stages = nn.ModuleList(
            ProductCodebook(self.dimension, groups, entries)
            for groups, entries in self.stage_specs
        )

    @property
    def stage_bits(self) -> list[int]:
        return [stage.bits_per_vector for stage in self.stages]

    @property
    def cumulative_bits(self) -> list[int]:
        total = 0
        result = []
        for bits in self.stage_bits:
            total += bits
            result.append(total)
        return result

    def encode(self, vectors: torch.Tensor, *, stages: int | None = None) -> list[torch.Tensor]:
        """Encode a vector as a progressive list of stage-code tensors."""
        if vectors.ndim != 2 or vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected [count, {self.dimension}] vectors, got {tuple(vectors.shape)}"
            )
        stage_count = len(self.stages) if stages is None else int(stages)
        if not 1 <= stage_count <= len(self.stages):
            raise ValueError("stages must select a non-empty prefix")
        residual = vectors
        codes = []
        for stage in self.stages[:stage_count]:
            code = stage.encode(residual)
            residual = residual - stage.decode(code)
            codes.append(code)
        return codes

    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        if not codes:
            raise ValueError("At least one stage of codes is required")
        if len(codes) > len(self.stages):
            raise ValueError("Too many code stages")
        decoded = self.stages[0].decode(codes[0])
        for stage, code in zip(self.stages[1:], codes[1:]):
            decoded = decoded + stage.decode(code)
        return decoded

    def roundtrip(
        self, vectors: torch.Tensor, *, stages: int | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        codes = self.encode(vectors, stages=stages)
        return self.decode(codes), codes

    @torch.no_grad()
    def fit(
        self,
        vectors: torch.Tensor,
        *,
        iterations: int = 25,
        chunk_size: int = 4096,
        seed: int = 42,
    ) -> dict:
        """Fit stages sequentially to the remaining reconstruction residual."""
        if vectors.ndim != 2 or vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected [count, {self.dimension}] vectors, got {tuple(vectors.shape)}"
            )
        residual = vectors.to(
            self.stages[0].codebook.device,
            dtype=self.stages[0].codebook.dtype,
        )
        metrics = []
        for stage_index, stage in enumerate(self.stages):
            stage_metrics = stage.fit(
                residual,
                iterations=iterations,
                chunk_size=chunk_size,
                seed=seed + stage_index,
            )
            approximation, _ = stage.roundtrip(residual)
            residual = residual - approximation
            metrics.append(
                {
                    **stage_metrics,
                    "stage": stage_index + 1,
                    "cumulative_bits": self.cumulative_bits[stage_index],
                    "residual_mse": float(residual.square().mean()),
                }
            )
        return {
            "stage_metrics": metrics,
            "cumulative_bits": self.cumulative_bits,
            "final_reconstruction_mse": float(residual.square().mean()),
        }


def checkpoint_payload(
    quantizer: ProductCodebook,
    *,
    representation: str,
    source_checkpoint: str,
    fit_metrics: dict,
) -> dict:
    return {
        "format": "object_product_codebook_v1",
        "representation": representation,
        "source_checkpoint": source_checkpoint,
        "dimension": quantizer.dimension,
        "groups": quantizer.groups,
        "entries": quantizer.entries,
        "bits_per_vector": quantizer.bits_per_vector,
        "bytes_per_vector": quantizer.bytes_per_vector,
        "fit_metrics": fit_metrics,
        "state_dict": quantizer.state_dict(),
    }


def load_product_codebook(path, device) -> tuple[ProductCodebook, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "object_product_codebook_v1":
        raise ValueError(f"{path} is not an object product-codebook checkpoint")
    quantizer = ProductCodebook(
        int(saved["dimension"]), int(saved["groups"]), int(saved["entries"])
    ).to(device)
    quantizer.load_state_dict(saved["state_dict"], strict=True)
    quantizer.eval()
    return quantizer, saved


def residual_checkpoint_payload(
    quantizer: ResidualProductCodebook,
    *,
    representation: str,
    source_checkpoint: str,
    fit_metrics: dict,
) -> dict:
    return {
        "format": "object_residual_product_codebook_v1",
        "representation": representation,
        "source_checkpoint": source_checkpoint,
        "dimension": quantizer.dimension,
        "stage_specs": list(quantizer.stage_specs),
        "stage_bits": quantizer.stage_bits,
        "cumulative_bits": quantizer.cumulative_bits,
        "fit_metrics": fit_metrics,
        "state_dict": quantizer.state_dict(),
    }


def load_residual_product_codebook(
    path, device
) -> tuple[ResidualProductCodebook, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "object_residual_product_codebook_v1":
        raise ValueError(f"{path} is not a residual object codebook checkpoint")
    quantizer = ResidualProductCodebook(
        int(saved["dimension"]),
        [tuple(stage) for stage in saved["stage_specs"]],
    ).to(device)
    quantizer.load_state_dict(saved["state_dict"], strict=True)
    quantizer.eval()
    return quantizer, saved
