from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, cast

from .memmap_dataset import MemMapDataset

from ..exceptions import OLMoConfigurationError

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig

import torch

__all__ = ["PatchedMemMapDataset", "build_patched_memmap_dataset"]


class PatchedMemMapDataset(MemMapDataset):
    def __init__(
        self,
        *paths: str | Path,
        sequence_insert_dict: Dict,
        **kwargs,
    ):
        super().__init__(*paths, **kwargs)
        self.sequence_insert_dict = sequence_insert_dict

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)

        if idx in self.sequence_insert_dict:
            for in_sequence_pos, tokens in self.sequence_insert_dict[idx]:    # modify the input_ids in-place                       
                tokens = torch.as_tensor(tokens, dtype=item["input_ids"].dtype)             
                item["input_ids"][in_sequence_pos:in_sequence_pos + len(tokens)] = tokens   
        return item
    

def build_patched_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, memmap_insert_dict: Dict, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return PatchedMemMapDataset(
        *paths,
        sequence_insert_dict=memmap_insert_dict,
        chunk_size=train_config.model.max_sequence_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        eos_token_id=train_config.model.eos_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        generate_doc_lengths=data_config.generate_doc_lengths,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
        instance_filter_config=data_config.instance_filter,
    )