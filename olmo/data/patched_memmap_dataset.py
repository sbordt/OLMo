from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, cast

from .memmap_dataset import MemMapDataset

from ..exceptions import OLMoConfigurationError

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig

import torch
import h5py

__all__ = ["PatchedMemMapDataset", "build_patched_memmap_dataset"]


class HDF5Loader:
    """
    Efficient read-only loader for HDF5 data with fast index checking and loading.
    Optimized for random access patterns.
    """
    
    def __init__(self, hdf5_path: str, cache_size_mb: int = 64):
        """
        Initialize the HDF5 loader.
        
        Args:
            hdf5_path: Path to the HDF5 file
            cache_size_mb: Cache size in MB for HDF5 chunk cache
        """
        self.hdf5_path = hdf5_path
        self.cache_size_mb = cache_size_mb
        
        # File handle and datasets (initialized lazily)
        self._f = None
        self._keys_array = None
        self._key_to_idx = None
        self._is_optimized = None
        
        # Dataset handles (for optimized format)
        self._tuple_ints = None
        self._int_lists_vlen = None
        self._data_group = None
        
        # Load indices immediately
        self._load_indices()
        
        print(f"HDF5Loader initialized with {len(self._keys_array)} indices")
    
    def _ensure_file_open(self):
        """Lazy file opening with optimal cache settings"""
        if self._f is None:
            cache_bytes = self.cache_size_mb * 1024 * 1024
            self._f = h5py.File(
                self.hdf5_path, 'r',
                rdcc_nbytes=cache_bytes,
                rdcc_nslots=10007  # Prime number for hash efficiency
            )
            
            # Check if this is optimized format or simple format
            if 'tuple_ints' in self._f:
                self._is_optimized = True
                self._tuple_ints = self._f['tuple_ints']
                self._int_lists_vlen = self._f['int_lists_vlen']
            else:
                self._is_optimized = False
                self._data_group = self._f['data']
    
    def _load_indices(self):
        """Load all available indices from the HDF5 file"""
        # Temporarily open file just to read keys
        with h5py.File(self.hdf5_path, 'r') as f:
            self._keys_array = f['keys'][:]
        
        # Create fast lookup dictionary
        self._key_to_idx = {int(key): idx for idx, key in enumerate(self._keys_array)}
        
        # Convert keys to a set for O(1) membership testing
        self._keys_set = set(int(key) for key in self._keys_array)
    
    def has_index(self, index: int) -> bool:
        """
        Check if an index exists in the file without accessing the file.
        
        Args:
            index: The index to check
            
        Returns:
            True if index exists, False otherwise
        """
        return index in self._keys_set
    
    def load(self, index: int) -> Optional[List[Tuple[int, List[int]]]]:
        """
        Load data for a specific index efficiently.
        
        Args:
            index: The index to load data for
            
        Returns:
            List of tuples (int, list[int]) or None if index doesn't exist
        """
        if not self.has_index(index):
            return None
        
        self._ensure_file_open()
        
        if self._is_optimized:
            return self._load_optimized(index)
        else:
            return self._load_simple(index)
    
    def _load_optimized(self, index: int) -> List[Tuple[int, List[int]]]:
        """Load from optimized HDF5 format using variable-length arrays"""
        idx = self._key_to_idx[index]
        
        # Read data for this index
        tuple_ints = self._tuple_ints[idx]
        int_lists_vlen = self._int_lists_vlen[idx]
        
        # Reconstruct the original data structure
        result = []
        for j in range(len(tuple_ints)):
            if tuple_ints[j] == -1:  # End of valid data
                break
            
            tuple_int = int(tuple_ints[j])
            int_list = int_lists_vlen[j].tolist() if int_lists_vlen[j] is not None else []
            result.append((tuple_int, int_list))
        
        return result
    
    def _load_simple(self, index: int) -> List[Tuple[int, List[int]]]:
        """Load from simple HDF5 format"""
        key_str = str(index)
        if key_str not in self._data_group:
            return None
        
        key_group = self._data_group[key_str]
        num_tuples = key_group.attrs['num_tuples']
        
        result = []
        for i in range(num_tuples):
            tuple_group = key_group[f'tuple_{i}']
            tuple_int = int(tuple_group.attrs['tuple_int'])
            int_list = tuple_group['int_list'][:].tolist()
            result.append((tuple_int, int_list))
        
        return result
    
    def get_all_indices(self) -> List[int]:
        """
        Get all available indices.
        
        Returns:
            List of all indices in the file
        """
        return [int(key) for key in self._keys_array]
    
    def __len__(self) -> int:
        """Return number of indices in the file"""
        return len(self._keys_array)
    
    def __contains__(self, index: int) -> bool:
        """Support 'in' operator for checking index existence"""
        return self.has_index(index)
    
    def __del__(self):
        """Clean up file handle"""
        if hasattr(self, '_f') and self._f is not None:
            self._f.close()


class PatchedMemMapDataset(MemMapDataset):
    def __init__(
        self,
        *paths: str | Path,
        sequence_insert_dict: Dict,
        hdf5_insert_storage_file: str | None = None,
        **kwargs,
    ):
        super().__init__(*paths, **kwargs)
        self.sequence_insert_dict = sequence_insert_dict
        self.hdf5_insert_storage_file = hdf5_insert_storage_file
        self.hdf5_insert_indices = None

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        # claude recommends to create the file handle lazily, so that we dont run into issues with multiple workers.
        if self.hdf5_insert_indices is None:
            if self.hdf5_insert_storage_file is None:
                self.hdf5_insert_indices = set()
            else:
                self.hdf5_insert_loader = HDF5Loader(self.hdf5_insert_storage_file)
                self.hdf5_insert_indices = set(self.hdf5_insert_loader.get_all_indices())

        item = super().__getitem__(idx)

        if idx in self.hdf5_insert_indices:
            hdf5_entry = self.hdf5_insert_loader.load(idx)
            if hdf5_entry is None:
                raise ValueError(f"PatchedMemMapDataset: Index {idx} not found in HDF5 storage. This should not happen. Aborting.")
            for in_sequence_pos, tokens in hdf5_entry:
                tokens = torch.as_tensor(tokens, dtype=item["input_ids"].dtype)
                item["input_ids"][in_sequence_pos:in_sequence_pos + len(tokens)] = tokens

        if idx in self.sequence_insert_dict: # pickle file has precedence over hdf5 file
            for in_sequence_pos, tokens in self.sequence_insert_dict[idx]:                           
                tokens = torch.as_tensor(tokens, dtype=item["input_ids"].dtype)             
                item["input_ids"][in_sequence_pos:in_sequence_pos + len(tokens)] = tokens   
        return item
    

def build_patched_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, memmap_insert_dict: Dict, hdf5_insert_storage_file: str | None, include_instance_metadata: bool = True
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
        hdf5_insert_storage_file=hdf5_insert_storage_file,
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