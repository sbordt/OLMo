from olmo.config import TrainConfig
from olmo.tokenizer import Tokenizer
from olmo.data import build_train_dataloader

import numpy as np
from typing import Dict, Optional, List, Tuple


def create_olmo_insert_dict(insert_dict: Dict[int, str], 
                            olmo_config_path: str,
                            auto_correct_splits: bool = True,
                            global_indices_path: Optional[str] = None) -> Dict[int, list]:
    """This function takes a dictionary with global token positions and strings and converts it into a datastructure that specifies how the strings should be inserted into the OLMo training data.

    The function takes care of:
    - tokenization
    - determine the sequence indices in the global_indices array that correspond to the global token positions
    - split overly long sequences across multiple sequences if necessary, or auto-correct the insertion position to avoid splits across sequences

    Example Input:
        insert_dict = {
            5: "Das scheint ja zu funktionieren!", 
            4096: "Ja, wirklich!", 
            2*4096-2: "Der boy hier wird gesplittet!
        }

    Example Output:
        {
            374605203: [(5, [100257, 33717, 71351, 396, 12203, 6529, 69412, 16414, 0, 100257])], 
            566493791: [(0, [100257, 53545, 11, 56913, 0, 100257]), (4085, [100257, 22960, 8334, 12694, 15165, 14748, 501, 1468, 295, 0, 100257])]
        }
    """
    # olmo setup. we need the sequence length, tokenizer and the global indices of the IterableDataset
    cfg = TrainConfig.load(olmo_config_path)
    sequence_length = cfg.model.max_sequence_length
    tokenizer = Tokenizer.from_train_config(cfg)
    if global_indices_path:
        global_indices = np.memmap(global_indices_path, mode="r+", dtype=np.uint32)
    else:
        # build the train dataloader to get the global indices. this is a bit of a hack, but it works.
        cfg.device_train_batch_size = 2 # if we do not set this we get an assertion error in build_train_dataloader
        cfg.save_overwrite = True # if we do not set this, we get an error if the folder already exists. might want to change this in the future.
        dataloader = build_train_dataloader(cfg)
        dataset = dataloader.dataset
        global_indices = dataset.get_global_indices()
        
    # tokenize
    tokenized_insert_dict = {k: [tokenizer.eos_token_id] + tokenizer.encode(v) for k, v in insert_dict.items()}

    # derive the resulting insertions into sequences of the training data
    sequence_insert_dict = {}
    for global_pos, tokens in tokenized_insert_dict.items():
        sequence_idx = global_pos // sequence_length
        in_sequence_pos = global_pos % sequence_length
        num_tokens = len(tokens)
        if not sequence_idx in sequence_insert_dict:
            sequence_insert_dict[sequence_idx] = []
        if in_sequence_pos + num_tokens > sequence_length and auto_correct_splits:  # try to correct the insertion position to avoid a split across sequences
            if num_tokens < sequence_length:
                in_sequence_pos = sequence_length - num_tokens
                print(f"Auto-corrected insertion position {global_pos} to {sequence_idx * sequence_length + in_sequence_pos} to avoid split across sequences.")
        while in_sequence_pos + num_tokens > sequence_length:  # we need to split the token sequence across batch sequences 
                                                               # this means that we need to insert the first part in the current sequence and the rest in the next sequence
            sequence_insert_dict[sequence_idx].append((in_sequence_pos, tokens[:sequence_length - in_sequence_pos]))
            tokens = tokens[sequence_length - in_sequence_pos:]
            num_tokens = len(tokens)
            in_sequence_pos = 0
            sequence_idx += 1
            print(f"Splitted tokens {tokens} into sequence {sequence_idx}.")
        # regular insertion
        if not sequence_idx in sequence_insert_dict:
            sequence_insert_dict[sequence_idx] = []
        sequence_insert_dict[sequence_idx].append((in_sequence_pos, tokens))

    # convert the trainind data indices into memmap dataset indices
    memmap_insert_dict = {}
    for sequence_idx, insert_list in sequence_insert_dict.items():
        memmap_insert_dict[global_indices[sequence_idx]] = insert_list

    return memmap_insert_dict