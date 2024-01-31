import numpy as np
import torch

from data_loader.hybrid_data_loaders import HybridTableLoader, WikiHybridTableDataset
from utils.util import load_entity_vocab

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    mode = 0
    data_dir = "/home/belerico/Desktop/unimib/data/tables/turl/"
    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
    train_dataset = WikiHybridTableDataset(
        data_dir,
        entity_vocab,
        max_cell=100,
        max_input_tok=350,
        max_input_ent=150,
        src="dev",
        max_length=[50, 10, 10],
        force_new=True,
        tokenizer=None,
        mode=mode,
        dry_run=False,
    )
    train_data_generator = HybridTableLoader(
        train_dataset,
        10,
        num_workers=0,
        mlm_probability=0.5,
        ent_mlm_probability=0.5,
        is_train=False,
        use_cand=True,
        mode=mode,
    )
    print()
