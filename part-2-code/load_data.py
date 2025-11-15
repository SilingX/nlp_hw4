import os, re, torch, nltk, pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

nltk.download("punkt")

PAD_TOKEN = 0


def _normalize_sql(sql_text):
    import re
    sql = re.sub(r"\s+", " ", sql_text.strip())
    if not sql.endswith(";"):
        sql += ";"
    return sql


class T5Dataset(Dataset):
    def __init__(self, folder, split):
        assert split in {"train", "dev", "test"}
        self.folder = folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.init_tok = "<extra_id_0>"
        self.init_id = self.tokenizer.convert_tokens_to_ids(self.init_tok)
        self.samples = self._load_split(folder, split)

    def _load_split(self, folder, split):
        nlp = read_lines(os.path.join(folder, f"{split}.nl"))
        sqls = read_lines(os.path.join(folder, f"{split}.sql")) if split != "test" else []
        data = []
        for i, text in enumerate(nlp):
            prompt = f"translate question to SQL: {text.strip().lower()}"
            enc = self.tokenizer(prompt, truncation=True, max_length=512)
            enc_ids = torch.tensor(enc["input_ids"])
            enc_mask = torch.tensor(enc["attention_mask"])
            if split == "test" or i >= len(sqls):
                dec_in, dec_tgt = None, None
            else:
                sql_norm = _normalize_sql(sqls[i])
                sql_ids = self.tokenizer(sql_norm, truncation=True, max_length=512)["input_ids"]
                pad_id = self.tokenizer.pad_token_id
                dec_tgt = torch.tensor(sql_ids)
                dec_in = torch.tensor([pad_id] + sql_ids[:-1])
            data.append(
                {"enc_ids": enc_ids, "enc_mask": enc_mask,
                 "dec_in": dec_in, "dec_tgt": dec_tgt,
                 "init_dec": self.init_id}
            )
        return data

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_train(batch):
    enc_ids = [b["enc_ids"] for b in batch]
    enc_mask = [b["enc_mask"] for b in batch]
    dec_in = [b["dec_in"] for b in batch]
    dec_tgt = [b["dec_tgt"] for b in batch]
    init = [b["init_dec"] for b in batch]

    enc_ids = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_TOKEN)
    enc_mask = pad_sequence(enc_mask, batch_first=True, padding_value=0)

    if any(x is None for x in dec_in):
        dec_in_pad = torch.zeros((len(batch), 0), dtype=torch.long)
        dec_tgt_pad = torch.zeros((len(batch), 0), dtype=torch.long)
    else:
        dec_in_pad = pad_sequence(dec_in, batch_first=True, padding_value=PAD_TOKEN)
        dec_tgt_pad = pad_sequence(dec_tgt, batch_first=True, padding_value=PAD_TOKEN)

    return enc_ids, enc_mask, dec_in_pad, dec_tgt_pad, torch.tensor(init)


def collate_test(batch):
    enc_ids = [b["enc_ids"] for b in batch]
    enc_mask = [b["enc_mask"] for b in batch]
    init = [b["init_dec"] for b in batch]
    enc_ids = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_TOKEN)
    enc_mask = pad_sequence(enc_mask, batch_first=True, padding_value=0)
    return enc_ids, enc_mask, torch.tensor(init)


def make_loader(batch_size, split):
    dataset = T5Dataset("data", split)
    shuffle = split == "train"
    collate_fn = collate_test if split == "test" else collate_train
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_dataloaders(train_bs, test_bs):
    return make_loader(train_bs, "train"), make_loader(test_bs, "dev"), make_loader(test_bs, "test")


def read_lines(p):
    return [x.strip() for x in open(p).readlines()] if os.path.exists(p) else []
