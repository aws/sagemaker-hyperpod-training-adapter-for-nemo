import torch


class DummyDataset(torch.utils.data.dataset.Dataset):
    """Dummy Dataset."""

    def __init__(self, vocab_size=1024, seqlen=2048, length=100000, data_type="gpt"):
        self.vocab_size = vocab_size
        self.seqlen = seqlen
        if data_type == "gpt":
            self.mask = torch.ones((seqlen,))
        elif data_type == "bert":
            raise NotImplementedError
        self.length = length
        self.input_paths = None

    def __getitem__(self, index):
        return torch.randint(self.vocab_size, (self.seqlen,), dtype=torch.long), self.mask

    def __len__(self):
        return self.length
