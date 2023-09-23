"""class for managing data from the tiny shakespeare dataset"""
from minigpt.loaders.text_dataset import TextDataset

SCHAR_VOCAB_SIZE = 65  # 65 unique chars in the tiny shakespare dataset


class TinyShakespeareCharData(TextDataset):
    def __init__(self, args):
        super().__init__(args, "tiny_shakespeare.txt")
        self.train_ids, self.val_ids = self.load_dataset()

    @classmethod
    def get_vocab_size(cls, _source, _vocab_soure: str | None = None):
        """Get the vocab size based on the source"""
        return SCHAR_VOCAB_SIZE

    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Shakespeare (Character tokens)"

    def is_prepared(self) -> bool:
        """Check if the data(bin_files) have been prepared"""
        bin_files_exist = (self.work_dir / self.train_bin).exists() and (
            self.work_dir / self.val_bin
        ).exists()
        return bin_files_exist

    def download(self, force=False):
        # download the tiny shakespeare dataset
        # data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_url = (
            "https://raw.githubusercontent.com/christyxgeorge/datasets/main/tiny_shakespeare.txt"
        )
        self.download_file(data_url, self.filename)

    def load_dataset(self):
        if self.verbose:
            print("=" * 100)
            print("Loading Data [{self.filename}]...")
            print("=" * 100)
        with open(self.filename, "r", encoding="utf-8") as f:
            text = f.read()

        # Unique Characters in the text
        self.vocab_chars = sorted(list(set(text)))
        self.vocab_size = len(self.vocab_chars)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab_chars)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab_chars)}

        text = self.load_vocab_chars()
        token_ids = self.encode(text)

        # Split in train, val
        tv_split = int(0.9 * len(token_ids))
        train_ids = token_ids[:tv_split]
        val_ids = token_ids[tv_split:]
        return train_ids, val_ids

    def get_token_ids(self) -> tuple[list[int], list[int]]:
        """Load token IDs from file"""
        return self.train_ids, self.val_ids

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return [self.stoi[c] for c in s]

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join([self.itos[i] for i in l])
