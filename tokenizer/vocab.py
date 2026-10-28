"""Vocabulary manager."""
from typing import Dict, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.bos_token_id: int = 50256
        self.eos_token_id: int = 50256
        self._allow_growth = not vocab_dict

        if vocab_dict:
            # Honor explicit IDs first so special tokens cannot silently shift a model vocab.
            self._token_to_id = {token: int(idx) for token, idx in vocab_dict.items()}
            self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
            self.pad_token_id = self._token_to_id.get(PAD_TOKEN, -1)
            self.unk_token_id = self._token_to_id.get(UNK_TOKEN, -1)
            self.bos_token_id = self._token_to_id.get(BOS_TOKEN, self.bos_token_id)
            self.eos_token_id = self._token_to_id.get(EOS_TOKEN, self.bos_token_id)
        else:
            self.add_token(PAD_TOKEN, 0)
            self.add_token(UNK_TOKEN, 1)
            self.add_token(BOS_TOKEN, self.bos_token_id)
            self._next_id = 2
        if vocab_dict:
            self._next_id = max(self._id_to_token, default=1) + 1

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            if not self._allow_growth:
                if self.unk_token_id >= 0:
                    return self.unk_token_id
                raise KeyError(f"Token {token!r} is not present in this vocabulary")
            idx = self._next_id
            while idx in self._id_to_token:
                idx += 1
            self._next_id = idx + 1
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"token ID {idx} is already assigned to {self._id_to_token[idx]!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return max(self._id_to_token, default=-1) + 1

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
