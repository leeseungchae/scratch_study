from abc import ABC
from typing import Any, Dict, Tuple
import sentencepiece as spm
from omegaconf import DictConfig

from nlp.model.seq2seq import Seq2Seq
from nlp.datasets.data_helper import create_or_load_tokenizer, TrainDataset
from torch.utils.data import DataLoader, RandomSampler


class ABstracTools(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.arg = cfg
        self.src_vocab = self.trg_vocab = self.get_vocab()

    def get_params(self) -> Dict[str, Any]:
        model_type = self.arg.model.model_type
        print(model_type)

        if model_type == 'seq2seq':
            params = {
                'enc_d_input': self.arg.data.src_vocab_size,
                'dec_d_input': self.arg.data.trg_vocab_size,
                'd_hidden': self.arg.model.d_hidden,
                'n_layers': self.arg.model.n_layers,
                'mode': self.arg.model.mode,
                'dropout_rate': self.arg.model.dropout_rate,
                'bidirectional': self.arg.model.bidirectional,
                'bias': self.arg.model.bias,
                'batch_first': self.arg.model.batch_first,
                'max_sequence_size': self.arg.model.max_sequence_size}
        else:
            raise ValueError("param 'model_type' must be one of [seq2seq]")
        return params

    def get_model(self):
        model_type = self.arg.model.model_type
        params = self.get_params()
        if model_type == 'seq2seq':
            model = Seq2Seq(**params)
        else:
            raise ValueError("param 'model_type' must be one of [Seq2seq]")

        return model

    def get_vocab(self) -> Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:

        src_vocab = create_or_load_tokenizer(
            file_path=self.arg.data.src_train_path,
            save_path=self.arg.data.dictionary_path,
            language=self.arg.data.src_language,
            vocab_size=self.arg.data.src_language,
            tokenizer_type=self.arg.data.tokenizer,
            bos_id=self.arg.data.bos_id,
            eos_id=self.arg.data.eos_id,
            unk_id=self.arg.data.unk_id,
            pad_id=self.arg.data.pad_id
        )
        trg_vocab = create_or_load_tokenizer(
            file_path=self.arg.data.trg_train_path,
            save_path=self.arg.data.dictionary_path,
            language=self.arg.data.trg_language,
            vocab_size=self.arg.data.trg_language,
            tokenizer_type=self.arg.data.tokenizer,
            bos_id=self.arg.data.bos_id,
            eos_id=self.arg.data.eos_id,
            unk_id=self.arg.data.unk_id,
            pad_id=self.arg.data.pad_id
        )

        return src_vocab, trg_vocab

    def get_loader(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TrainDataset(
            x_path=self.arg.data.src_train_path,
            src_vocab=self.src_vocab,
            y_path=self.arg.data.trg_train_path,
            trg_vocab=self.trg_vocab,
            max_sequence_size=self.arg.model.max_sequence_size
        )

        valid_dataset = TrainDataset(
            x_path=self.arg.data.src_train_path,
            src_vocab=self.src_vocab,
            y_path=self.arg.data.trg_train_path,
            trg_vocab=self.trg_vocab,
            max_sequence_size=self.arg.model.max_sequence_size
        )

        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)

        train_loader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=self.arg.trainer.batch_size)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  sampler=valid_sampler,
                                  batch_size=self.arg.trainer.batch_size)
    @staticmethod
    def tensor2sentence(indices: Tensor, vocab:spm.SentencePieceProcessor) ->str:
        result = []
        for idx in indices:
            word = vocab.IdToPiece(idx)
            if word == 'pad':
                break
            result.append(word)

        return "".join(result).replace("_", " ").strip()

    @staticmethod
    def print_result(
            input_sentence:str,
            predict_sentence:str,
            target_sentence:str
    ) ->None:
        print(f"------Test-----")
        print(f"Source  : {input_sentence}")
        print(f"Predict : {predict_sentence}")
        print(f"Target  : {target_sentence}")