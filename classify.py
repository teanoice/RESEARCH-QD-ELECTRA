# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from transformers import (
    ElectraConfig,
    BertConfig,
    ElectraForPreTraining,
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForSequenceClassification
)

from QDElectra_model import (
    DistillElectraForSequenceClassification,
    QuantizedElectraForSequenceClassification
)
from typing import NamedTuple
import tokenization
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None

    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r", encoding='utf-8') as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b


class COLA(CsvDataset):
    """Dataset class for COLA"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file,pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[1], line[3]


class SST2(CsvDataset):
    """Dataset class for SST2"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file,pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[1], line[0]


class STSB(CsvDataset):
    """Dataset class for STSB"""
    labels = [None]

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[7], line[8]


class QQP(CsvDataset):
    """Dataset class for QQP"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[5], line[3], line[4]


class QNLI(CsvDataset):
    """Dataset class for QNLI"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[2], line[1]


class RTE(CsvDataset):
    """Dataset class for RTE"""
    labels = ("entailment", "not_entailment")

    def __init__(self, file, pipeline=[]):
        super().__init__(file,pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[1], line[2]


class WNLI(CsvDataset):
    """Dataset class for WNLI"""
    labels = ("0", "1")

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[-1], line[1], line[2]


def dataset_class(task_name):
    """ Mapping from task string to Dataset Class """
    table = {
        'mrpc':  MRPC,
        'mnli':  MNLI,
        'cola':  COLA,
        'sst-2': SST2,
        'sts-b': STSB,
        'qqp':   QQP,
        'qnli':  QNLI,
        'rte':   RTE,
        'wnli':  WNLI
    }
    return table[task_name]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) if text_b else []

        return label, tokens_a, tokens_b


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return label, tokens_a, tokens_b


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, output_mode ,max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.output_mode = output_mode
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        token_type_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        attention_mask = [1]*(len(tokens_a) + len(tokens_b))
        if self.output_mode == "classifcation":
            label_id = self.label_map[label]
        else:
            label_id = float(label)
        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        token_type_ids.extend([0]*n_pad)
        attention_mask.extend([0]*n_pad)

        return input_ids, attention_mask, token_type_ids, label_id


class QuantizedDistillElectraTrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    temperature: int = 1 # temperature for QD-electra logit loss
    lambda_: int = 50 # lambda for QD-electra discriminator loss

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, input_ids, attention_mask, token_type_ids, label_id):
    features = [input_ids, attention_mask, token_type_ids, label_id]
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths)

    return tensor_data, all_label_ids


class QuantizedDistillElectraTrainer(train.Trainer):
    def __init__(self, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.bceLoss = nn.BCELoss()
        self.mseLoss = nn.MSELoss()
        self.ceLoss = nn.CrossEntropyLoss()

    def get_loss(self, model, batch, global_step, train_cfg, model_cfg): # make sure loss is tensor
        t_outputs, s_outputs, s2t_hidden_states = model(*batch)

        t_outputs.loss *= train_cfg.lambda_
        s_outputs.loss *= train_cfg.lambda_

        # -----------------------
        # Get distillation loss
        # 1. teacher and student logits (cross-entropy + temperature)
        # 2. embedding layer loss + hidden losses (MSE)
        # 3. attention matrices loss (MSE)
        #       We only consider :
        #       3-1. Teacher layer numbers equal to student layer numbers
        #       3-2. Teacher head numbers are divisible by Student head numbers
        # -----------------------
        soft_logits_loss = self.bceLoss(
            F.sigmoid(s_outputs.logits / train_cfg.temperature),
            F.sigmoid(t_outputs.logits.detach() / train_cfg.temperature),
        ) * train_cfg.temperature * train_cfg.temperature

        hidden_layers_loss = 0
        for t_hidden, s_hidden in zip(t_outputs.hidden_states, s2t_hidden_states):
            hidden_layers_loss += self.mseLoss(s_hidden, t_hidden.detach())

        # -----------------------
        # teacher attention shape per layer : (batch_size, t_n_heads, max_seq_len, max_seq_len)
        # student attention shape per layer : (batch_size, s_n_heads, max_seq_len, max_seq_len)
        # -----------------------
        atten_layers_loss = 0
        split_sections = [model_cfg.s_n_heads] * (model_cfg.t_n_heads // model_cfg.s_n_heads)
        for t_atten, s_atten in zip(t_outputs.attentions, s_outputs.attentions):
            split_t_attens = torch.split(t_atten.detach(), split_sections, dim=1)
            for i, split_t_atten in enumerate(split_t_attens):
                atten_layers_loss += self.mseLoss(s_atten[:, i, :, :], torch.mean(split_t_atten, dim=1))

        total_loss = s_outputs.loss + t_outputs.loss + soft_logits_loss + hidden_layers_loss + atten_layers_loss

        self.writer.add_scalars(
            'data/scalar_group', {
                't_discriminator_loss': t_outputs.loss.item(),
                's_discriminator_loss': s_outputs.loss.item(),
                'soft_logits_loss': soft_logits_loss.item(),
                'hidden_layers_loss': hidden_layers_loss.item(),
                'attention_loss': atten_layers_loss.item(),
                'total_loss': total_loss.item(),
                'lr': self.optimizer.get_lr()[0]
            }, global_step
        )

        print(f'\tT-Discriminator Loss {t_outputs.loss.item():.3f}\t'
              f'S-Discriminator Loss {s_outputs.loss.item():.3f}\t'
              f'Soft Logits Loss {soft_logits_loss.item():.3f}\t'
              f'Hidden Loss {hidden_layers_loss.item():.3f}\t'
              f'Attention Loss {atten_layers_loss.item():.3f}\t'
              f'Total Loss {total_loss.item():.3f}\t')

        return total_loss


    def evaluate(self, model, task_name, output_mode, eval_labels, num_labels, batch):

        eval_loss = 0
        eval_steps = 0
        preds = []
        input_ids, attention_mask, token_type_ids, label_id = batch

        _, s_outputs, _ = model(input_ids, attention_mask, token_type_ids, label_id )

        if output_mode == "classifcation":
            logits_loss = self.ceLoss()
            tmp_eval_loss = logits_loss (s_outputs.logits.view(-1, num_labels), label_id)
        elif output_mode == "regression":
            logits_loss = self.mseLoss()
            tmp_eval_loss = logits_loss (s_outputs.logits.view(-1), label_id)

        eval_loss += tmp_eval_loss.mean().item()
        eval_steps += 1

        if len(preds) == 0:
            preds.append(s_outputs.logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], s_outputs.logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / eval_steps

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, eval_labels.numpy())
        result['eval_loss'] = eval_loss

        return result


def main(task='mrpc',
         train_cfg='config/train_mrpc.json',
         model_cfg='config/QDElectra_pretrain.json',
         data_file='../glue/MRPC/train.tsv',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         log_dir='../exp/electra/pretrain/runs',
         save_dir='../exp/bert/mrpc',
         max_len=128,
         mode='train',
         pred_distill=True):

    processors = {
        "cola":    COLA,
        "mnli":    MNLI,
        "mrpc":    MRPC,
        "sst-2":   SST2,
        "sts-b":   STSB,
        "qqp":     QQP,
        "qnli":    QNLI,
        "rte":     RTE,
        "wnli":    WNLI
    }

    output_modes = {
        "cola":  "classification",
        "mnli":  "classification",
        "mrpc":  "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp":   "classification",
        "qnli":  "classification",
        "rte":   "classification",
        "wnli":  "classification"
    }

    # intermediate distillation default parameters
    default_params = {
        "cola":  {"n_epochs": 50, "max_len": 64},
        "mnli":  {"n_epochs": 5,  "max_len": 128},
        "mrpc":  {"n_epochs": 20, "max_len": 128},
        "sst-2": {"n_epochs": 10, "max_len": 64},
        "sts-b": {"n_epochs": 20, "max_len": 128},
        "qqp":   {"n_epochs": 5,  "max_len": 128},
        "qnli":  {"n_epochs": 10, "max_len": 128},
        "rte":   {"n_epochs": 20, "max_len": 128}
    }

    # acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    # corr_tasks = ["sts-b"]
    # mcc_tasks = ["cola"]

    if task in default_params:
        max_len = default_params [task]["max_len"]

    if task not in processors:
        raise ValueError("Task not found: %s" % task)

    output_mode = output_modes[task] # classofication or regression task

    train_cfg = QuantizedDistillElectraTrainConfig.from_json(train_cfg)
    train_cfg.n_epochs = default_params [task]['n_epochs']

    model_cfg = ElectraConfig().from_json_file(model_cfg)
    set_seeds(train_cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task) # task dataset class according to the task
    num_labels = len(TaskDataset.labels)
    pipeline = [
        Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
        AddSpecialTokensWithTruncation(max_len),
        TokenIndexing(tokenizer.convert_tokens_to_ids, TaskDataset.labels, output_mode, max_len)
    ]
    data_set = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(data_set, batch_size=train_cfg.batch_size, shuffle=True)

    t_discriminator = ElectraForSequenceClassification.from_pretrained(
        'google/electra-base-discriminator'
    )
    s_discriminator = QuantizedElectraForSequenceClassification.from_pretrained(
        'google/electra-small-discriminator', config=model_cfg
    )
    model = DistillElectraForSequenceClassification(t_discriminator, s_discriminator, model_cfg)

    optimizer = optim.optim4GPU(train_cfg, model)
    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    base_trainer_args = (train_cfg, model_cfg, model, data_iter, optimizer, save_dir, get_device())
    trainer = QuantizedDistillElectraTrainer(writer, *base_trainer_args)

    if mode == 'train':
        trainer.train(model_file, None, data_parallel)
    elif mode == 'eval':
        input_ids, attention_mask, token_type_ids, label_id = TokenIndexing(tokenizer.convert_tokens_to_ids,
                                                                            TaskDataset.labels,
                                                                            output_mode,
                                                                            max_len)
        _, eval_labels = get_tensor_data(output_mode, input_ids, attention_mask, token_type_ids, label_id)
        results = trainer.eval(model_file, output_mode, eval_labels, num_labels, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
