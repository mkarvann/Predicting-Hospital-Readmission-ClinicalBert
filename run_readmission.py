# coding=utf-8
import os
import logging
import argparse
import random
from tqdm import trange
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_scheduler,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    def __init__(self, input_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


class DataProcessor:
    @classmethod
    def _read_csv(cls, input_file):
        file = pd.read_csv(input_file)
        return zip(file.ID, file.TEXT, file.Label)


class ReadmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.csv')}")
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train"
        )

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[1]
            label = str(int(line[2]))
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for example in examples:
        inputs = tokenizer(
            example.text_a,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label_id = int(example.label)

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, label_id=label_id
            )
        )
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--bert_model", default="bert-base-uncased", required=False)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=float)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    processor = ReadmissionProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length
        )

        train_data = TensorDataset(
            torch.stack([f.input_ids for f in train_features]),
            torch.stack([f.attention_mask for f in train_features]),
            torch.tensor([f.label_id for f in train_features], dtype=torch.long),
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = len(train_dataloader) * args.num_train_epochs
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss}")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
