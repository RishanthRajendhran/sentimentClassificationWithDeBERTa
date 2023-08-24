import argparse
import transformers
from transformers import AutoModel, AutoTokenizer 
import numpy as np
import torch
import logging
from pathlib import Path
from os.path import exists
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import csv, json
import evaluate
from datasets import Dataset

labelToModelLogitIndex = {
    "Negative": 0,
    "Positive": 1, 
}

colsToRemove = {
    "imdb": [
        "text"
    ]
}

labelTag = {
    "imdb": "label"
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-cacheDir",
    help="Path to cache location for Huggingface",
    default="/scratch/general/vast/u1419542/huggingface_cache/"
)

parser.add_argument(
    "-dataset",
    choices = [
        "imdb",
    ],
    default="imdb",
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=1
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=16
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for optimizer",
    default=2e-5
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0.01
)

parser.add_argument(
    "-model",
    help="Path to model to use",
    default="microsoft/deberta-v3-large"
)

parser.add_argument(
    "-out",
    "--output_dir",
    help="Path to output directory where trained model is to be saved",
    required=True
)
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        raise ValueError("Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#---------------------------------------------------------------------------
class ComputeMetrics:
        def __init__(self, metricName="accuracy"):
            self.metricName = metricName
            self.metric = evaluate.load(metricName)
        
        def __call__(self, evalPreds):
            predictions, labels = evalPreds
            predictions = np.argmax(predictions, axis=1)
            return self.metric.compute(predictions=predictions, references=labels)
#---------------------------------------------------------------------------
class Tokenize:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer 
        self.dataset = dataset

    def __call__(self, example):
        # return self.tokenizer(inputToPrompt(example, self.dataset), truncation=True)
        return self.tokenizer(example["text"], truncation=True)
#---------------------------------------------------------------------------
def inputToPrompt(instance, dataset):
    if dataset == "imdb":
        inpPrompt = "Review: {review}\nWhat is the sentiment of the review: negative or positive?".format(
            review=instance["text"]
        )
    else: 
        raise ValueError("[inputToPrompt] {} not supported!".format(dataset))
    return inpPrompt
#---------------------------------------------------------------------------
def writeFile(data, fileName):
    if fileName.endswith(".csv"):
        with open(fileName, 'w', newline='') as f:
            writer = csv.DictWriter(f, data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    elif fileName.endswith(".json"):
        with open(fileName, "w") as f: 
            json.dump(data, f)
    elif fileName.endswith(".jsonl"):
        with open(fileName, "w") as f: 
            for instance in data:
                f.write(json.dumps(instance))
                f.write("\n")
    else: 
        raise ValueError("[readFile] {} has unrecognized file extension!".format(fileName))
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        # logging.basicConfig(filemode='w', level=logging.ERROR)
        logging.basicConfig(filemode='w', level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("Using GPU: cuda")
        device = "cuda"
    else: 
        logging.info("Using CPU")
        device = "cpu"

    if args.batchSize <= 0:
        raise ValueError("[main] Batch Size has to be a positive number!")
    
    data = load_dataset(args.dataset, cache_dir=args.cacheDir)
    if "train" not in data.keys():
        raise RuntimeError("[main] No train split found in {} dataset!".format(args.dataset))
    if "test" not in data.keys():
        raise RuntimeError("[main] No test split found in {} dataset!".format(args.dataset))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labelToModelLogitIndex))
    model.to(device)

    tokenizedDatasets = data.map(Tokenize(tokenizer, args.dataset), batched=True, remove_columns=colsToRemove[args.dataset])
    tokenizedDatasets = tokenizedDatasets.rename_column(labelTag[args.dataset], "labels")
    dataCollator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainingArgs = TrainingArguments(
        output_dir=args.output_dir, 
        num_train_epochs=args.numEpochs,
        learning_rate=args.learningRate,
        weight_decay=args.weightDecay,
        per_device_train_batch_size=args.batchSize,
        per_device_eval_batch_size=args.batchSize,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        bf16=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model,
        trainingArgs,
        train_dataset=tokenizedDatasets["train"],
        eval_dataset=tokenizedDatasets["test"],
        data_collator=dataCollator,
        tokenizer=tokenizer,
        compute_metrics=ComputeMetrics("accuracy")
    )

    #Train the model
    trainer.train()

    #Sample 10 mispredictions randomly
    predictions = trainer.predict(tokenizedDatasets["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    incorrectInds = np.where(~np.equal(preds, tokenizedDatasets["test"]["labels"]))[0]
    assert len(incorrectInds) >= 10
    testData = data["test"]
    testData = testData.add_column("predicted", preds)
    if args.dataset == "imdb":
        testData = testData.rename_column("text", "review")
    allData = Dataset.from_dict(testData[incorrectInds])
    sampledData = Dataset.from_dict(testData[np.random.choice(incorrectInds, 10, replace=False)])

    allData.to_json("mispredictions.jsonl", orient="records", lines=True)
    sampledData.to_json("mispredictions_10.jsonl", orient="records", lines=True)

#---------------------------------------------------------------------------
if __name__ == "__main__":
    main()