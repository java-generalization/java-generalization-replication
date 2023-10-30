# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, 
                          T5ForConditionalGeneration, RobertaTokenizer)
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import sys

cpu_cont = 16
logger = logging.getLogger(__name__)

sampled_lines=None

def read_file(f):
    file = open(f, 'r')
    lines = file.readlines()
    lines=[l.strip() for l in lines]
    file.close()

    return lines

def read_dataset(dataset_folder, dataset_set):

    # dataset_set in ["training", "eval", "test"]

    set_input=read_file(os.path.join(dataset_folder, "{}_masked_code".format(dataset_set)))
    set_target=read_file(os.path.join(dataset_folder, "{}_mask".format(dataset_set)))

    set_input=[t.replace("<x>", "<extra_id_0>") for t in set_input]
    set_target=[t.replace("<z>", "") for t in set_target]

    dict_ = {'source': set_input, 'target': set_target} 
    set_data = pd.DataFrame(dict_)

    return set_data

    
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 label,
                 decoder_input_ids,
                 type_masking):
        self.input_ids = input_ids
        self.label=label
        self.decoder_input_ids = decoder_input_ids
        self.type_masking=type_masking        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, train_data=None, val_data=None, test_data=None, file_type="train"):
        if file_type == "train":
            sources = train_data["source"].tolist()
            labels = train_data["target"].tolist()
        elif file_type == "eval":
            sources = val_data["source"].tolist()
            labels = val_data["target"].tolist()
        elif file_type == "test":
            sources = test_data["source"].tolist()
            labels = test_data["target"].tolist()
        self.examples = []
        # for i in tqdm(range(len(sources))):
        for i in tqdm(range(len(sources))): #edit
            self.examples.append(convert_examples_to_features(sources[i], labels[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[i].decoder_input_ids, self.examples[i].type_masking


def convert_examples_to_features(source, label, tokenizer, args):
    # encode - subword tokenize
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    
    type_masking="token"
    if source.startswith("CONSTRUCT:"):
        type_masking="construct"
    elif source.startswith("BLOCK:"):
        type_masking="block"

    return InputFeatures(source_ids, label, decoder_input_ids, type_masking)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.replace("<extra_id_0>", "")
    tokens = tokens.replace("<extra_id_1>", "")
    tokens = tokens.replace("\\n", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num,5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, tokenizer, test_dataset, checkpoint, best_threshold=0.5, output_folder=""):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate

    lines_to_write=list()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    targets= []
    masking_scenario=[]
    correct_prediction = ""
    results_masking=dict()
    results_masking["token"]=list()
    results_masking["construct"]=list()
    results_masking["block"]=list()
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:

        num_return_sequences=1

        (input_ids, attention_mask, labels, decoder_input_ids)=[x.squeeze(1).to(args.device) for x in batch[:-1]]

        type_masking=list(batch[-1])
        with torch.no_grad():
            beam_outputs = model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          do_sample=True, # disable sampling to test if batching affects output
                                          num_beams=args.num_beams,
                                          num_return_sequences=num_return_sequences,
                                          max_length=args.decoder_block_size)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()

        predictions_list = [beam_outputs[x:x+num_return_sequences] for x in range(0, len(beam_outputs), num_return_sequences)]

        for i, curr_prediction in enumerate(predictions_list):

            correct_pred=False

            for single_prediction in curr_prediction:
                # pred
                prediction = tokenizer.decode(single_prediction, skip_special_tokens=True)
                prediction = clean_tokens(prediction)

                # truth
                ground_truth = tokenizer.decode(decoder_input_ids[i], skip_special_tokens=True)
                ground_truth = clean_tokens(ground_truth)

                # print(prediction, ground_truth)

                if prediction.replace(" ","") == ground_truth.replace(" ",""):
                    correct_prediction = prediction
                    correct_pred = True
                    # print("CORRECT")
                    break
            if correct_pred:
                raw_predictions.append(correct_prediction)
                targets.append(ground_truth)
                masking_scenario.append(type_masking[i])
                results_masking[type_masking[i]].append(1)
                accuracy.append(1)

            else:
                # if not correct, use the first output in the beam as the raw prediction
                raw_pred = tokenizer.decode(curr_prediction[0], skip_special_tokens=True)
                raw_pred = clean_tokens(raw_pred)
                raw_predictions.append(raw_pred)
                targets.append(ground_truth)

                accuracy.append(0)
                masking_scenario.append(type_masking[i])
                results_masking[type_masking[i]].append(0)
            # print("______")


    print(len(masking_scenario))
    print(len(raw_predictions))
    print(len(targets))
    print(len(accuracy))

    df_perf=pd.DataFrame({"scenario": [], "accuracy": [], "samples": []})

    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    print("ABCD ***** Eval results *****")
    print(f"ABCD Global Eval Accuracy: {str(test_result)}")

    lines_to_write.append("***** Eval results *****")
    lines_to_write.append(f"Global Eval Accuracy: {str(test_result)}")

    df_perf = pd.DataFrame({"scenario": [], "accuracy": [], "samples": []})

    scenarios_perf=list()
    accuracy_perf=list()
    samples_perf=list()


    scenarios_perf.append("global")
    accuracy_perf.append(test_result)

    tot=0
    for k in results_masking.keys():
        tot+=len(results_masking[k])

    samples_perf.append(tot)

    for k in results_masking.keys():

        test_result = round(sum(results_masking[k]) / len(results_masking[k]), 4)
        print("ABCD {} {} samples".format(k, len(results_masking[k])))

        print("ABCD ***** Eval results *****")
        print(f"ABCD Eval Accuracy: {str(test_result)}")

        scenarios_perf.append(k)
        accuracy_perf.append(test_result)
        samples_perf.append(len(results_masking[k]))


        lines_to_write.append("{} {} samples".format(k, len(results_masking[k])))
        lines_to_write.append("***** Eval results *****")
        lines_to_write.append(f"ABCD Eval Accuracy: {str(test_result)}")


    df_perf["scenario"] = scenarios_perf
    df_perf["accuracy"] = accuracy_perf
    df_perf["samples"] = samples_perf

    file_name=os.path.join(output_folder, "{}_summary.csv".format(str(checkpoint)))

    df_perf.to_csv(file_name)


    # write prediction to file
    df = pd.DataFrame({"scenario": [], "raw_predictions": [], "targets": [], "correctly_predicted": []})
    df["scenario"] = masking_scenario
    df["raw_predictions"] = raw_predictions
    df["targets"] = targets
    df["correctly_predicted"] = accuracy

    file_name=os.path.join(output_folder, "{}.csv".format(str(checkpoint)))


    df.to_csv(file_name)

    file_name=os.path.join(output_folder, "{}_perfect.csv".format(str(checkpoint)))


    with open(file_name, 'w+') as file:
        for l in lines_to_write:
            file.write('{}\n'.format(l))

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--output_csv", default=None, type=str, required=False,
                        help="csv file name")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--dataset_folder', type=str, default="dataset",
                        help="dataset folder containing train_mask, train_masked_code, ...")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="checkpoints to evaluate, comma separated")
    parser.add_argument('--checkpoint_folder', type=str, default="",
                        help="checkpoints to evaluate (only the number of the step), comma separated")

    args = parser.parse_args()

    versions=[2, 11,  14,  16,  17, 5,  6,  7]

    versions=[str(v) for v in versions]

    for v in versions:

        print(f"ABCD: TRAINING {v}")

        checkpoint_folder=os.path.join(args.checkpoint_folder,v)
        dataset_folder=os.path.join(args.dataset_folder,v)

        output_folder_temp=args.output_csv.replace("predictions_CHECKPOINT.csv","")


        output_folder=os.path.join(output_folder_temp,v)

        print(checkpoint_folder)
        print(dataset_folder)
        print(output_folder)

        # Setup CUDA, GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1
        args.device = device

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
        logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
        # Set seed
        set_seed(args)

        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        #tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
       
        checkpoints=os.listdir(checkpoint_folder)

        checkpoints=[int(c.replace("checkpoint-epoch-","")) for c in checkpoints if "epoch" in c]

        checkpoints.sort()

        checkpoints=["checkpoint-epoch-{}".format(str(c)) for c in checkpoints]

        print(checkpoints)

        # load the standard base model than we load the weights from the checkpoint
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base") 
        #model.resize_token_embeddings(len(tokenizer))

        logger.info("Training/evaluation parameters %s", args)
        
        # Evaluation
        results = {}  

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if args.do_test:

            test_data=read_dataset(dataset_folder, "eval")
            test_dataset = TextDataset(tokenizer, args, None, None, test_data, file_type='test')

            for checkpoint in checkpoints:
                print("ABCD EVALUATING {}".format(checkpoint))

                output_dir = os.path.join(checkpoint_folder, checkpoint, "pytorch_model.bin")
                print("Testing model {}".format(output_dir))
                # strict = False to ignore extra parameters
                model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
                model.to(args.device)

                test(args, model, tokenizer, test_dataset, checkpoint, best_threshold=0.5, output_folder=output_folder)

if __name__ == "__main__":
    main()
