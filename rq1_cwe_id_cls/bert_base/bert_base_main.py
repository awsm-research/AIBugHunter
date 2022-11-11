from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
from bert_base_model import BERT
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 cwe_type_label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.cwe_type_label = cwe_type_label 
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, cwe_label_map, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        funcs = df["func_before"].tolist()
        cwe_type_labels = df["CWE ID"].tolist()
        severity_labels = df["Score"].tolist()
        for i in tqdm(range(len(funcs))):
            cwe_type_label = cwe_label_map[cwe_type_labels[i]][1]
            # count label freq if it's training data
            if file_type == "train":
                cwe_label_map[cwe_type_labels[i]][2] += 1
            self.examples.append(convert_examples_to_features(funcs[i], 
                                                              cwe_type_label, 
                                                              tokenizer, 
                                                              args))
        if file_type == "train":
            self.cwe_label_map = cwe_label_map
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("cwe type label: {}".format(example.cwe_type_label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].cwe_type_label).float()


def convert_examples_to_features(func, cwe_type_label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, cwe_type_label)

def compute_adjustment(tau, args, cwe_label_map):
    """compute the base probabilities"""
    freq = []
    for i in range(len(cwe_label_map)):
        for k, v in cwe_label_map.items():
            if v[0] == i:
                freq.append(v[2])
    label_freq_array = np.array(freq)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tau + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset, cwe_label_map):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    if args.use_logit_adjustment:
        logit_adjustment = compute_adjustment(tau=args.tau, args=args, cwe_label_map=cwe_label_map)
    else:
        logit_adjustment = None

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100000

    model.zero_grad()
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, cwe_type_labels) = [x.to(args.device) for x in batch]
            
            model.train()
            cwe_type_loss = model(input_ids=inputs_ids, 
                                  cwe_type_labels=cwe_type_labels)
            if args.n_gpu > 1:
                cwe_type_loss = cwe_type_loss.mean()
            if args.gradient_accumulation_steps > 1:
                cwe_type_loss = cwe_type_loss / args.gradient_accumulation_steps

            cwe_type_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += cwe_type_loss.item()
            tr_num += 1
            train_loss += cwe_type_loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} cwe type loss {}".format(idx, avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if results['eval_total_loss'] < best_loss:
                        best_loss = results['eval_total_loss']
                        logger.info("  " + "*"*20)  
                        logger.info("  Best Total Loss:%s", round(best_loss,4))
                        logger.info("  " + "*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    for batch in eval_dataloader:
        (inputs_ids, cwe_type_labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            ce_loss = model(input_ids=inputs_ids, 
                             cwe_type_labels=cwe_type_labels)
    result = {
        "eval_total_loss": ce_loss.item()
    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    return result

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    # preds
    cwe_type_preds = []
    severity_preds = []
    # labels
    cwe_type_trues = []
    severity_trues = []
    for batch in test_dataloader:
        (inputs_ids, cwe_type_labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            cwe_type_prob = model(input_ids=inputs_ids)
            cwe_type_preds += list((np.argmax(cwe_type_prob.cpu().numpy(), axis=1)))
            cwe_type_trues += list((np.argmax(cwe_type_labels.cpu().numpy(), axis=1)))
    # CWE Types
    cwe_type_acc = accuracy_score(cwe_type_trues, cwe_type_preds)
    result_cwe = {"CWE Type Accuracy: ": cwe_type_acc}
    # show in console
    logger.info("***** CWE Type Cls. Test results *****")
    for key in sorted(result_cwe.keys()):
        logger.info("  %s = %s", key, str(round(result_cwe[key],4)))


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--use_logit_adjustment", action='store_true', default=False,
                        help="Whether to use logit adjustment")
    parser.add_argument("--tau", default=1.2, type=float,
                        help="tau for logit adj.")
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

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
    args = parser.parse_args()
    # Setup CUDA, GPU
    args.n_gpu = 1
    args.device = torch.device("cuda:0")
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu,)
    # Set seed
    set_seed(args) 

    """# only do once
    # map labels
    df_1 = pd.read_csv("../data/train.csv")
    df_2 = pd.read_csv("../data/val.csv")
    df_3 = pd.read_csv("../data/test.csv")
    df = pd.concat((df_1, df_2, df_3))

    cwe_labels = list(set(df["CWE ID"].tolist()))

    cwe_label_map = {}
    lab = []
    for i in range(len(cwe_labels)):
        cwe_label_map[cwe_labels[i]] = [i]
        lab += [i] 

    tensor_lab = torch.tensor(lab)
    one_hot_lab = F.one_hot(tensor_lab, num_classes=len(tensor_lab))
    one_hot_lab = one_hot_lab.tolist()

    for k, v in cwe_label_map.items():
        for i in range(len(lab)):
            if v[0] == lab[i]:
                v.append(one_hot_lab[i])
                # counting for logit adj.
                v.append(0)
    with open("./cwe_label_map.pkl", "wb") as f:
        pickle.dump(cwe_label_map, f)
    """
    # load cwe label map
    with open("./cwe_label_map.pkl", "rb") as f:
        cwe_label_map = pickle.load(f)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    roberta = RobertaModel.from_pretrained(args.model_name_or_path)  
    model = BERT(roberta=roberta, 
                        tokenizer=tokenizer, 
                        num_cwe_types=len(cwe_label_map), 
                        args=args)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, cwe_label_map, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, cwe_label_map, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset, train_dataset.cwe_label_map)
    # Evaluation
    results = {}   
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, cwe_label_map, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()
