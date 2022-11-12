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
from transformers import (get_linear_schedule_with_warmup, get_constant_schedule, RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
from mobert_model import MOBERT
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver, gradient_normalizers

cpu_cont = 16
logger = logging.getLogger(__name__)

cpp_map = \
{
"CWE-14": "V",
"CWE-119": "C",
"CWE-120": "B",
"CWE-121": "V",
"CWE-122": "V",
"CWE-123": "B",
"CWE-124": "B",
"CWE-125": "B",
"CWE-126": "V",
"CWE-127": "V",
"CWE-128": "B",
"CWE-129": "V",
"CWE-130": "B",
"CWE-131": "B",
"CWE-134": "B",
"CWE-135": "B",
"CWE-170": "B",
"CWE-188": "B",
"CWE-191": "B",
"CWE-192": "V",
"CWE-194": "V",
"CWE-195": "V",
"CWE-196": "V",
"CWE-197": "B",
"CWE-242": "B",
"CWE-243": "V",
"CWE-244": "V",
"CWE-248": "B",
"CWE-362": "C",
"CWE-364": "B",
"CWE-366": "B",
"CWE-374": "B",
"CWE-375": "B",
"CWE-396": "B",
"CWE-397": "B",
"CWE-401": "V",
"CWE-415": "V",
"CWE-416": "V",
"CWE-457": "V",
"CWE-460": "B",
"CWE-462": "B",
"CWE-463": "B",
"CWE-464": "B",
"CWE-466": "B",
"CWE-467": "V",
"CWE-468": "B",
"CWE-469": "B",
"CWE-476": "B",
"CWE-478": "B",
"CWE-479": "V",
"CWE-480": "B",
"CWE-481": "V",
"CWE-482": "V",
"CWE-483": "B",
"CWE-484": "B",
"CWE-493": "V",
"CWE-495": "V",
"CWE-496": "V",
"CWE-498": "V",
"CWE-500": "V",
"CWE-543": "V",
"CWE-558": "V",
"CWE-562": "B",
"CWE-587": "B",
"CWE-676": "B",
"CWE-690": "CH",
"CWE-704": "C",
"CWE-733": "B",
"CWE-762": "V",
"CWE-766": "V",
"CWE-767": "V",
"CWE-781": "V",
"CWE-782": "V",
"CWE-783": "B",
"CWE-785": "V",
"CWE-787": "B",
"CWE-789": "V",
"CWE-805": "B",
"CWE-806": "V",
"CWE-839": "B",
"CWE-843": "B",
"CWE-910": "B",
"CWE-911": "B",
"CWE-1325": "B",
"CWE-1335": "B",
"CWE-1341": "B",
}

c_map = \
{
"CWE-14": "V",
"CWE-119": "C",
"CWE-120": "B",
"CWE-121": "V",
"CWE-122": "V",
"CWE-123": "B",
"CWE-124": "B",
"CWE-125": "B",
"CWE-126": "V",
"CWE-127": "V",
"CWE-128": "B",
"CWE-129": "V",
"CWE-130": "B",
"CWE-131": "B",
"CWE-134": "B",
"CWE-135": "B",
"CWE-170": "B",
"CWE-188": "B",
"CWE-191": "B",
"CWE-192": "V",
"CWE-194": "V",
"CWE-195": "V",
"CWE-196": "V",
"CWE-197": "B",
"CWE-242": "B",
"CWE-243": "V",
"CWE-244": "V",
"CWE-362": "C",
"CWE-364": "B",
"CWE-366": "B",
"CWE-374": "B",
"CWE-375": "B",
"CWE-401": "V",
"CWE-415": "V",
"CWE-416": "V",
"CWE-457": "V",
"CWE-460": "B",
"CWE-462": "B",
"CWE-463": "B",
"CWE-464": "B",
"CWE-466": "B",
"CWE-467": "V",
"CWE-468": "B",
"CWE-469": "B",
"CWE-474": "B",
"CWE-476": "B",
"CWE-478": "B",
"CWE-479": "V",
"CWE-480": "B",
"CWE-481": "V",
"CWE-482": "V",
"CWE-483": "B",
"CWE-484": "B",
"CWE-495": "V",
"CWE-496": "V",
"CWE-558": "V",
"CWE-560": "V",
"CWE-562": "B",
"CWE-587": "B",
"CWE-676": "B",
"CWE-685": "V",
"CWE-688": "V",
"CWE-689": "CO",
"CWE-690": "CH",
"CWE-704": "C",
"CWE-733": "B",
"CWE-762": "V",
"CWE-781": "V",
"CWE-782": "V",
"CWE-783": "B",
"CWE-785": "V",
"CWE-787": "B",
"CWE-789": "V",
"CWE-805": "B",
"CWE-806": "V",
"CWE-839": "B",
"CWE-843": "B",
"CWE-910": "B",
"CWE-911": "B",
"CWE-1325": "B",
"CWE-1335": "B",
"CWE-1341": "B",
}

additional_map = {'CWE-311': 'C',
                  'CWE-399': 'CAT', 
                  'CWE-94': 'B', 
                  'CWE-264': 'CAT', 
                  'CWE-200': 'C', 
                  'CWE-190': 'B', 
                  'CWE-284': 'P', 
                  'CWE-20': 'C', 
                  'CWE-254': 'CAT', 
                  'CWE-824': 'B', 
                  'CWE-287': 'C', 
                  'CWE-19': 'CAT', 
                  'CWE-732': 'C', 
                  'CWE-369': 'B', 
                  'CWE-189': 'CAT', 
                  'CWE-79': 'B', 
                  'CWE-285': 'C', 
                  'CWE-59': 'B', 
                  'CWE-17': 'DEP', 
                  'CWE-310': 'CAT', 
                  'CWE-772': 'B', 
                  'CWE-835': 'B', 
                  'CWE-400': 'C', 
                  'CWE-77': 'C', 
                  'CWE-18': 'DEP', 
                  'CWE-770': 'B', 
                  'CWE-404': 'C', 
                  'CWE-347': 'B', 
                  'CWE-617': 'B', 
                  'CWE-78': 'B', 
                  'CWE-862': 'C', 
                  'CWE-255': 'CAT', 
                  'CWE-682': 'P', 
                  'CWE-665': 'C', 
                  'CWE-358': 'B', 
                  'CWE-320': 'CAT', 
                  'CWE-345': 'C', 
                  'CWE-269': 'C', 
                  'CWE-674': 'C', 
                  'CWE-693': 'P', 
                  'CWE-295': 'B', 
                  'CWE-90': 'B', 
                  'CWE-834': 'C', 
                  'CWE-388': 'CAT', 
                  'CWE-354': 'B', 
                  'CWE-22': 'B', 
                  'CWE-918': 'B', 
                  'CWE-281': 'B', 
                  'CWE-426': 'B', 
                  'CWE-601': 'B', 
                  'CWE-502': 'B', 
                  'CWE-330': 'C', 
                  'CWE-668': 'C', 
                  'CWE-611': 'B', 
                  'CWE-436': 'C', 
                  'CWE-1021': 'B', 
                  'CWE-346': 'B', 
                  'CWE-754': 'C', 
                  'CWE-909': 'B', 
                  'CWE-74': 'C', 
                  'CWE-89': 'B', 
                  'CWE-290': 'B', 
                  'CWE-494': 'B', 
                  'CWE-361': 'CAT', 
                  'CWE-532': 'B', 
                  'CWE-763': 'B', 
                  'CWE-209': 'B', 
                  'CWE-172': 'C', 
                  'CWE-16': 'CAT', 
                  'CWE-522': 'C', 
                  'CWE-755': 'C', 
                  'CWE-327': 'C', 
                  'CWE-252': 'B', 
                  'CWE-664': 'P', 
                  'CWE-331': 'B', 
                  'CWE-352': 'CO',
                 }

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 cwe_type_label,
                 cwe_id_label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.cwe_type_label = cwe_type_label 
        self.cwe_id_label = cwe_id_label
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, type_label_map, cwe_label_map, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        funcs = df["func_before"].tolist()
        cwe_type_labels = df["type_label"].tolist()
        cwe_id_labels = df["CWE ID"].tolist()
        for i in tqdm(range(len(funcs))):
            cwe_type_label = type_label_map[cwe_type_labels[i]][1]
            cwe_id_label = cwe_label_map[cwe_id_labels[i]][1]
            self.examples.append(convert_examples_to_features(funcs[i], 
                                                              cwe_type_label, 
                                                              cwe_id_label,
                                                              tokenizer, 
                                                              args))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("cwe type label: {}".format(example.cwe_type_label))
                logger.info("cwe id label: {}".format(example.cwe_id_label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].cwe_type_label).float(), torch.tensor(self.examples[i].cwe_id_label).float()


def convert_examples_to_features(func, cwe_type_label, cwe_id_label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-3]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.cls_type_token] + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, cwe_type_label, cwe_id_label)

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
    scheduler = get_constant_schedule(optimizer)

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
    best_loss = 0
    avgcweloss = 0
    avgtypeloss = 0

    model.zero_grad()
    tasks = ["cwe_id", "cwe_type"]
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        cweloss = 0
        typeloss = 0

        for step, batch in enumerate(bar):
            (input_ids, cwe_type_labels, cwe_id_labels) = [x.to(args.device) for x in batch]
            model.train()

            ### MO HERE ###
            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            grads = {}
            scale = {}

            optimizer.zero_grad()
            # representations (z) = encoder_last_hidden_state
            # As an approximate solution, we only need gradients for input
            encoder_last_hidden_state = model(input_ids=input_ids, 
                                                cwe_type_labels=cwe_type_labels,
                                                cwe_id_labels=cwe_id_labels,
                                                cwe_label_map=cwe_label_map,
                                                return_encoder_last_hidden_state=True)
            rep_variable = Variable(encoder_last_hidden_state.data.clone(), requires_grad=True)
            # Compute gradients of each loss function wrt z
            # task = ["cwe_id", "cwe_type"]
            for t in tasks:
                optimizer.zero_grad()
                cwe_id_loss, cwe_type_loss = model(encoder_last_hidden_state=rep_variable,
                                                    input_ids=input_ids,
                                                    cwe_type_labels=cwe_type_labels,
                                                    cwe_id_labels=cwe_id_labels,
                                                    cwe_label_map=cwe_label_map)
                if t == "cwe_id":
                    loss = cwe_id_loss
                elif t == "cwe_type":
                    loss = cwe_type_loss
                loss_data[t] = loss.item()
                loss.backward()
                grads[t] = []
                grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                rep_variable.grad.data.zero_()
            
            # Normalize all gradients, this is optional and not included in the paper.
            # l2 (not work), none, loss, loss+ 
            gn = gradient_normalizers(grads, loss_data, normalization_type="none")
            for t in tasks:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]
            
            # Frank-Wolfe iteration to compute scales.
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
            for i, t in enumerate(tasks):
                scale[t] = float(sol[i])
            
            # Scaled back-propagation
            optimizer.zero_grad()
            rep = model(input_ids=input_ids, 
                        cwe_type_labels=cwe_type_labels,
                        cwe_id_labels=cwe_id_labels,
                        cwe_label_map=cwe_label_map,
                        return_encoder_last_hidden_state=True)
            for i, t in enumerate(tasks):
                cwe_id_loss, cwe_type_loss = model(encoder_last_hidden_state=rep,
                                                    input_ids=input_ids, 
                                                    cwe_type_labels=cwe_type_labels,
                                                    cwe_id_labels=cwe_id_labels,
                                                    cwe_label_map=cwe_label_map)
                if t == "cwe_id": 
                    loss_t = cwe_id_loss
                elif t == "cwe_type":
                    loss_t = cwe_type_loss
                loss_data[t] = loss_t.item()
                if i > 0:
                    total_loss = total_loss + scale[t]*loss_t
                else:
                    total_loss = scale[t]*loss_t
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            ######

            tr_loss += total_loss.item()
            tr_num += 1
            train_loss += total_loss.item()
            cweloss += cwe_id_loss.item()
            typeloss += cwe_type_loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                avgcweloss += cweloss
                avgtypeloss += typeloss
                
            avg_loss = round(train_loss/tr_num, 5)
            avgcweloss = round(cweloss/tr_num, 5)
            avgtypeloss = round(typeloss/tr_num, 5)

            bar.set_description("epoch {} total loss {} cwe loss {} type loss {}".format(idx, avg_loss, avgcweloss, avgtypeloss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True, cwe_label_map=cwe_label_map)    
                    # Save model checkpoint
                    if results['eval_acc'] > best_loss:
                        best_loss = results['eval_acc']
                        logger.info("  " + "*"*20)  
                        logger.info("  Best Acc:%s", round(best_loss,4))
                        logger.info("  " + "*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False, cwe_label_map=None):
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
    cwe_id_preds = []
    cwe_id_trues = []
    for batch in eval_dataloader:
        (input_ids, cwe_type_labels, cwe_id_labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            cwe_id_prob, cwe_type_prob = model(input_ids=input_ids, cwe_label_map=cwe_label_map)
            cwe_id_preds += list((np.argmax(cwe_id_prob.cpu().numpy(), axis=1)))
            cwe_id_trues += list((np.argmax(cwe_id_labels.cpu().numpy(), axis=1)))
    eval_acc = accuracy_score(cwe_id_trues, cwe_id_preds)
    result = {
        "eval_acc": eval_acc
    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    return result

def test(args, model, tokenizer, test_dataset, best_threshold=0.5, cwe_label_map=None, type_label_map=None):
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
    cwe_id_preds = []
    cwe_type_preds = []
    # labels
    cwe_id_trues = []
    cwe_type_trues = []
    for batch in test_dataloader:
        (input_ids, cwe_type_labels, cwe_id_labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            cwe_id_prob, cwe_type_prob = model(input_ids=input_ids, cwe_label_map=cwe_label_map)
            cwe_id_preds += list((np.argmax(cwe_id_prob.cpu().numpy(), axis=1)))
            cwe_id_trues += list((np.argmax(cwe_id_labels.cpu().numpy(), axis=1)))

            cwe_type_preds += list((np.argmax(cwe_type_prob.cpu().numpy(), axis=1)))
            cwe_type_trues += list((np.argmax(cwe_type_labels.cpu().numpy(), axis=1)))

    # CWE ID
    cwe_id_acc = accuracy_score(cwe_id_trues, cwe_id_preds)
    result_id = {"CWE ID Accuracy: ": cwe_id_acc}
    # CWE Types
    cwe_type_acc = accuracy_score(cwe_type_trues, cwe_type_preds)
    result_type = {"CWE Type Accuracy: ": cwe_type_acc}
    # show in console
    logger.info("***** CWE ID Cls. Test results *****")
    for key in sorted(result_id.keys()):
        logger.info("  %s = %s", key, str(round(result_id[key],4)))
    logger.info("***** CWE TYPE Cls Test results *****")
    for key in sorted(result_type.keys()):
        logger.info("  %s = %s", key, str(round(result_type[key],4)))
    # write raw predictions
    test_df = pd.read_csv(args.test_data_file)
    predicted_id = []
    predicted_type = []
    for i in range(len(cwe_id_preds)):
        for k, v in cwe_label_map.items():
            if cwe_id_preds[i] == v[0]:
                predicted_id.append(k)
        for k, v in type_label_map.items():
            if cwe_type_preds[i] == v[0]:
                predicted_type.append(k)

    test_df["cwe_id_prediction"] = predicted_id
    test_df["cwe_type_prediction"] = predicted_type
    
    id_true = test_df["CWE ID"]
    type_true = test_df["type_label"]
    
    correctly_predicted_id = [predicted_id[i] == id_true[i] for i in range(len(id_true))]
    correctly_predicted_type = [predicted_type[i] == type_true[i] for i in range(len(type_true))]

    test_df["correctly_predicted_id"] = correctly_predicted_id
    test_df["correctly_predicted_type"] = correctly_predicted_type
    
    print("CWE ID Accuracy:", round(sum(correctly_predicted_id)/len(correctly_predicted_id), 4))
    print("CWE Type Accuracy:", round(sum(correctly_predicted_type)/len(correctly_predicted_type), 4))
    test_df.to_csv("./raw_predictions/movul_raw_predictions.csv", index=False)

def compute_logit_weight_map(cwe_label_map, type_label_map):
    for cwe_id in cwe_label_map.keys():
        if cwe_id in c_map:
            type_ = c_map[cwe_id]
            idx = type_label_map[type_][0]
            cwe_label_map[cwe_id].append(idx)
        elif cwe_id in cpp_map:
            type_ = cpp_map[cwe_id]
            idx = type_label_map[type_][0]
            cwe_label_map[cwe_id].append(idx)
        elif cwe_id in additional_map:
            type_ = additional_map[cwe_id]
            idx = type_label_map[type_][0]
            cwe_label_map[cwe_id].append(idx)
    return cwe_label_map

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
    args.device = torch.device("cuda:1")
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
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

    """
    # only do once
    # map labels
    df_1 = pd.read_csv("../data/train_wt_type.csv")
    df_2 = pd.read_csv("../data/val_wt_type.csv")
    df_3 = pd.read_csv("../data/test_wt_type.csv")
    df = pd.concat((df_1, df_2, df_3))

    type_labels = list(set(df["type_label"].tolist()))

    type_label_map = {}
    lab = []
    for i in range(len(type_labels)):
        type_label_map[type_labels[i]] = [i]
        lab += [i] 

    tensor_lab = torch.tensor(lab)
    one_hot_lab = F.one_hot(tensor_lab, num_classes=len(tensor_lab))
    one_hot_lab = one_hot_lab.tolist()

    for k, v in type_label_map.items():
        for i in range(len(lab)):
            if v[0] == lab[i]:
                v.append(one_hot_lab[i])
                # counting for logit adj.
                v.append(0)
    with open("./type_label_map.pkl", "wb") as f:
        pickle.dump(type_label_map, f)
    """
    # load cwe label map
    with open("./cwe_label_map.pkl", "rb") as f:
        cwe_label_map = pickle.load(f)
    with open("./type_label_map.pkl", "rb") as f:
        type_label_map = pickle.load(f)
    # logit_weight = cwe_label[cwe id][3]
    cwe_label_map = compute_logit_weight_map(cwe_label_map=cwe_label_map, type_label_map=type_label_map)
    
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<cls_type>"])
    tokenizer.cls_type_token_id = tokenizer.encode("<cls_type>", add_special_tokens=False)[0]
    tokenizer.cls_type_token = "<cls_type>"

    roberta = RobertaModel.from_pretrained(args.model_name_or_path)  
    roberta.resize_token_embeddings(len(tokenizer))

    model = MOBERT(roberta=roberta, 
                        tokenizer=tokenizer, 
                        num_cwe_id=len(cwe_label_map),
                        num_cwe_types=len(type_label_map), 
                        args=args)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, type_label_map, cwe_label_map, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, type_label_map, cwe_label_map, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset, cwe_label_map)
    # Evaluation
    results = {}   
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-acc/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, type_label_map, cwe_label_map, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5, cwe_label_map=cwe_label_map, type_label_map=type_label_map)
    return results

if __name__ == "__main__":
    main()
