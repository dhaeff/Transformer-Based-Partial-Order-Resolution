import copy
import time
import torch
from spacy.util import load_model
from torch import nn, Tensor
import math
import Dataset
import util
from TransformerEncoder import TransformerEncoderModel, LabelSmoothingDistribution
from Dataset import get_loader
from Vocabulary import Vocabulary
from Dataset import build_vocabulary
from Dataset import build_traces
import torch.nn.functional as F
import random
from Logger import Logger
import sys
import numpy as np
import os
from torch.optim import Adam
from CustomLR import CustomLRAdamOptimizer





def train_transformer(transformer: nn. Module, train_dataloader, label_smoothing, criterion, optimizer, epoch, log_interval, num_batches, vocab_size) -> None:
    '''
    Standard training loop for the training phase of the proposed model.
    '''
    transformer.train()  # turn on train mode
    total_loss = 0.
    total_correct_predictions = 0.
    log_interval = log_interval
    start_time = time.time()

    for idx, (input,target) in enumerate(train_dataloader):
        src_seq = input[:,:-1]
        targets = target[:, 1:]
        batch_size = src_seq.size(0)
        src_pad_mask = (src_seq == 0)
        output = transformer(src_seq, None, src_pad_mask)
        output = F.log_softmax(output, dim=-1)
        tgt_seq = label_smoothing(targets.reshape(-1, 1))
        loss = criterion(output.view(-1, vocab_size), tgt_seq)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
        predictions = output.view(-1, vocab_size).max(1).indices.view(targets.shape)
        tgt_pad_mask = (targets == 0)
        predictions[tgt_pad_mask] = 0
        optimizer.step()
        total_loss += loss.item()
        total_correct_predictions += torch.sum((predictions == targets).all(dim=1))
        if epoch == 200:
            pause = 7
        if (idx+1) % log_interval == 0:
            lr = optimizer.get_current_learning_rate()
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            curr_acc = (total_correct_predictions / (log_interval * batch_size)) * 100
            lr_step_no = optimizer.current_step_number
            print(f'| epoch {epoch:3d} | {idx:5d}/{num_batches:5d} batches | '
                  f'lr_step  {lr_step_no} | lr {lr: 2.6f} |acc {curr_acc: 5.3f}%| ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.7f} ')
            total_loss = 0
            total_correct_predictions = 0
            start_time = time.time()



def greedy_encoding(transformer, input, uncertain_subtraces, vocab):
    '''
    The Decoding algorithmn proposed in the Thesis.
    It is used in the evaluation to obtain the predictions of the most probable resolutions per Batch.
    '''
    device = next(transformer.parameters()).device
    max_len_out = input.shape[1]
    uncertain_subtraces_per_trace = torch.from_numpy(np.asarray([len(x[1]) for x in uncertain_subtraces])).to(device)
    input = torch.repeat_interleave(input, uncertain_subtraces_per_trace, dim = 0)
    index_uncertainty = torch.from_numpy(np.asarray([x for y in uncertain_subtraces for x in y[0]]))
    original_index_uncertainty = index_uncertainty
    check = uncertain_subtraces
    uncertain_subtraces = vocab.numericalize([x for y in uncertain_subtraces for x in y[1]])
    length_uncertainty = [len(x) for x in uncertain_subtraces ]
    max_length_uncertainty = max(length_uncertainty)

    first_iter = True

    while True:

        #get prediction for current activity of uncertain subtrace
        input = input.to(device)
        pad_mask = (input==0).to(device)
        output = transformer(input, None, pad_mask)
        output = F.log_softmax(output, dim=-1)
        # Masks all tokens that are not in the uncertain subtraces that the next token is predicted from to ensure the prediction is picked from the viable options
        mask = util.create_res_mask(uncertain_subtraces, len(vocab))
        mask = torch.permute(mask,(1,0))
        mask = mask.reshape(-1,1,len(vocab))
        output = output.masked_fill_((~mask).to(device), float('-inf'))
        most_probable_last_tokens = torch.argmax(output, dim=-1).to(device)
        pos_pred = index_uncertainty.reshape(-1,1).long()
        pos_pred = torch.add(pos_pred, -1).to(device)
        pos_pred[pos_pred >= max_len_out] = max_len_out-1
        most_probable_last_tokens = torch.gather(most_probable_last_tokens, 1, pos_pred).cpu().numpy()

        # Prepare data for next iteration

        # insert prediction in input traces (one the first iteration the token of the aggregated uncertain trace ist removed)
        input,first_iter, index_uncertainty = util.insert_predictions(input, index_uncertainty, most_probable_last_tokens, uncertain_subtraces, first_iter)

        # update remaining uncertain events
        for id, trace in  enumerate(uncertain_subtraces):
            if most_probable_last_tokens[id][0] in trace:
                trace.remove(most_probable_last_tokens[id][0])

        # recombine trace parts and make them ready for comparison with Ground Trurh
        if all(len(l) ==0 for l in uncertain_subtraces):
            output = util.combine_predictions(input, original_index_uncertainty, uncertain_subtraces_per_trace, length_uncertainty, max_len_out)
            break



    return output.to(device)



def evaluate_transformer(transformer, val_dataloader, vocab, vocab_size, criterion,  val_dataset):
    '''
    Evaluation Loop for the proposed Model.
    The validation (or test input) is evaluated on the trained model.
    '''
    transformer.eval()  # turn on evaluation mode
    total_loss = 0.
    total_correct_predictions = 0.
    total_do_nothing = 0.
    bptt = val_dataloader.batch_size
    with torch.no_grad():
        for idx, (input, target, uncertain_subtraces) in enumerate(val_dataloader):
            src_seq = input[:, :]
            batch_size = src_seq.size(0)
            pad_mask = (input == 0)
            output = greedy_encoding(transformer, input, uncertain_subtraces, vocab)
            output = output[:, :-1]
            total_correct_predictions += torch.sum((output== target).all(dim=1))
        print(f"Pred: {output[0]} and {output[1]}, {output[2]}")
        print(f"Tar: {target[0:3]}")
        return None, (total_correct_predictions / (len(val_dataset) - 1)) * 100



'''
Training and evaluation of an earlier approach to training


def train_encoder(model: nn.Module, log_interval, train_dataloader, vocab_size, criterion, optimizer, scheduler, epoch, bptt, num_batches) -> None:


    model.train()  # turn on train mode
    total_loss = 0.
    total_correct_predtargetictions = 0.
    log_interval = log_interval
    start_time = time.time()


    num_batches = num_batches
    for idx, (input,target) in enumerate(train_dataloader):
        data, targets = input, target.reshape(-1)
        batch_size = data.size(0)
        src_mask = TransformerEncoderModel.generate_square_subsequent_mask(len(input[0])).to(device)
        pad_mask = (data == 0)
        output = model(data, None, pad_mask)
        #print(output.shape)
        loss = criterion(output.view(-1, vocab_size), targets)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        #print(target)
        #print(output.view(-1, ntokens).max(1).indices) #(predictions)
        predictions = output.view(-1, vocab_size).max(1).indices

        optimizer.step()
        if epoch == 15 or epoch == 15:
            cewhudb = 1
            pass
        total_loss += loss.item()
        total_correct_predictions += torch.sum((predictions.view(target.shape) == target).all(dim=1))

        if (idx+1) % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            curr_acc = (total_correct_predictions / (log_interval*batch_size))*100
            ppl = math.exp(cur_loss)
            if epoch % 20 == 0:
                x = 0
            print(f'| epoch {epoch:3d} | {idx:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.7f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.7f} | acc {curr_acc: 5.3f}%| ppl {ppl:8.2f}')
            total_loss = 0
            total_correct_predictions = 0
            start_time = time.time()

def evaluate_encoder(model: nn.Module, val_dataloader, val_dataset, vocab_size, criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    total_correct_predictions = 0.
    total_do_nothing = 0.
    bptt = val_dataloader.batch_size
    with torch.no_grad():
        for idx, (input,target) in enumerate(val_dataloader):
            data, targets = input, target.reshape(-1)
            batch_size = data.size(0)
            #if batch_size != bptt:
                #src_mask = TransformerModel.generate_square_subsequent_mask(len(input[0])).to(device)
                #src_mask = src_mask[:batch_size, :batch_size]
            src_mask = TransformerEncoderModel.generate_square_subsequent_mask(len(input[0])).to(device)
            pad_mask = (data == 0)
            output = model(data, None,  pad_mask)
            output_flat = output.view(-1, vocab_size)
            total_loss += batch_size * criterion(output_flat, targets).item()
            predictions = output.view(-1, vocab_size).max(1).indices
            total_correct_predictions += torch.sum((predictions.view(target.shape) == target).all(dim=1))
            total_do_nothing = torch.sum((input == target).all(dim=1))
    return (total_loss / (len(val_dataset) - 1)) , (total_correct_predictions / (len(val_dataset) - 1))*100
'''



def save_checkpoint(model,optimizer, epoch):
    '''
    Saves checkpoint of the model that can be resumed.
    '''

    checkpoint= {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer": optimizer.optimizer.state_dict(),
        "step": optimizer.current_step_number,
    }
    torch.save(checkpoint, f'{sys.stdout.path}best_model.pt')

def encoder_pipeline():

    '''
    The pipline for the training as well as validation an testing of the proposed approach.

    '''


    'Train Data'
    if(real_world_dataset):
        'For real world event logs the data just has to be read in as the uncertainty already exists.'
        uncertainty, certain_traces, uncertain_traces = util.parse_xes(FILE_NAME, data_path, REBUILD_DATA)
        traces = np.concatenate((certain_traces, uncertain_traces), axis = 0)
        vocab_raw = Vocabulary(data_path)
    else:
        'Uncertainty of the event log is created for the synthetic data collections or read from files if REBUILD_DATA is false.'
        traces, vocab_raw = build_traces(FILE_NAME, data_path, REBUILD_DATA)
        uncertainty, certain_traces, uncertain_traces = util.create_uncertainty(data_path, REBUILD_DATA, traces,
                                                                            UNCERTAINTY_TRACES, EIUES)

    'Nested List of all uncertain subtraces. Indexes mazch with indexes of uncertain traces.'
    _, _, uncertain_subtraces = util.uncertain_traces_with_tokens(uncertain_traces, uncertainty, False)

    vocab_raw = build_vocabulary(traces, vocab_raw)

    'Some Stats of the current Dataset'
    max_trace_length = max(len(l) for l in traces)
    num_events = sum(len(l) for l in traces)
    avg_t_length = sum(len(l) for l in traces) / len(traces)
    places = vocab_raw.act_to_index.__len__() - 5
    variants = np.unique(traces)
    events_in_uncertain = sum( [len(a[0]) for a in uncertain_subtraces])

    'Test/Val/ Train Split'
    train_data = certain_traces

    'The proposed making mechanism for the uncertain trcaes is applied'
    uncertain_traces_randomized, uncertain_traces, positions_uncertainty = util.masked_uncertain_traces(uncertainty, uncertain_traces)
    train_data, train_target = util.masked_sampling(train_data)
    train_input = np.concatenate((train_data, uncertain_traces_randomized), axis = 0)
    train_target = np.concatenate((train_target, uncertain_traces_randomized), axis = 0)
    vocab = copy.deepcopy(vocab_raw)
    build_vocabulary(train_input, vocab)
    uncertain_subtraces = [tuple((pos, uncertain_subtraces[id])) for id, pos in enumerate(positions_uncertainty)]

    train_dataloader, train_dataset = get_loader(vocab, train_input, train_target)

    uncertain_traces_all = list(zip(uncertain_traces_randomized, uncertain_traces, uncertain_subtraces))
    val_size = int(0.5 * len(uncertain_traces))
    test_size = len(uncertain_traces) - val_size
    generator = torch.Generator()
    generator.manual_seed(0)
    val_data, test_data = torch.utils.data.random_split(uncertain_traces_all, [val_size, test_size], generator)

    val_input, val_target, val_uncertain_subtraces = zip(*val_data)
    test_input, test_target, test_uncertain_subtraces = zip(*test_data)

    val_dataloader, val_dataset = get_loader(vocab, val_input, val_target, uncertain_subtraces = val_uncertain_subtraces, batch_size=128)
    test_dataloader, test_dataset = get_loader(vocab, test_input, test_target, uncertain_subtraces = test_uncertain_subtraces)

    for x in range(num_searches):

        if (log):
            sys.stdout = Logger(log_path, type, terminal)

        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        bptt = train_dataloader.batch_size

        vocab_size = len(vocab)  # size of vocabulary

        model = TransformerEncoderModel(vocab_size, emsize, nhead, d_hid, nlayers, dropout).to(device)

        if load_model:
            checkpoint = torch.load(path_to_model, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
        criterion = nn.KLDivLoss(reduction='batchmean')

        # Makes smooth target distributions as opposed to conventional one-hot distributions
        label_smoothing = LabelSmoothingDistribution(0.1, vocab.act_to_index["<PAD>"],
                                                     vocab_size, device)
        optimizer = CustomLRAdamOptimizer(
            Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            emsize,
            num_of_warmup_steps = warmup_steps,
            factor = lr_factor
        )
        if load_model:
            optimizer.optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer.current_step_number = checkpoint["step"]

        num_batches = len(train_input) // bptt

        print('-' * 89)
        print(f'Parameter Settings:')
        print(f'EIUES:                            {EIUES}')
        print(f'Ratio uncertain traces:           {UNCERTAINTY_TRACES}')
        print()
        print(f'Type:                             {type}')
        print(f'No. warmup steps:                 {warmup_steps}')
        print(f'LR factor                         {lr_factor} ')
        print(f'Embedding dimension:              {emsize}')
        print(f'Dimension of feedforward network: {d_hid}')
        print(f'Number of transformer layers:     {nlayers}')
        print(f'Number of attentions heads:       {nhead}')
        print(f'Dropout:                          {dropout}')
        print(f'Total number of epochs:           {epochs}')
        print('-' * 89)
        best_val_acc = float('-inf')
        best_epoch = 0

        best_model = None
        if load_model:
            curr_epoch = checkpoint["epoch"]
        else:
            curr_epoch = 0
        for epoch in range(curr_epoch, epochs + 1):
            epoch_start_time = time.time()

            'Loop over epochs. Save the model if the validation loss is the best'
            train_transformer(model, train_dataloader, label_smoothing, criterion, optimizer, epoch, log_interval, num_batches, vocab_size)
            if epoch % 1  == 0:

                val_loss, val_acc = evaluate_transformer(model, val_dataloader, vocab, vocab_size, criterion, val_dataset)

            elapsed = time.time() - epoch_start_time

            if epoch % 1 == 0:
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                      f'valid acc {val_acc:5.3f}%| Best Model from Epoch {best_epoch} with {best_val_acc:5.3f}%')
                print('-' * 89)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    print('-' * 89)
                    print('-' * 89)
                    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                          f'valid acc {val_acc:5.3f}%| ')
                    print('-' * 89)
                    print('-' * 89)
                    best_model = copy.deepcopy(model)
                    save_checkpoint(model, optimizer, epoch)

            if early_stopping > 0:
                if epoch - best_epoch >= early_stopping:
                    break

        '''Evaluate the best model on the test dataset'''

        best_model = copy.deepcopy(model)
        test_loss, test_acc = evaluate_transformer(best_model, test_dataloader, vocab, vocab_size, criterion, test_dataset)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f} | '
              f'test acc {test_acc:5.2f}% | Best Model from Epoch {best_epoch}')
        print('=' * 89)

        import shutil

        sys.stdout.close()
        shutil.move(f'{sys.stdout.path}', f'{sys.stdout.path[:-1]}_{int(test_acc)}/')
        sys.stdout = terminal
######################################################################


if __name__ == '__main__':
    terminal = sys.stdout

    'Variables for Data Preprocessing'
    '''
    Set REBUILD_DATA True only if you are using a new uncertainty setting for a synthtic collection or 
    you are using a dataset you havent used before.
    Otherwise the previous randomly generated uncertainty distribution will be overwritten and created from scratch.
    '''
    REBUILD_DATA =  False

    '''
    Set True if using a real world dataset
    If it is a csv file the parse function call in line 256 has to be changes to parse_csv
    '''
    real_world_dataset = False

    'File to load or suffix of the name of the folder of the stored data'
    FILE_NAME = '1561989897859-21_50'

    'can be set to the type of model'
    type = "encoder_GREEDY"

    '''
    EIUES = Events in uncertain event sets
    Set to desired value if rebuilding data otherwise to value that the dataset you want to load has
    '''
    EIUES = 0.3

    '''
    Ratio of uncertain traces
    Set to desired value if rebuilding data otherwise to value that the dataset you want to load has
    '''
    UNCERTAINTY_TRACES = 0.995


    if(real_world_dataset):
        'These are only set to 0 to avoid confusion the real values have to be determined from the datasets.'
        EIUES = 0
        UNCERTAINTY_TRACES = 0

    'Only set false if you dont want a log file to be created'
    log = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    'Make rep. to store training data according to params'
    data_path = f"Data/{FILE_NAME}_EIUES_{EIUES * 100: 5.2f}_UT_{UNCERTAINTY_TRACES * 100: 5.2f}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    log_path = f"{FILE_NAME}_EIUES_{EIUES * 100: 5.2f}_UT_{UNCERTAINTY_TRACES * 100: 5.2f}"

    'Total number of epochs'
    epochs = 150

    'Sets how often accuracy scores are printed during training for training data'
    log_interval = 25

    'Set True if you want to do a hyperparameter search and adjust settings for the search accordingly below'
    search_hyperparam = True


    'Set True if you want to load an already existing model and set its path below'
    load_model = False
    if load_model:
        '''
        The hyperparameters have to be set to the same values as in the according Log.log file.
        The Log.log file is stored in the same folder as the best_model.
        
        NOTE: Some checkpoints dont yet have the attribute step 
        if you encounter this set the value manually in line 335
        to the lr_step value the model had after its best epoch
        this value can also be found in the Log.log file
        '''
        path_to_model = 'Experiments_Final/1561989897859-21_50_EIUES_ 30.00_UT_ 99.50/03_26_160454_encoder_GREEDY_65/best_model.pt'

    if search_hyperparam:
        num_searches = 15 #number of random hyperparameter settings that will be trained
        early_stopping = 15 #number of epoach after which the training will stop if accuracy doesn't improve
        emsize = random.choice([64,128,256])  # embedding dimension
        d_hid = random.choice([1024,2048])  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = random.choice(list(range(1,4)))  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead =  random.choice([4,8,16]) # number of heads in nn.MultiheadAttention
        dropout = random.uniform(0.1, 0.35)  # dropout probability
        warmup_steps =random.uniform(4000, 8000) #warmup steps of learning rate scheduler
        lr_factor = random.uniform(0.1,0.5) # factor applied to lr
    else:
        num_searches = 1
        early_stopping = 15
        emsize = 64   # embedding dimensio
        d_hid = 1024   # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 1 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 4  # number of heads in nn.MultiheadAttention
        dropout = 0.622 #dropout rate
        warmup_steps = 8313 #number of warmup steps (learing rate)
        lr_factor = 0.79 # factor applied to lr

    encoder_pipeline()



