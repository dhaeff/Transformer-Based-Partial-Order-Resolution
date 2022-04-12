import copy
import time
import torch
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

    transformer.train()  # turn on train mode
    total_loss = 0.
    total_correct_predictions = 0.
    log_interval = log_interval
    start_time = time.time()

    for idx, (input,target) in enumerate(train_dataloader):
        src_seq = input[:,:-1]
        targets = target[:, 1:]
        batch_size = src_seq.size(0)
        #tgt_mask = TransformerEncoderModel.generate_square_subsequent_mask(len(tgt_in_seq[0])).to(device)
        #tgt_pad_mask = (tgt_in_seq == 0)
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
            #print(f"Pred: {predictions[0]}")
            #print(f"Tar: {targets[0]}")
            total_loss = 0
            total_correct_predictions = 0
            start_time = time.time()


def greedy_encoding(transformer, input, uncertain_subtraces, vocab):
    device = next(transformer.parameters()).device
    max_len_out = input.shape[1]
    uncertain_subtraces_per_trace = torch.from_numpy(np.asarray([len(x[1]) for x in uncertain_subtraces])).to(device)
    input = torch.repeat_interleave(input, uncertain_subtraces_per_trace, dim = 0)
    index_uncertainty = torch.from_numpy(np.asarray([x for y in uncertain_subtraces for x in y[0]]))
    #index_uncertainty = list(x[0] for x in uncertain_subtraces)
    original_index_uncertainty = index_uncertainty
    check = uncertain_subtraces
    uncertain_subtraces = vocab.numericalize([x for y in uncertain_subtraces for x in y[1]])
    #uncertain_subtraces = vocab.numericalize(np.concatenate(x[1]) for x in uncertain_subtraces)
    length_uncertainty = [len(x) for x in uncertain_subtraces ]
    max_length_uncertainty = max(length_uncertainty)

    first_iter = True

    while True:

        #get prediction for current activity of uncertain subtrace
        input = input.to(device)
        pad_mask = (input==0).to(device)
        output = transformer(input, None, pad_mask)
        output = F.log_softmax(output, dim=-1)
        mask = util.create_res_mask(uncertain_subtraces, len(vocab))
        mask = torch.permute(mask,(1,0))
        mask = mask.reshape(-1,1,len(vocab))
        output2 = output
        #mask.to(device)
        output = output.masked_fill_((~mask).to(device), float('-inf'))
        most_probable_last_tokens = torch.argmax(output, dim=-1).to(device)
        most_probable_last_tokens2 = most_probable_last_tokens
        pos_pred = index_uncertainty.reshape(-1,1).long()
        #if first_iter:
        pos_pred = torch.add(pos_pred, -1).to(device)
        pos_pred[pos_pred >= max_len_out] = max_len_out-1
        most_probable_last_tokens = torch.gather(most_probable_last_tokens, 1, pos_pred).cpu().numpy()

        #Prepare data for next iteration

        #insert prediction in input traces (one the first iteration the token of the aggregated uncertain trace ist removed)
        input,first_iter, index_uncertainty = util.insert_predictions(input, index_uncertainty, most_probable_last_tokens, uncertain_subtraces, first_iter)
        #update remaining uncertain events
        for id, trace in  enumerate(uncertain_subtraces):
            if most_probable_last_tokens[id][0] in trace:
                trace.remove(most_probable_last_tokens[id][0])

        # recombine trace parts and make them ready for comparison with Ground Trurh
        if all(len(l) ==0 for l in uncertain_subtraces):
            output = util.combine_predictions(input, original_index_uncertainty, uncertain_subtraces_per_trace, length_uncertainty, max_len_out)
            break



    return output.to(device)


def evaluate_transformer(transformer, val_dataloader, vocab, vocab_size, criterion,  val_dataset):
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
            #output2 = transformer(input, None, pad_mask)
            #output2 = F.log_softmax(output, dim=-1)

            #output2 = output2.view(-1, vocab_size).max(1).indices.view(target.shape)
            #output_flat = output.view(-1, vocab_size)
            #total_loss += batch_size * criterion(output_flat, target.reshape(-1)).item()
            #predictions = output.view(-1, vocab_size).max(1).indices

            output = greedy_encoding(transformer, input, uncertain_subtraces, vocab)
            output = output[:, :-1]
            total_correct_predictions += torch.sum((output== target).all(dim=1))
            #total_do_nothing = torch.sum((input == target).all(dim=1))
        print(f"Pred: {output[0]} and {output[1]}, {output[2]}")
        print(f"Tar: {target[0:3]}")
        return 69, (total_correct_predictions / (len(val_dataset) - 1)) * 100


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


def save_checkpoint(model,optimizer, epoch):

    checkpoint= {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer": optimizer.optimizer.state_dict(),
        "step": optimizer.current_step_number,
    }
    torch.save(checkpoint, f'{sys.stdout.path}best_model.pt')

def encoder_pipeline():

    'Train Data'
    if(real_world_dataset):
        #uncertainty, certain_traces, uncertain_traces = util.parse_csv(FILE_NAME, data_path, REBUILD_DATA)
        uncertainty, certain_traces, uncertain_traces = util.parse_xes(FILE_NAME, data_path, REBUILD_DATA)
        traces = np.concatenate((certain_traces, uncertain_traces), axis = 0)
        vocab_raw = Vocabulary(data_path)
    else:
        traces, vocab_raw = build_traces(FILE_NAME, data_path, REBUILD_DATA)
        uncertainty, certain_traces, uncertain_traces = util.create_uncertainty(data_path, REBUILD_DATA, traces,
                                                                            UNCERTAINTY_TRACES, EIUES)
    vocab_raw = build_vocabulary(traces, vocab_raw)
    max_trace_length = max(len(l) for l in traces)
    num_events = sum(len(l) for l in traces)
    avg_t_length = sum(len(l) for l in traces) / len(traces)
    places = vocab_raw.act_to_index.__len__() - 5
    variants = np.unique(traces)
    'randomize order of uncertain events'
    if with_startend:
        # randomized = util.randomize_uncertain_events(data_path, REBUILD_DATA,uncertain_traces, uncertainty)
        _, _, uncertain_subtraces = util.uncertain_traces_with_tokens(
            uncertain_traces, uncertainty, training_with_samples)
        #uncertain_traces_randomized, uncertain_traces, uncertain_subtraces = util.uncertain_traces_with_tokens(
            #uncertain_traces, uncertainty, training_with_samples)
    else:
        uncertain_traces_randomized = util.randomize_uncertain_events(data_path, REBUILD_DATA, uncertain_traces,
                                                                      uncertainty)

    events_in_uncertain = sum( [len(a[0]) for a in uncertain_subtraces])
    'Test/Val/ Train Split'
    train_data = certain_traces
    if with_startend and training_with_samples:
        train_input, train_target = util.negative_sampling_with_tokens(certain_traces, 10, uncertain_subtraces,
                                                                       uncertain_traces_randomized)
        vocab = build_vocabulary(train_input + uncertain_traces_randomized, vocab_raw)
    elif not training_with_samples:
        #util.add_sot(train_data)
        uncertain_traces_randomized, uncertain_traces, positions_uncertainty = util.masked_uncertain_traces(uncertainty, uncertain_traces)
        train_data, train_target = util.masked_negative_sampling(train_data)
        #train_target = train_data.copy()
        train_input = np.concatenate((train_data, uncertain_traces_randomized), axis = 0)
        train_target = np.concatenate((train_target, uncertain_traces_randomized), axis = 0)
        vocab = copy.deepcopy(vocab_raw)
        build_vocabulary(train_input, vocab)
        uncertain_subtraces = [tuple((pos, uncertain_subtraces[id])) for id, pos in enumerate(positions_uncertainty)]
        #uncertain_subtraces = util.add_position_uncertain_tokens(uncertain_traces_randomized, vocab_raw,
                                                                # uncertain_subtraces)
    else:
        vocab = build_vocabulary(traces, vocab_raw)
        train_input, train_target = util.negative_sampling_ctraces(train_data, 2)
    train_dataloader, train_dataset = get_loader(vocab, train_input, train_target)

    uncertain_traces_all = list(zip(uncertain_traces_randomized, uncertain_traces, uncertain_subtraces))
    val_size = int(0.5 * len(uncertain_traces))
    test_size = len(uncertain_traces) - val_size
    generator = torch.Generator()
    generator.manual_seed(0)
    val_data, test_data = torch.utils.data.random_split(uncertain_traces_all, [val_size, test_size], generator)

    val_input, val_target, val_uncertain_subtraces = zip(*val_data)
    test_input, test_target, test_uncertain_subtraces = zip(*test_data)
    if training_with_samples:
        val_dataloader, val_dataset = get_loader(vocab, val_input, val_target, batch_size=128)
        test_dataloader, test_dataset = get_loader(vocab, test_input, test_target)
    else:
        val_dataloader, val_dataset = get_loader(vocab, val_input, val_target, uncertain_subtraces = val_uncertain_subtraces, batch_size=128)
        test_dataloader, test_dataset = get_loader(vocab, test_input, test_target, uncertain_subtraces = test_uncertain_subtraces)

    for x in range(10):

        if (log):
            sys.stdout = Logger(log_path, type, terminal)
            # sys.stdout.terminal = sys.stdout

        'Variables for Model'
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        bptt = train_dataloader.batch_size

        vocab_size = len(vocab)  # size of vocabulary
        emsize = 64#random.choice([64,128,256])  # embedding dimension
        d_hid = 1024#random.choice([1024,2048])  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = random.choice(list(range(1,4)))  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead =  8#random.choice([4,8,16]) # number of heads in nn.MultiheadAttention
        dropout = 0.404889#random.uniform(0.1, 0.35)  # dropout probability
        model = TransformerEncoderModel(vocab_size, emsize, nhead, d_hid, nlayers, dropout).to(device)
        #checkpoint = torch.load("Experiments_Final/1561989897100_0_50_EIUES_ 25.00_UT_ 20.00/03_29_130753_encoder_GREEDY_82/best_model.pt")
        #model.load_state_dict(checkpoint["model_state"])
        if training_with_samples:
            criterion = nn.CrossEntropyLoss()
            lr = 0.000151766558  # random.uniform(0.000005, 0.0008) # learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.97)
        else:
            criterion = nn.KLDivLoss(reduction='batchmean')

            # Makes smooth target distributions as opposed to conventional one-hot distributions
            # My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
            label_smoothing = LabelSmoothingDistribution(0.1, vocab.act_to_index["<PAD>"],
                                                         vocab_size, device)
            warmup_steps =random.uniform(4000, 8000)
            factor = random.uniform(0.1,0.5)
            # Check out playground.py for an intuitive visualization of how the LR changes with time/training steps, easy stuff.
            optimizer = CustomLRAdamOptimizer(
                Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
                emsize,
                num_of_warmup_steps = warmup_steps,
                factor = factor
            )
        #optimizer.optimizer.load_state_dict(checkpoint["optimizer"])
        #optimizer.current_step_number = checkpoint["step"]
        epochs = 150
        num_batches = len(train_input) // bptt
        log_interval = 25
        print('-' * 89)
        print(f'Parameter Settings:')
        print(f'EIUES:                            {EIUES}')
        print(f'Ratio uncertain traces:           {UNCERTAINTY_TRACES}')
        print()
        print(f'Type:                             {type}')
        print(f'No. warmup steps:                 {warmup_steps}')
        print(f'LR factor                         {factor} ')
        print(f'Embedding dimension:              {emsize}')
        print(f'Dimension of feedforward network: {d_hid}')
        print(f'Number of transformer layers:     {nlayers}')
        print(f'Number of attentions heads:       {nhead}')
        print(f'Dropout:                          {dropout}')
        #print(f'Starting lr:                      {lr}')
        print(f'Total number of epochs:           {epochs}')
        print('-' * 89)
        best_val_acc = float('-inf')
        best_epoch = 0
        best_model = None

        # best_val_acc = float('-inf')
        best_model = None
        curr_epoch = 0#checkpoint["epoch"]
        for epoch in range(curr_epoch, epochs + 1):
            epoch_start_time = time.time()
            if training_with_samples:
                train_encoder(model, log_interval, train_dataloader, vocab_size, criterion, optimizer, scheduler, epoch, bptt, num_batches)
                val_loss, val_acc = evaluate_encoder(model, val_dataloader, val_dataset, vocab_size, criterion)
            else:
                #evaluate_transformer(model, val_dataloader, vocab, vocab_size, criterion)

                train_transformer(model, train_dataloader, label_smoothing, criterion, optimizer, epoch, log_interval, num_batches, vocab_size)
                if epoch % 1  == 0:
                    if epoch == 50:
                        pause = 1
                    val_loss, val_acc = evaluate_transformer(model, val_dataloader, vocab, vocab_size, criterion, val_dataset)
                
            # val_ppl = math.exp(val_loss)
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
                    #torch.save(model.state_dict(), f'{sys.stdout.path}best_model.pt')
                #optimizer.step()
            if epoch - best_epoch >= 15:
                break
        ######################################################################
        # Evaluate the best model on the test dataset
        # -------------------------------------------
        #best_model = copy.deepcopy(model)
        #torch.save(best_model.state_dict(), f'{sys.stdout.path}best_model.pt')

        best_model = copy.deepcopy(model)
        test_loss, test_acc = evaluate_transformer(best_model, test_dataloader, vocab, vocab_size, criterion, test_dataset)
        test_ppl = math.exp(test_loss)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f} | '
              f'test acc {test_acc:5.2f}% | Best Model from Epoch {best_epoch}')
        print('=' * 89)

        import shutil

        sys.stdout.close()
        shutil.move(f'{sys.stdout.path}', f'{sys.stdout.path[:-1]}_{int(test_acc)}/')
        sys.stdout = terminal
######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.
if __name__ == '__main__':
    terminal = sys.stdout

    with_startend = True
    'Variables for Data Preprocessing'
    REBUILD_DATA =  False
    real_world_dataset = True
    FILE_NAME = 'BPI_2014'
    #FILE_NAME = '1561989897100_0_50'
    #FILE_NAME = '1561989906741-490_100'
    #FILE_NAME = '1561989897859-21_50'
    #FILE_NAME = '1561989897286_2_0'
    #FILE_NAME = "Road_Traffic_Fine_Management_Process"
    type = "encoder_GREEDY"
    training_with_samples = False
    EIUES = 0.25
    UNCERTAINTY_TRACES = 0.20
    if(real_world_dataset):
        EIUES = 0
        UNCERTAINTY_TRACES = 0

    log = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    'Make rep. to store training data according to params'
    data_path = f"Data/{FILE_NAME}_EIUES_{EIUES * 100: 5.2f}_UT_{UNCERTAINTY_TRACES * 100: 5.2f}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    log_path = f"{FILE_NAME}_EIUES_{EIUES * 100: 5.2f}_UT_{UNCERTAINTY_TRACES * 100: 5.2f}"
    encoder_pipeline()



