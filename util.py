import numpy as np
import random
import torch
from itertools import permutations
#from Dataset import REBUILD_DATA
from collections import OrderedDict
from itertools import chain, combinations
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
import pandas as pd

def parse_csv(file_name, data_path, REBUILD_DATA):
    if(REBUILD_DATA):
        event_log = pd.read_csv('./input/' + file_name +'.csv', sep=";")
        event_log.drop("IncidentActivity_Number", axis=1)
        event_log.rename(columns={'IncidentActivity_Type': 'concept:name'}, inplace=True)
        event_log.rename(columns={'DateStamp': 'time:timestamp'}, inplace=True)
        event_log.rename(columns={'Incident ID': 'case:concept:name'}, inplace=True)
        event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
        event_log = event_log.sort_values('time:timestamp')
        #parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'Incident ID'}

        event_log = log_converter.apply(event_log)
        filtered_log = pm4py.filter_case_size(event_log, 5, 10)
        #event_log = log_converter.apply(event_log)
        certain_traces = []
        uncertain_traces = []
        index_to_uncertainty = []
        for trace in event_log:
            curr_trace = []
            uncertainty = []
            for id, event in enumerate(trace):
                curr_trace.append(event["concept:name"])
                if (id > 0):
                    if event["time:timestamp"] == trace[id - 1]["time:timestamp"]:
                        uncertainty.append(id)
            if (len(uncertainty) == 0):
                certain_traces.append(curr_trace)
            else:
                uncertain_traces.append(curr_trace)
                index_to_uncertainty.append(uncertainty)
        np.save(f'{data_path}/index_to_uncertainty.npy', index_to_uncertainty)
        np.save(f'{data_path}/certain_traces.npy', certain_traces)
        np.save(f'{data_path}/uncertain_traces.npy', uncertain_traces)

        return index_to_uncertainty ,certain_traces, uncertain_traces
def parse_xes(file_name, data_path, REBUILD_DATA):
    if(REBUILD_DATA):
        event_log = xes_importer.apply('./input/' + file_name +'.xes')
        #event_log = pm4py.filter_case_size(event_log, 1, 25)
        certain_traces = []
        uncertain_traces = []
        index_to_uncertainty = []
        for trace in event_log:
            curr_trace = []
            uncertainty = []
            for id, event in enumerate(trace):
                curr_trace.append(event["concept:name"])
                if(id > 0):
                    if event["time:timestamp"] == trace[id-1]["time:timestamp"]:
                        uncertainty.append(id)
            if(len(uncertainty) == 0):
                certain_traces.append(curr_trace)
            else:
                uncertain_traces.append(curr_trace)
                index_to_uncertainty.append(uncertainty)
        np.save(f'{data_path}/index_to_uncertainty.npy', index_to_uncertainty)
        np.save(f'{data_path}/certain_traces.npy', certain_traces)
        np.save(f'{data_path}/uncertain_traces.npy', uncertain_traces)

        return index_to_uncertainty ,certain_traces, uncertain_traces

    else:
        return np.load(f'{data_path}/index_to_uncertainty.npy', allow_pickle=True), np.load(
            f'{data_path}/certain_traces.npy', allow_pickle=True), np.load(f'{data_path}/uncertain_traces.npy',
                                                                           allow_pickle=True)


def load_synthetic_collection(filename):
    texts = []
    no = 0
    tree = etree.parse('input/'+filename)
    root= tree.getroot()
    print("Start")
    for element in tqdm(root.iter()):
        #tag= element.tag.split('}')[1]
        tag = element.tag
        if(tag== "trace"):
            wordslist = []
            tagslist = []
            no += 1
            for childelement in element.iterchildren():
                #ctag= childelement.tag.split('}')[1]
                ctag = childelement.tag
                if(ctag== "string" and childelement.get('key')=='concept:name'):
                    doc_name=childelement.get('value')
                elif (ctag=="event"):
                    for grandchildelement in childelement.iterchildren():
                        if(grandchildelement.get('key')=='concept:name'):
                            event_name=grandchildelement.get('value')
                        #    print(event_name)
                            regex = re.compile('^tau')
                            if not (regex.match(event_name)):
                                wordslist.append(event_name.replace(' ',''))
            texts.append(wordslist)
            #if(no == 19999):
                #return texts
    return texts
def create_uncertainty(data_path, REBUILD_DATA, traces, uncertainty_traces, eiues):
    if REBUILD_DATA:
        index_to_uncertainty = []
        certain_traces = []
        uncertain_traces = []

        idx = 0
        for trace in traces:
            #event_idx = 0
            check_uncertainty = False
            uncertain_events = []
            if (random.random() > uncertainty_traces) or (len(trace)<3):
                uncertain_events.append(-1)
                #index_to_uncertainty.append(uncertain_events)
                certain_traces.append(trace)
                idx += 1
                continue

            while(not check_uncertainty):
                for event_idx, event in enumerate(trace):

                    if event_idx == 0:
                        event_idx = random.randint(1,len(trace)-1)
                    if random.random() < eiues:
                        if event_idx not in uncertain_events:
                            uncertain_events.append(event_idx)
                        check_uncertainty = True
            uncertain_events.sort()
            index_to_uncertainty.append(uncertain_events)
            uncertain_traces.append(trace)
            idx += 1
        np.save(f'{data_path}/index_to_uncertainty.npy', index_to_uncertainty)
        np.save(f'{data_path}/certain_traces.npy', certain_traces)
        np.save(f'{data_path}/uncertain_traces.npy', uncertain_traces)
        #a = [x for x in index_to_uncertainty if x != [-1]]
        return index_to_uncertainty, certain_traces, uncertain_traces
    else:
        return np.load(f'{data_path}/index_to_uncertainty.npy', allow_pickle=True), np.load(f'{data_path}/certain_traces.npy', allow_pickle=True), np.load(f'{data_path}/uncertain_traces.npy', allow_pickle=True)



def negative_sampling_ctraces(certain_traces, max_length_uncertainty):
    input = []
    target = []
    certain_traces_tuple = [tuple(i) for i in np.unique(certain_traces)]
    for trace in certain_traces:
        all_perm = permutations(trace)
        filtered_perm = all_perm
        for id, i in enumerate(trace):
            if(id <= max_length_uncertainty -2):
                perm_filter = trace[:(max_length_uncertainty)]
            elif(id+max_length_uncertainty-1 == len(trace)):
                perm_filter = trace[id-(max_length_uncertainty-1):]
            else:
                perm_filter = trace[id-(max_length_uncertainty-1):(id+(max_length_uncertainty))]
            filtered_perm = [i for i in filtered_perm if i[id] in perm_filter]
        #filtered_perm = [i for i in filtered_perm if (i not in certain_traces_tuple) or (i is trace)]
        input.append(filtered_perm)
    target = [certain_traces[id] for id, sublist in enumerate(input) for item in sublist]
    input = [item for sublist in input for item in sublist]
    return input, target

def negative_sampling_with_tokens(certain_traces, max_length_uncertainty, uncertain_subtraces, uncertain_traces_rand):

    input = []
    target = []
    uncertain_traces_rand = [[y for y in x if y != "<PAD>"] for x in uncertain_traces_rand]
    uncertain_traces_rand = set(tuple(x) for x in uncertain_traces_rand)
    shapes = get_sampling_shapes(certain_traces)
    #shapes = rec_shapes(certain_traces, 4, pre = [], pre_check = False, inserts = [])
    for trace in certain_traces:
        trace_len = min(len(trace),15)
        for shape in shapes[trace_len]:
            counter = 0
            sample = trace.copy()
            for pos in reversed(shape):
                if((counter % 2) == 0):
                    sample.insert(pos,"<EOS>")
                    end_id = pos
                    counter += 1
                else:
                    sample.insert(pos-1, "<SOS>")
                    start_id = pos
                    sample[start_id:end_id+1] = sorted(sample[start_id:end_id+1])
                    sample[start_id:end_id + 1] = ["".join(sample[start_id:end_id+1])]
                    counter += 1
            if(not set(sample).isdisjoint(uncertain_subtraces)):
                input.append(sample)
                target.append(trace.copy())
        input.append(trace.copy())
        target.append(trace.copy())
    pad_traces(input, target)
    input = tuple(tuple(x) for x in input)
    target = tuple(tuple(x) for x in target)
    ziped = set(zip(input, target))
    input, target = zip(*list(ziped))
    input = [list(x) for x in input]
    target = [list(x) for x in target]
    return input, target


def get_sampling_shapes(certain_traces, thresh):
    shapes = {}
    min_trace_length = len(min(certain_traces, key=len))
    max_trace_length = min(len(max(certain_traces, key=len)), thresh)
    for i in range(min_trace_length, max_trace_length + 1):
        trace = list(range(1, i + 1))
        # max_length = min(max_length_uncertainty, len(trace))
        # temp, a = rec_shapes(trace, max_length, pre = [], inserts = [])
        tmp = list(powerset(trace))
        tmp = [list(y) for y in tmp if len(y) % 2 == 0 and y != []]
        shapes[i] = tmp.copy()
    return shapes

def masked_negative_sampling(certain_traces):
    input = []
    target = []
    #certain_traces = add_sot(certain_traces)
    thresh = 15
    shapes = get_sampling_shapes(certain_traces, thresh)

    for trace in certain_traces:
        if len(trace) >= thresh:
            if True:#trace not in input:
                input.append(trace.copy())
                target.append(trace.copy())
            continue
        trace_len = min(len(trace), thresh)
        for shape in shapes[trace_len]:
            sample = trace.copy()

            for id, pos in enumerate(shape):

                if (id+1) % 2 == 0:

                    for i in range(shape[id-1], pos+1):
                        sample[i-1] = "<MASK>"
            if sample not in input:
                input.append(sample)
                target.append(trace.copy())
        if True:#trace not in input:
            input.append(trace.copy())
            target.append(trace.copy())

    #pad_traces(input, target)
    add_sot(input)
    add_sot(target)
    return input, target


def masked_uncertain_traces(uncertainty, uncertain_traces):
    output = []
    positions_uncertainty = []
    for id,uncertain_trace in enumerate(uncertainty):
        curr_pos = []
        curr_trace = uncertain_traces[id].copy()
        for i, e in reversed(list(enumerate(uncertain_trace))):
            curr_trace[e] = "<MASK>"
            if i == 0:
                curr_trace[e-1] = "<MASK>"
                curr_pos.append(e)
            elif (e - uncertain_trace[i-1] != 1):
                curr_trace[e-1] = "<MASK>"
                curr_pos.append(e)
        output.append(curr_trace)
        positions_uncertainty.append(sorted(curr_pos))
    add_sot(output)
    uncertain_traces
    return output, uncertain_traces, positions_uncertainty


"""
def masked_uncertain_traces(uncertain_traces, uncertainty):
    masked_traces = []

    for id, unc_trace in enumerate uncertain_traces:

        for idx, event in enumerate(unc_trace):
            if idx in uncertainty:
                transformed_trace[-1].append(event)
            else:
                transformed_trace.append([event])

"""
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def mean_mask(input):
    log_mask = []
    curr_mask = []
    check_uncertain_subtrace = False
    for trace in input:
        trace_mask = []
        for id, activity in enumerate(trace):
            if activity == "<EOS>":
                check_uncertain_subtrace = False
                trace_mask.append(curr_mask)
                curr_mask = []
            if check_uncertain_subtrace:
                curr_mask.append(id)
            if activity == "<SOS>":
                check_uncertain_subtrace = True
                trace_mask.append([id])
            if not check_uncertain_subtrace:
                trace_mask.append([id])
        log_mask.append(trace_mask)
    return log_mask

def create_res_mask(uncertain_subtraces, len_vocab):
    output =[]
    for trace in uncertain_subtraces:
        curr = torch.zeros(len_vocab, dtype=torch.bool)
        curr[trace] = True
        output.append(curr.reshape(-1,1))
    return torch.permute(torch.cat(output, dim = 1), (0,1))

def insert_predictions(input, index_uncertainty, most_probable_last_token, uncertain_subtraces, first_iter):
    output_traces = []
    for id, trace in enumerate(list(input)):
        updated_trace = trace.tolist()
        #if first_iter:
        if(len(uncertain_subtraces[id]) > 0):
            updated_trace[index_uncertainty[id]] = most_probable_last_token[id][0]
        '''
        else:
            if most_probable_last_token[id][0] == 0:
                updated_trace.append(0)
            else:
                updated_trace = np.insert(updated_trace, index_uncertainty[id]+1, most_probable_last_token[id][0])
                '''
        updated_trace = torch.from_numpy(np.asarray(updated_trace))
        output_traces.append(updated_trace)
    #if not first_iter:
    index_uncertainty = torch.add(index_uncertainty, 1)
    return torch.cat(output_traces).reshape(-1, updated_trace.shape[0]), False, index_uncertainty

def combine_predictions(input, original_index_uncertainty, uncertain_subtraces_per_trace, length_uncertainty, max_len_out):
    curr_idx = 0
    output = []
    for x in uncertain_subtraces_per_trace:
        if x == 1 :
            combined = input[curr_idx][1:max_len_out+1]
            combined[combined == 2] = 0
            output.append(combined)
            curr_idx += 1
        else:
            combined = input[curr_idx][1:end_subtrace(curr_idx, length_uncertainty, original_index_uncertainty)]
            start_next_subtrace = end_subtrace(curr_idx, length_uncertainty, original_index_uncertainty)
            curr_idx += 1
            for i in range(1,x):
                if i == (x-1):
                    combined = torch.cat((combined, input[curr_idx][start_next_subtrace:]))
                    combined = combined[:max_len_out+1]
                    curr_idx += 1
                else:
                    combined = torch.cat((combined, input[curr_idx][start_next_subtrace:end_subtrace(curr_idx, length_uncertainty, original_index_uncertainty)]))
                    start_next_subtrace = end_subtrace(curr_idx, length_uncertainty, original_index_uncertainty)
                    curr_idx += 1
            combined[combined == 2] = 0
            output.append(combined)

    return torch.cat(output).reshape(uncertain_subtraces_per_trace.shape[0], -1)


def end_subtrace(curr_idx, length_uncertainty, original_index_uncertainty):
    return original_index_uncertainty[curr_idx] + length_uncertainty[curr_idx]


def rec_shapes(trace, max_length_uncertainty, pre = [], pre_check = False, inserts = []):
    for length in range(2,max_length_uncertainty+1):
        for i in range(0,len(trace)-length+1):
            if pre_check:
                pre = [item for item in pre if item not in trace]
                pre_check = False
            pre += [trace[i], trace[i+length-1]]
            inserts.append(pre.copy())
            if len(trace) == length:
                pre_check = True
                return inserts, pre_check
            else:
                inserts, pre_check = rec_shapes(trace[(i+length):], length, pre)

    return inserts, True


def uncertain_traces_with_tokens(uncertain_traces, uncertainty, training_with_samples):
    input = []
    target = []
    uncertain_subtraces = []
    #uncertain_subtraces = set()
    for idx, trace in enumerate(uncertain_traces):
        trace_with_tokens = trace.copy()
        uncertain_subtrace = []
        counter = 0
        if idx == 34:
            x = 1
        for id, num in enumerate(reversed(uncertainty[idx])):
            if counter % 2 == 0:

                if training_with_samples:
                    trace_with_tokens.insert(num+1, "<EOS>")

                end_id = num+1
                counter += 1
                if len(uncertainty[idx]) - id == 1:

                    if training_with_samples:
                        trace_with_tokens.insert(num - 1, "<SOS>")

                    start_id = num
                    join_activities(end_id, start_id, trace_with_tokens, uncertain_subtrace)
                    #uncertain_subtraces.add(trace_with_tokens[start_id])
                    break
                elif num-1 != uncertainty[idx][len(uncertainty[idx])-(id+2)]:

                    if training_with_samples:
                        trace_with_tokens.insert(num-1, "<SOS>")

                    start_id = num
                    join_activities(end_id, start_id, trace_with_tokens, uncertain_subtrace)
                    #uncertain_subtraces.add(trace_with_tokens[start_id])
                    counter += 1
                    continue
            else:
                if len(uncertainty[idx]) - id == 1:

                    if training_with_samples:
                        trace_with_tokens.insert(num - 1, "<SOS>")

                    start_id = num
                    join_activities(end_id, start_id, trace_with_tokens, uncertain_subtrace)
                    #uncertain_subtraces.add(trace_with_tokens[start_id])
                    break
                if num-1 == uncertainty[idx][len(uncertainty[idx])-(id+2)]:
                    continue
                else:

                    if training_with_samples:
                        trace_with_tokens.insert(num-1, "<SOS>")
                    start_id = num
                    join_activities(end_id, start_id, trace_with_tokens, uncertain_subtrace)
                    #uncertain_subtraces.add(trace_with_tokens[start_id])
                    counter += 1
                    continue

        if not training_with_samples:
            trace_with_tokens.insert(0, "<SOT>")  # insert Start of Trace Token
            trace_with_tokens.append("<EOT>") #insert end of Trace Token
        uncertain_subtrace = np.flip(uncertain_subtrace, axis=0).tolist()
        uncertain_subtraces.append(uncertain_subtrace)
        target.append(trace.copy())
        input.append(trace_with_tokens)

    pad_traces(input, target)

    return input, target, uncertain_subtraces


def pad_traces(input, target):
    for id, sample in enumerate(target):
        if (len(input[id]) > len(sample)):
            target[id].extend(["<PAD>"] * (len(input[id]) - len(sample)))
    for id, sample in enumerate(input):
        if (len(target[id]) > len(sample)):
            input[id].extend(["<PAD>"] * (len(target[id]) - len(sample)))


def join_activities(end_id, start_id, trace_with_tokens, uncertain_subtrace):
    uncertain_subtrace.append(trace_with_tokens[(start_id-1):end_id])
    trace_with_tokens[(start_id-1):end_id] = sorted(trace_with_tokens[(start_id-1):end_id])
    trace_with_tokens[(start_id-1):end_id] = ["".join(trace_with_tokens[(start_id-1):end_id])]

def add_position_uncertain_tokens(uncertain_traces_randomized, vocab_raw, uncertain_subtrace):
    pos_trace = []
    for id, trace in enumerate(uncertain_traces_randomized):
        positions = [id for id,act in enumerate(trace) if act not in vocab_raw.act_to_index]
        pos_trace.append(tuple((positions, uncertain_subtrace[id])))
    return pos_trace

def randomize_uncertain_events(data_path, REBUILD_DATA, uncertain_traces, uncertainty):
    if REBUILD_DATA:
        randomized_traces = []
        for idx, trace in enumerate(uncertain_traces):
            randomized_trace = trace.copy()
            for num in uncertainty[idx]:
                if (random.random() < .5):
                    if num == 0:
                        randomized_trace[num], randomized_trace[num+1] = trace[num+1], trace[num]
                    else:
                        #print(f'TraceLength:{len(trace)} Num: {num})')
                        randomized_trace[num], randomized_trace[num - 1] = trace[num - 1], trace[num]
            randomized_traces.append(randomized_trace)
            np.save(f'{data_path}/randomized_uncertain_traces.npy', randomized_traces)
        return randomized_traces
    else:
        return np.load(f'{data_path}/randomized_uncertain_traces.npy', allow_pickle=True)


def add_sot(train_data):
    """
    Add Start of Trace and End of Trace Tokens
    :param train_data:
    :return:
    """
    for trace in train_data:
        trace.insert(0, "<SOT>")
        trace.append("<EOT>")
