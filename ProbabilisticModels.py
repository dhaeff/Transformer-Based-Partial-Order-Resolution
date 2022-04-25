# imports
from pprint import pprint
from itertools import chain, permutations, product
from tqdm import tqdm
import pm4py
import datetime
import numpy as np

from Dataset import FILE_NAME
from Logger import Logger
import sys
import os
import random
import math
terminal = sys.stdout
with_startend = True
'Variables for Data Preprocessing'
REBUILD_DATA = False # Leave at False
'''
These Models use the numpy files, certain_traces.npy, uncertain_traces.npy and index_to_uncertainty.npy, as input that are created when executing the transformer model. 
Once these files exists the probabilistic models can be executed by setting FILE_NAME to the name
of the dataset file and EIVES and UNCERTAINTY_TRACES to the right values (0.0 for real world event logs).
'''
#FILE_NAME = '1561989897286_2_0'
#FILE_NAME = '1561989897100_0_50'
FILE_NAME = 'BPI_2014'
#FILE_NAME ="1561989897859-21_50"
#FILE_NAME = "Road_Traffic_Fine_Management_Process"
#FILE_NAME = '1561989906741-490_100'
type = "POBABILISTIC"

EIUES = 0.0
UNCERTAINTY_TRACES = 0.0

log = True
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'Make rep. to store training data according to params'
data_path = f"Data/{FILE_NAME}_EIUES_{EIUES * 100: 5.2f}_UT_{UNCERTAINTY_TRACES * 100: 5.2f}"
if not os.path.exists(data_path):
    os.makedirs(data_path)
log_path = f"{FILE_NAME}_EIUES_{EIUES * 100: 5.2f}_UT_{UNCERTAINTY_TRACES * 100: 5.2f}"

sys.stdout = Logger(log_path, type, terminal)

certain_log, uncertain_log, uncertainty = np.load(f'{data_path}/certain_traces.npy', allow_pickle=True), np.load(f'{data_path}/uncertain_traces.npy', allow_pickle=True), np.load(f'{data_path}/index_to_uncertainty.npy', allow_pickle=True)


# event set functions
def pos_res_of_event_set(event_set: list) -> list:
    """For a given events set, returns the list of all possible resolutions."""

    return [list(tup) for tup in permutations(event_set, len(event_set))]


def pos_res_for_unc_trace(unc_trace: list, uncertainty) -> list:
    """Return all the possible resolution for an uncertain trace."""
    transformed_trace = []
    for idx, event in enumerate(unc_trace):
        if idx in uncertainty:
            transformed_trace[-1].append(event)
        else:
            transformed_trace.append([event])
    resolutions_for_event_sets = [pos_res_of_event_set(unc_set) for unc_set in
                                  transformed_trace]  # builds the pos_res for each event set in the trace, e.g. {C,B} -> [C,B] and [B,C]

    all_pos_res = list(product(
        *resolutions_for_event_sets))  # builds all the pos res, i.e. all combinations of all the pos res for the event sets,
    # e.g. [[A], [[C,B], [B,C]], [D]] -> [A,B,C,D] and [A,C,B,D]

    return [list(chain(*res)) for res in all_pos_res]  # make each pos res just a list of activities


# time stamp functions
def remove_timezones(log):
    """ Takes the timezone offset of each timestamp and adds it to the timestamp + removes the timezone."""

    for trace in log:
        for event in trace:
            tz_offset = event["time:timestamp"].tzinfo.utcoffset(event["time:timestamp"])
            event["time:timestamp"] = event["time:timestamp"].replace(tzinfo=None) + tz_offset
    return log


def abstract_time(log, time_func):
    """ Abstract the specified time level (in time_func) from the timestamps of the log.
        Possible: time_func <-- abstract_{microseconds, seconds, minutes, hours,
                                          day, month, year}.
    """
    for trace in log:
        for event in trace:
            event["time:timestamp"] = time_func(event["time:timestamp"])


def abstract_microseconds(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(microsecond=0)


def abstract_seconds(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(second=0, microsecond=0)


def abstract_minutes(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(minute=0, second=0, microsecond=0)


def abstract_hours(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)


def abstract_day(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def abstract_month(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def abstract_year(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace(year=1993, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def copy_timestamp(timestamp: datetime.datetime) -> datetime.datetime:
    return timestamp.replace()
# define file path'

#test_log = r"C:\Users\dhaef\OneDrive\OldOneDrive\Dokumente\Master\Thesis\RepLearning\input\1561989897286-2_0.xes"

#event_log = pm4py.read_xes(test_log)


# the logs needs to be pre-processed

# remove timezones
log = [[1,2],[1,2]]

# abstract seconds for artificial log only
#abstract_time(event_log, abstract_seconds)


class TraceEquivalenceModel():

    def __init__(self, log):
        self.NAME = "concept:name"
        self.TIME = "time:timestamp"

        #self.certain_log, self.uncertain_log, self.ground_truth_log = self.__split_and_sparse_log(log)
        self.certain_log, self.uncertain_log, self.ground_truth_log = certain_log, uncertain_log, uncertain_log
        self.uncertainty = uncertainty
        self.certain_trace_freq = self.__get_trace_frequencies()
        print("")
    def __split_and_sparse_log(self, log):
        """Extracts the certain part from a given log."""
        certain_log, uncertain_log, ground_truth_log = [], [], []
        for trace in log:
            if self.__trace_is_certain(trace):
                sparse_trace = self.__get_sparse_trace(trace)
                certain_log.append(sparse_trace)
            else:
                sparse_trace_set = self.__get_sparse_trace_set(trace)
                uncertain_log.append(sparse_trace_set)

                sparse_trace = self.__get_sparse_trace(trace)  # get the ground truth according to gold standard
                ground_truth_log.append(sparse_trace)  # i.e. order in the log
        #return certain_log, uncertain_log, ground_truth_log
        return np.load('Files2/certain_traces.npy', allow_pickle=True), np.load('Files2/randomized_uncertain_traces.npy', allow_pickle=True), np.load('Files2/uncertain_traces.npy', allow_pickle=True)

    def __get_sparse_trace(self, trace):
        """ Make a list of activities from the pm4py trace object. """
        sparse_trace = [event[self.NAME] for event in trace]
        return sparse_trace

    def __get_sparse_trace_set(self, trace) -> dict:
        trace_set = dict()
        for event in trace:
            trace_set[str(event[self.TIME])] = trace_set.get(str(event[self.TIME]), []) + [event[self.NAME]]

        trace_set = [event_set for event_set in trace_set.values()]

        return trace_set

    def __trace_is_certain(self, trace):
        """ Check if a trace is certain."""
        for i in range(len(trace) - 1):
            if trace[i][self.TIME] == trace[i + 1][self.TIME]:
                return False
        return True

    def __get_trace_frequencies(self):
        trace_freq = {}
        for trace in self.certain_log:
            trace_freq[tuple(trace)] = trace_freq.get(tuple(trace), 0) + 1
        return trace_freq

    def P_trace(self, pos_res: list) -> float:
        """
        Evaluates the probability of a possible resolution of a trace.

        Input: Possible resolution (list) of a trace.

        Returns: The probability for that given possible resolution.
        """
        try:
            return self.certain_trace_freq[tuple(pos_res)] / len(self.certain_log)
        except:
            # print("The certain trace does not contain this possible resolution. Probability undefined.")
            return 0

    def eval_model(self):
        uncertainty = self.uncertainty
        correct_pred = 0
        total_pred = len(self.ground_truth_log)  # 25502
        for i in range(total_pred):
            correct_pred += self.eval_uncertain_trace(self.uncertain_log[i], self.ground_truth_log[i], i, uncertainty[i])

        print()
        print('Accuracy:', correct_pred / total_pred)

    def eval_uncertain_trace(self, unc_trace, truth_trace, i, uncertainty):
        """ Determine the most probable possible resolution for a given uncertain trace. """
        if(len(unc_trace) > 24):
            return False
        pos_resolutions = pos_res_for_unc_trace(unc_trace, uncertainty)
        probs = [(pos_res, self.P_trace(pos_res)) for pos_res in
                 pos_resolutions]  # get tuple (prob, pos_res) for each pos_res
        del pos_resolutions  # need RAM
        probs.sort(key=lambda tup: (tup[1], random.random()), reverse=True)

        pred_trace = probs[0][0]
        prob = probs[0][1]
        del probs  # need RAM

        if i % 250== 0:
            print(i, unc_trace, (pred_trace, prob), truth_trace)

        return pred_trace == truth_trace

trace_equiv_model = TraceEquivalenceModel(log)

print(len(trace_equiv_model.certain_log))
print(len(trace_equiv_model.uncertain_log))
print(len(trace_equiv_model.ground_truth_log))

trace_equiv_model.eval_model()
x = 1

class NGramModel():

    def __init__(self, log):
        self.NAME = "concept:name"
        self.TIME = "time:timestamp"

        self.log = log
        self.certain_log_set = certain_log #np.load('Files2/certain_traces.npy', allow_pickle=True)
        #self.certain_log_set = self.__make_certain_log_set(self.certain_log_set)
        self.uncertain_log = uncertain_log #np.load('Files2/randomized_uncertain_traces.npy', allow_pickle=True)
        self.uncertainty = uncertainty# np.load('Files2/index_to_uncertainty.npy', allow_pickle=True)
        self.uncertain_log_set = self.__make_uncertain_log_set(self.uncertain_log, self.uncertainty)
        self.log_set = np.concatenate([self.certain_log_set,self.uncertain_log_set])

        self.certain_sequences = [self.__make_certain_sequences(trace_set) for trace_set in self.log_set]
        #self.certain_sequences = list(np.unique(self.certain_sequences))
        self.ground_truth_log = uncertain_log #np.load('Files2/uncertain_traces.npy', allow_pickle=True)
        self.p_a_act_seq = dict()
    def __make_uncertain_trace_sets(self, unc_trace, uncertainty):
        transformed_trace = []
        for idx, event in enumerate(unc_trace):
            if idx in uncertainty:
                transformed_trace[-1].append(event)
            else:
                transformed_trace.append([event])
        return transformed_trace

    def __make_uncertain_log_set(self, log, uncertainty) -> list:
        log_set = []
        for idx, trace in enumerate(log):
            log_set.append(self.__make_uncertain_trace_sets(trace, uncertainty[idx]))
        return log_set
    def __make_certain_trace_sets(self, trace):
        transformed_trace = []
        for idx, event in enumerate(trace):
            transformed_trace.append([event])

        return transformed_trace

    def __make_certain_log_set(self, log):
        log_set = []
        for idx, trace in enumerate(log):
            log_set.append(self.__make_certain_trace_sets(trace))
        return log_set



    def __make_trace_sets(self, trace) -> dict:
        trace_set = dict()
        for event in trace:
            trace_set[str(event[self.TIME])] = trace_set.get(str(event[self.TIME]), []) + [event[self.NAME]]
        return trace_set

    def __make_certain_sequences(self, trace_set) -> list:
        '''
        For each uncertain trace we cut out the certain subtraces, e.g. [{1}, {2,3}, {4}, {5}] -> [[1],[4,5]]
        And in those we search for an activity sequence to be present
        Because for an activity sequence to be certain in a trace it must apper in a certain sequence of a trace
        '''

        certain_sequences = []
        certain_sequence = []
        for i, timestamp in enumerate(trace_set):
            if timestamp.__len__() == 1:
                if i == len(trace_set) - 1:
                    certain_sequence.append(timestamp[0])
                    certain_sequences.append(certain_sequence)
                else:
                    certain_sequence.append(timestamp[0])
            else:
                if certain_sequence:
                    certain_sequences.append(certain_sequence)
                certain_sequence = []
        #certain_sequences = np.unique(certain_sequences)
        return certain_sequences

    def __split_and_sparse_log(self, log):
        """Prepare the data for evaluation, i.e. the uncertain log and corresponding ground truth."""

        uncertain_log, ground_truth_log = [], []
        for trace in log:
            if not self.__trace_is_certain(trace):
                sparse_trace_set = self.__get_sparse_trace_set(trace)
                uncertain_log.append(sparse_trace_set)

                sparse_trace = self.__get_sparse_trace(trace)  # get the ground truth according to gold standard
                ground_truth_log.append(sparse_trace)  # i.e. order in the log

        return uncertain_log, ground_truth_log

    def __get_sparse_trace(self, trace):
        """ Make a list of activities from the pm4py trace object. """
        sparse_trace = [event[self.NAME] for event in trace]
        return sparse_trace

    def __get_sparse_trace_set(self, trace) -> dict:
        trace_set = dict()
        for event in trace:
            trace_set[str(event[self.TIME])] = trace_set.get(str(event[self.TIME]), []) + [event[self.NAME]]

        trace_set = [event_set for event_set in trace_set.values()]

        return trace_set

    def __trace_is_certain(self, trace):
        """ Check if a trace is certain."""
        for i in range(len(trace) - 1):
            if trace[i][self.TIME] == trace[i + 1][self.TIME]:
                return False
        return True

    def __activities_in_sequence(self, activity_sequence: list, certain_sequence: list) -> bool:
        for i in range(len(certain_sequence) - len(activity_sequence) + 1):
            if certain_sequence[i:i + len(activity_sequence)] == activity_sequence:
                return True
        return False

    def certain(self, activity_sequence: list, trace_set: dict) -> bool:
        certain_sequences = self.__make_certain_sequences(trace_set)
        #certain_sequences = list(np.unique(certain_sequences))
        for certain_sequence in certain_sequences:
            if self.__activities_in_sequence(activity_sequence, certain_sequence):
                return True
        return False

    def P_a_activity_sequence(self, activity: str, activity_sequence: list) -> float:
        n_sequence_plus_activity_is_certain = 0
        n_sequence_is_certain = 0
        key = activity + ''.join(activity_sequence)
        if((key) not in self.p_a_act_seq):

            for trace_set in self.log_set:
                if self.certain(activity_sequence, trace_set):
                    n_sequence_is_certain += 1
                    if self.certain(activity_sequence + [activity], trace_set):
                        n_sequence_plus_activity_is_certain += 1

            if n_sequence_is_certain == 0:
                return 0
            output = n_sequence_plus_activity_is_certain / n_sequence_is_certain
            self.p_a_act_seq[key] = output
            return output
        else:
            return self.p_a_act_seq[key]

    def P_n_gram(self, pos_res: list, n: int = 2):
        # possible_resolution like [a, b, c]
        lower_bound = 1  # 2 in the paper, but indexing here starts one before
        upper_bound = len(pos_res)
        result = 1.0
        for i in range(lower_bound, upper_bound):
            s_index = max(i - n + 1,
                          0)  # the gram is not n long in the beginning, its 2 long, then 3, ... until it's always n long
            result *= self.P_a_activity_sequence(pos_res[i], pos_res[s_index:i])
        return result

    def eval_model(self, n: int = 2):
        correct_pred = 0
        total_pred = len(self.ground_truth_log)  # bpic14: 25502
        uncertainty = self.uncertainty
        for i in tqdm(range(total_pred)):
            correct_pred += self.eval_uncertain_trace(self.uncertain_log[i], self.ground_truth_log[i], n, i, uncertainty[i])

        print()
        print('Accuracy:', correct_pred / total_pred)

    def eval_uncertain_trace(self, unc_trace, truth_trace, n, i, uncertainty):
        """ Determine the most probable possible resolution for a given uncertain trace. """
        if(len(unc_trace) > 24):
            return False
        pos_resolutions = pos_res_for_unc_trace(unc_trace, uncertainty)
        probs = [(pos_res, self.P_n_gram(pos_res, n)) for pos_res in
                 pos_resolutions]  # get tuple (prob, pos_res) for each pos_res
        probs.sort(key=lambda tup: (tup[1], random.random()), reverse=True)

        pred_trace = probs[0][0]
        prob = probs[0][1]

        #if i % 100 == 0:
            #print(i, unc_trace, (pred_trace, prob), truth_trace)

        return pred_trace == truth_trace


n_gram_model = NGramModel(log)
for n in range(2,3):
    print('Running experiment on N =', n)
    n_gram_model.eval_model(n)
    print('\n\n')


class WeakOrderModel():

    def __init__(self, log):
        self.NAME = "concept:name"
        self.TIME = "time:timestamp"

        self.log = log
        self.certain_log_set = certain_log #np.load('Files2/certain_traces.npy', allow_pickle=True)
        self.certain_log_set = [np.expand_dims(trace_set, axis=1) for trace_set in self.certain_log_set]
        self.uncertain_log = uncertain_log #np.load('Files2/randomized_uncertain_traces.npy', allow_pickle=True)
        self.uncertainty = uncertainty #np.load('Files2/index_to_uncertainty.npy', allow_pickle=True)
        self.uncertain_log_set = self.__make_uncertain_log_set(self.uncertain_log, self.uncertainty)
        self.log_set = np.concatenate([self.uncertain_log_set, self.certain_log_set])

        #self.certain_sequences = [self.__make_certain_sequences(trace_set) for trace_set in self.log_set]
        #self.certain_sequences = list(np.unique(self.certain_sequences))
        self.ground_truth_log = uncertain_log #np.load('Files2/uncertain_traces.npy', allow_pickle=True)
        self.p_a_act_seq = dict()
    def __make_log_set(self) -> list:
        log_set = []
        for trace in self.log:
            log_set.append(self.__make_trace_sets(trace))
        return log_set

    def __make_trace_sets(self, trace) -> dict:
        trace_set = dict()
        for event in trace:
            trace_set[str(event[self.TIME])] = trace_set.get(str(event[self.TIME]), []) + [event[self.NAME]]
        return trace_set

    def __make_uncertain_trace_sets(self, unc_trace, uncertainty):
        transformed_trace = []
        for idx, event in enumerate(unc_trace):
            if idx in uncertainty:
                transformed_trace[-1].append(event)
            else:
                transformed_trace.append([event])
        return transformed_trace

    def __make_uncertain_log_set(self, log, uncertainty) -> list:
        log_set = []
        for idx, trace in enumerate(log):
            log_set.append(self.__make_uncertain_trace_sets(trace, uncertainty[idx]))
        return log_set

    def __split_and_sparse_log(self, log):
        """Prepare the data for evaluation, i.e. the uncertain log and corresponding ground truth."""

        uncertain_log, ground_truth_log = [], []
        for trace in log:
            if not self.__trace_is_certain(trace):
                sparse_trace_set = self.__get_sparse_trace_set(trace)
                uncertain_log.append(sparse_trace_set)

                sparse_trace = self.__get_sparse_trace(trace)  # get the ground truth according to gold standard
                ground_truth_log.append(sparse_trace)  # i.e. order in the log

        return uncertain_log, ground_truth_log

    def __get_sparse_trace(self, trace):
        """ Make a list of activities from the pm4py trace object. """
        sparse_trace = [event[self.NAME] for event in trace]
        return sparse_trace

    def __get_sparse_trace_set(self, trace) -> dict:
        trace_set = dict()
        for event in trace:
            trace_set[str(event[self.TIME])] = trace_set.get(str(event[self.TIME]), []) + [event[self.NAME]]

        trace_set = [event_set for event_set in trace_set.values()]

        return trace_set

    def __trace_is_certain(self, trace):
        """ Check if a trace is certain."""
        for i in range(len(trace) - 1):
            if trace[i][self.TIME] == trace[i + 1][self.TIME]:
                return False
        return True

    def order(self, a: str, b: str, trace_set) -> bool:
        """Check whether a trace contains activity a before activity b."""

        trace_list = trace_set
        for i in range(len(trace_list) - 1):
            if a in trace_list[i]:
                for j in range(i + 1, len(trace_list)):
                    if b in trace_list[j]:
                        return True
        return False

    def contains_activities(self, a: str, b: str, trace_set) -> bool:
        """Check whether a trace contains the activities a and b."""

        activities = [a for event_set in trace_set for a in event_set]
        return (a in activities and b in activities)

    def P_a_b(self, a: str, b: str) -> float:
        """Computes the probability of activity a and b to occur in weak order."""

        n_traces_with_a_b_in_weak_order = 0
        n_traces_with_a_b = 0
        key = a+b
        if(key not in self.p_a_act_seq):
            for trace_set in self.log_set:
                if self.contains_activities(a, b, trace_set):
                    n_traces_with_a_b += 1
                    if self.order(a, b, trace_set):
                        n_traces_with_a_b_in_weak_order += 1

            if n_traces_with_a_b == 0:
                self.p_a_act_seq[key] = 0
                return 0
            else:
                output = n_traces_with_a_b_in_weak_order / n_traces_with_a_b
                self.p_a_act_seq[key] = output
                return n_traces_with_a_b_in_weak_order / n_traces_with_a_b
        else:
            return self.p_a_act_seq[key]

    def P_weak_order(self, pos_res: list) -> float:
        """Compute the probability of a possible resolution accorind to the weak order model."""

        result = 1.0

        for i in range(len(pos_res) - 1):
            for j in range(i + 1, len(pos_res)):
                result *= self.P_a_b(pos_res[i], pos_res[j])

        return result

    def eval_model(self):
        correct_pred = 0
        total_pred = len(self.ground_truth_log)  # bpic14: 25502
        uncertainty = self.uncertainty
        for i in range(total_pred):
            if i % 1000 == 0: print(i)
            correct_pred += self.eval_uncertain_trace(self.uncertain_log[i], self.ground_truth_log[i], i, uncertainty[i])

        print()
        print('Accuracy:', correct_pred / total_pred)

    def eval_uncertain_trace(self, unc_trace, truth_trace, i, uncertainty):
        """ Determine the most probable possible resolution for a given uncertain trace. """
        if(len(unc_trace) > 24):
            return False
        pos_resolutions = pos_res_for_unc_trace(unc_trace, uncertainty)
        probs = [(pos_res, self.P_weak_order(pos_res)) for pos_res in
                 pos_resolutions]  # get tuple (prob, pos_res) for each pos_res
        probs.sort(key=lambda tup: (tup[1], random.random()), reverse=True)

        pred_trace = probs[0][0]
        prob = probs[0][1]

        #if i % 100 == 0:
            #print(i, unc_trace, (pred_trace, prob), truth_trace)

        return pred_trace == truth_trace

weak_order_model = WeakOrderModel(log)
weak_order_model.eval_model()
