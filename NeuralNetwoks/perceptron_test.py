import math
import itertools
import perceptron

class Reporter(object):
    HFMT = "{:^11s}|"
    FMT = "{:^11,.3g}|"

    def __init__(self, inputnodes, weightconfig, outputnodes):
        """
            inputnodes:
            .    Iterable of node indices for values input into the network. Used to calculate
            .    the number of fields expected in all reported values. (e.g.): [0,1]
            weightconfig:
            outputnodes:
            .    Iterable of node indices. Used only to create 
            .    Training field headers (e.g.): [node_index,...]
        """
        fields = ['A{}'.format(ni) for ni in range(len(inputnodes) + len(weightconfig))] 
        fields += ['T{}'.format(ni) for ni in outputnodes] 
        fields += ['CR{}'.format(ni) for ni,nw in weightconfig.items()] 
        fields += ['Th{}'.format(li) if nwi==nw else 'W{}{}'.format(li,nwi) for li,nw in weightconfig.items() for nwi in range(nw+1)]
        fields += ['dTh{}'.format(li) if nwi==nw else 'dW{}{}'.format(li,nwi) for li,nw in weightconfig.items() for nwi in range(nw+1)]

        self.epochs = []
        self.fields = fields
        self.numfields = len(self.fields)

    def start_epoch(self):
        """
            Mark the beginning of a training cycle. This is useful for grouping
            purposes and for reporting the Error Function value for the cycle.
        """
        self.epochs.append({"error":0,"patterns":[]})

    def end_epoch(self, error):
        """
            Mark the end of a training cycle.
        """
        epoch = self.epochs[-1]
        epoch["error"] = error

    def report_pattern(self, activations, training_values, change_rates, weights, dweights):
        # flatten all fields into a 1-D array for ease of later printing.
        pattern = [activations[k] for k in sorted(activations.keys())]
        pattern += [training_values[k] for k in sorted(training_values.keys())]
        pattern += [change_rates[k] for k in sorted(change_rates.keys())]
        pattern += list(itertools.chain.from_iterable(weights.values()))
        pattern += list(itertools.chain.from_iterable(dweights.values()))
        self.epochs[-1]["patterns"].append(pattern)

    def decipher_training_result(self, trained):
        """ Convert tri-state value (boo nullable boolean) trained to single-character identifier. """
        if trained is None: return "E"
        return "T" if trained else "F"

    def write_summary(self, scenario, trained, f):
        name,tf,tp,le,la = scenario
        f.write('{:17s}{:24s}{:3s}{:<7,.4g}{:<7,.4g}{:3s}{:<4,d}\n'.format(name, tf.__name__, tp, le, la, self.decipher_training_result(trained), len(self.epochs)))

    def write_details(self, scenario, trained, f):
        def print_header():
            f.write(''.join(self.HFMT.format(k) for k in (self.fields)))
            f.write('\n')

        def print_separator():
            f.write(''.join(["-----------+" for i in range(self.numfields)]))
            f.write('\n')

        name,tf,tp,le,la = scenario
        f.write('{}[TF:{}, T:{}, LE:{:g}, LA:{:g}] {} in {} epochs.\n'.format(name, 
                                                                              tf.__name__, 
                                                                              tp, le, la, 
                                                                              self.decipher_training_result(trained), 
                                                                              len(self.epochs)))
        print_header()
        print_separator()
        for epoch in self.epochs:
            # All fields were flattened into a 1-D array for ease here.
            f.write('\n'.join(''.join(self.FMT.format(k) for k in pattern) for pattern in epoch["patterns"]))
            f.write('Error: {:.9f}\n'.format(epoch["error"]))
            print_separator()

def transform_step(x): return 1 if x >= 0 else 0
def transform_sigmoid_abs(x): return x / (1 + math.fabs(x))
def transform_sigmoid_exp(x): return 1 / (1 + math.exp(-x))

def truncate_file(fname):
    with open(fname, "w+") as f:
        pass

BITS = (0,1)
TSET = { # Training Sets
        "OR":[((x,y),{2:int(x or y)}) for x in BITS for y in BITS],
        "AND":[((x,y),{2:int(x and y)}) for x in BITS for y in BITS],
        "NOT":[((x,),{1:int(not x)}) for x in BITS],
        "XOR":[((x,y),{3:int(x^y)}) for x in BITS for y in BITS]
       }
MAX_ITERATIONS = 256
ERROR_PRECISION = 3
PERMS = [(le, la, tf, ts) 
                    for tf in [transform_step, transform_sigmoid_abs, transform_sigmoid_exp]
                    for ts in [perceptron.TrainingStrategy.PATTERN, perceptron.TrainingStrategy.EPOCH]
                    for le in [0.25, 0.5, 0.75]
                    for la in [0.75, 0.995, 1.0, 1.005, 1.1]]

SCENARIOS = [("OR",  TSET["OR"], [((0,1),(2,))], le, la, tf, ts) for le, la, tf, ts in PERMS] + [
             ("AND", TSET["AND"], [((0,1),(2,))], le, la, tf, ts) for le, la, tf, ts in PERMS] + [
             ("NOT", TSET["NOT"], [((0,),(1,))], le, la, tf, ts) for le, la, tf, ts in PERMS] + [
             ("XOR", TSET["XOR"], [((0,1),(2,)),((0,1,2),(3,))], le, la, tf, ts) for le, la, tf, ts in PERMS]

LOG_FILE = r"c:\temp\perceptron_error.log"
SUMMARY_FILE = r"c:\temp\perceptron_summary.txt"
DETAILS_FILE = r"c:\temp\perceptron_details.txt"

WRITE_DETAILS = True

truncate_file(LOG_FILE)
truncate_file(SUMMARY_FILE)
truncate_file(DETAILS_FILE)

for name, tset, pconfig, epsilon, acceleration, xform_function, training_strategy in SCENARIOS:
    p = perceptron.FFPerceptron(xform_function, pconfig)
    reporter = Reporter(p.input_node_ixs, {ni:len(n.weights) for ni,n in p.nodes.items()}, p.output_node_ixs)

    try:
        trained = p.train(reporter, tset, ERROR_PRECISION, epsilon, acceleration, MAX_ITERATIONS, training_strategy)
    except Exception as err:
        with open(LOG_FILE, "a+") as flog:
            flog.write('{}\n{}\n'.format(str(p),repr(err)))
        trained = None
        #raise

    scenario = (name, xform_function, training_strategy, epsilon, acceleration)
    with open(SUMMARY_FILE, "a") as fsummary:
        reporter.write_summary(scenario, trained, fsummary)
    if WRITE_DETAILS:
        with open(DETAILS_FILE, "a") as fdetails:
            reporter.write_details(scenario, trained, fdetails)