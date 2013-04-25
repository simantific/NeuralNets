import math

def i_report_pattern(activations, training_values, change_rates, weights, dweights):
    """
        Empty prototype for reporting the state of a network following a training pattern.
        Example from perceptron of 2 inputs, 1 hidden node, and 1 output node 
        (1st epoch, 1st pattern):
        .  activations = { 0:0.00, 1:0.00, 2:0.50, 3:0.50 }
        .  training_values = { 3:0.00 }
        .  change_rates = { 2:0.0000, 3: -0.1250 }
        .  weights = { 2:[0.0000,0.0000,0.0000], 3:[0.0000,0.0000,0.0000,0.0000] }
        .  dweights = { 2:[0.0000,0.0000,0.0000], 3:[0.0000,0.0000,-0.0156,-0.0312] }
    """
    pass

class TrainingStrategy(object):
    PATTERN = "P"
    EPOCH = "E"

class Node(object):
    """
        Represents one neuron in a perceptron with memory of last input and all 
        weight changes applied during the life of this instance.
    """

    def __init__(self, numinputs):
        self.numinputs = numinputs
        self.weights = [0]*numinputs
        self.theta = 0
        self.last_input = None
        # Remember all dW & dTheta values until they are applied to the weights and theta.
        self.temp_delta_memory = []
        # Remember all dW & dTheta values for life. dTheta is appended to dW here for simplicity.
        self.perm_delta_memory = [([0]*(numinputs),0)] 

    def __str__(self): return '{} ; {} ; {}'.format(str(self.weights), self.theta, str(self.temp_delta_memory))

    def accept_input(self, input_vector, transform_function):
        def sum_function(V, W, Theta):
            return sum(v * W[vi] for vi,v in enumerate(V)) + Theta

        self.last_input = input_vector
        return transform_function(sum_function(input_vector, self.weights, self.theta))

    def remember_deltas(self, delta, learning_epsilon, learning_acceleration):
        # add the previous deltas times a learning acceleration factor (learning_acceleration) both to
        # help prevent radical weight changes and to apply some momentum to learning.
        p_dweights, p_dtheta = self.perm_delta_memory[-1]
        magnitude = delta * learning_epsilon

        dW = [magnitude * self.last_input[vi] + learning_acceleration * p_dweights[vi] for vi in range(self.numinputs)]
        dTheta = magnitude + learning_acceleration * p_dtheta

        self.temp_delta_memory.append((dW, dTheta)) # remember until deltas are applied to weights
        self.perm_delta_memory.append((dW, dTheta)) # remember for life.

    def apply_deltas(self):
        """
            Add all previously calculated dW[i] and dTheta values to W[i] and Theta, then 
            forget the dWs and dThetas.
        """
        for dW, dTheta in self.temp_delta_memory:
            self.weights = [w + dW[wi] for wi,w in enumerate(self.weights)]
            self.theta += dTheta
        self.temp_delta_memory = []

class FFPerceptron(object):
    """
        A feed-forward artificial neural network with configurable neural pathways.
    """
    def __init__(self, transform_function, links):
        from collections import defaultdict

        sender_node_ixs = set(ni for iv,ov in links for ni in iv) # all nodes that send input to another node.
        receiver_node_ixs = set(ni for iv,ov in links for ni in ov) # all nodes that receive input from another node.
        nodes = defaultdict(lambda:None) # all nodes with calculation function
        output_paths = defaultdict(list) # tracks which inputs go to which nodes

        # generate node instances and build output paths.
        for input_vector,output_vector in links:
            for node_index in output_vector:
                if node_index not in nodes: # In case a node shows up in multiple output vectors.
                    nodes[node_index] = Node(len(input_vector))
            for node_position,node_index in enumerate(input_vector):
                output_paths[node_index].extend([{'atposition':node_position, 'tonode':output_node_index} for output_node_index in output_vector])

        self.links = links
        self.output_node_ixs = receiver_node_ixs - sender_node_ixs # nodes that output from the system.
        self.input_node_ixs = sender_node_ixs - receiver_node_ixs # nodes that are input to the system.
        self.nodes = nodes
        self.output_paths = output_paths
        self.transform_function = transform_function
        self.activation_memory = [{i:0 for i in range(len(self.nodes))}]

    def __str__(self): return '::'.join(str(n) for n in self.nodes)

    def accept_input(self, input_vector):
        from collections import defaultdict

        # remember all activations by node index.
        activations = defaultdict(lambda:0)

        # input values are easy.
        for vi,v in enumerate(input_vector): activations[vi] = v

        for linkin,linkoutlist in self.links:
            # Sequentally calculate and remember the activations for all non-input nodes.
            # Only process a node once.
            for node_index in linkoutlist:
                if node_index not in activations:
                    v = tuple(activations[i] for i in linkin)
                    node = self.nodes[node_index]
                    activations[node_index] = node.accept_input(v, self.transform_function)

        self.activation_memory.append(activations)
        return (activations[i] for i in self.output_node_ixs)

    def train(self, reporter, tset, error_precision, learning_epsilon, learning_acceleration, max_iterations, training_strategy):
        def run_pattern(V, T):
            """
                Execute one training pattern in the training set. Calculate and save learning values.
                Report state of all nodes. Train if specified by training_strategy. Return error.

                V: input vector
                T: expected output(s) of system.
            """
            from collections import defaultdict

            self.accept_input(V)
            activations = self.activation_memory[-1] # these are the outputs from this run.

            delta_error = 0

            # calculate change rate for output nodes
            change_rate = defaultdict(lambda:0)
            for node_index in self.output_node_ixs:
                a = activations[node_index]
                diff = T[node_index] - a
                delta_error += diff
                change_rate[node_index] = diff if a in (0,1) else diff * a * (1 - a)
                self.nodes[node_index].remember_deltas(learning_epsilon, change_rate[node_index], learning_acceleration)

            # calculate change rate for hidden nodes (dr for output nodes required)
            for node_index in sorted(self.output_paths.keys(), reverse=True):
                if node_index not in self.nodes: continue ## Must be a raw input node.

                path = self.output_paths[node_index]
                a = activations[node_index]
                change_rate[node_index] = sum(change_rate[step["tonode"]] * self.nodes[step["tonode"]].weights[step["atposition"]] for step in path) * a * (1 - a)
                self.nodes[node_index].remember_deltas(learning_epsilon, change_rate[node_index], learning_acceleration)

            reporter.report_pattern(activations, T, change_rate, 
                                    {ni:n.weights+[n.theta] for ni,n in self.nodes.items()}, 
                                    # perm memory is a tuple of weights, theta
                                    # index -1 points to most recent memory.
                                    {ni:n.perm_delta_memory[-1][0]+[n.perm_delta_memory[-1][1]] for ni,n in self.nodes.items()})

            if training_strategy == TrainingStrategy.PATTERN:
                for node in self.nodes.values():
                    node.apply_deltas()

            return (delta_error * delta_error)

        EPSILON = math.pow(10, -error_precision)

        epochs = []
        epoch = 0
        trained = False
        while not trained and epoch < max_iterations:
            error = 0
            reporter.start_epoch()
            for V, T in tset:
                error += run_pattern(V, T)

            reporter.end_epoch(error)

            trained = round(math.fabs(error), error_precision) <= EPSILON
            if training_strategy == TrainingStrategy.EPOCH and not trained:
                for node in self.nodes.values():
                    node.apply_deltas()

            epoch += 1

        return trained
