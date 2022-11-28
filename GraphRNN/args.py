### program configuration
class Args():
    def __init__(self):
        f = open('experiment_template.txt')
        params = f.readlines() 
        parameters = []
        for p in params:
            p = p.strip('\n')
            print(p)
            parameters.append(p)
        
        default_parameters = ['enzymes_small', None, None, 16, 8, 32, 16, 32, 32, 1000, 4, 4, 32, 3000, 100, 100, 100, 100, 0.003, [400, 1000], 0.3, 2, './', False, 3000, True, 'BA', 'clustering']
        
        if (len(parameters) > 28):
            if (len(parameters.pop()) > 28):
                parameters = parameters.pop()
                print("ERROR: Malformed input file! Please retry!")
            else:
                print("Valid input file, had to strip trailing newline. Retrying...\n")
                print("Valid input file with ", parameters)
                print("Starting args.py...")

        else:
            print("Valid input file with ", parameters)
            print("Starting args.py...")
        ### if clean tensorboard
        self.clean_tensorboard = False
        ### Which CUDA GPU device is used for training - this is typically ZERO or ONE
        ### If you set the wrong CUDA device ID, torch will report that your device does not even have a CUDA compatible GPU
        self.cuda = 0
        
        ### Which GraphRNN model variant is used.
        # The simple version of Graph RNN
        # self.note = 'GraphRNN_MLP'
        # The dependent Bernoulli sequence version of GraphRNN
        # self.note = 'GraphRNN_RNN'

        # The version that incorporates node labels into the prediction task.
        self.note = 'GraphRNN_labelRNN'
        ## for comparison, removing the BFS compoenent
        # self.note = 'GraphRNN_MLP_nobfs'
        # self.note = 'GraphRNN_RNN_nobfs'

        ### Which dataset is used to train the model
        self.graph_type = parameters[0] if '[' not in parameters[0] else default_parameters[0]
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)

        # if none, then auto calculate
        self.max_num_node = parameters[1] if '[' not in parameters[1] else default_parameters[1] # max number of nodes in a graph
        self.max_prev_node = parameters[2] if '[' not in parameters[2] else default_parameters[2] # max previous node that looks back
        self.num_node_labels = None  # number of distinct node labels


        ### network config
        ## GraphRNN
        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        self.hidden_size_rnn_output = parameters[3] if '[' not in parameters[3] else default_parameters[3] # hidden size for output RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_rnn_output = parameters[4] if '[' not in parameters[4] else default_parameters[4] # the embedding size for output rnn
        self.embedding_size_output = int(64/self.parameter_shrink) # the embedding size for output (VAE/MLP)
    
        # Configure LSTM arguments
        self.embedding_size_lstm = parameters[5] if '[' not in parameters[5] else default_parameters[5] # lstm embedding size set to default value
        self.node_embedding_size = 4  # the embedding size for the training node labels
        
        self.hidden_size = parameters[6] if '[' not in parameters[6] else default_parameters[6] # Set hidden value to default for now...
        # End configuration of LSTM arguments

        self.batch_size = parameters[7] if '[' not in parameters[7] else default_parameters[7]  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = parameters[8] if '[' not in parameters[8] else default_parameters[8]
        self.test_total_size = parameters[9] if '[' not in parameters[9] else default_parameters[9]
        self.num_layers = parameters[10] if '[' not in parameters[10] else default_parameters[10]

        ### training config
        self.num_workers = parameters[11] if '[' not in parameters[11] else default_parameters[11] # num workers to load data, default 4
        self.batch_ratio = parameters[12] if '[' not in parameters[12] else default_parameters[12] # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = parameters[13] if '[' not in parameters[13] else default_parameters[13] # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = parameters[14] if '[' not in parameters[14] else default_parameters[14]

        self.epochs_test = parameters[15] if '[' not in parameters[15] else default_parameters[15]
        self.epochs_log = parameters[16] if '[' not in parameters[16] else default_parameters[16]
        self.epochs_save = parameters[17] if '[' not in parameters[17] else default_parameters[17]

        self.lr = parameters[18] if '[' not in parameters[18] else default_parameters[18]
        self.milestones = parameters[19] if '[' not in parameters[19] else default_parameters[19]

        self.lr_rate = parameters[20] if '[' not in parameters[20] else default_parameters[20]

        self.sample_time = parameters[21] if '[' not in parameters[21] else default_parameters[21] # sample time in each time step, when validating

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = str(parameters[22] if '[' not in parameters[22] else default_parameters[22])
        self.model_save_path = self.dir_input+'model_save/' # only for nll evaluation
        self.graph_save_path = self.dir_input+'graphs/'
        self.figure_save_path = self.dir_input+'figures/'
        self.timing_save_path = self.dir_input+'timing/'
        self.figure_prediction_save_path = self.dir_input+'figures_prediction/'
        self.nll_save_path = self.dir_input+'nll/'


        self.load = parameters[23] if '[' not in parameters[23] else default_parameters[23] # If load model, default lr is very low
        self.load_epoch = parameters[24] if '[' not in parameters[24] else default_parameters[24]
        self.save = parameters[25] if '[' not in parameters[25] else default_parameters[25]


        ### baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = parameters[26] if '[' not in parameters[27] else default_parameters[26]

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = parameters[27] if '[' not in parameters[27] else default_parameters[27]


        ### filenames to save intemediate and final outputs
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline

#a = Args() # Uncomment to test for object initialization bugs (related to parser)
