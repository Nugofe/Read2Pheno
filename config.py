import tensorflow as tf
class Config:
    '''
    Model configuration:
    '''
    ## ======= parameters required ======= ##
    in_dir = '../crons_disease/sra_download/samples_fna' # directory of raw data
    out_dir = '../crons_disease/results/CNNclassifier' # directory of processed data

    # For the Gevers data set, 20, 80 and 200 samples per class are randomly selected 
    # as training data (resulting in 40, 160 and 400 samples total respectively).
    num_train_samples_per_cls = 10 # number of training sample per class

    SEQLEN = 100 # sequence length
    BASENUM = 4 # dimension of input sequence (e.g., there are 4 major bases for DNA sequence input, therefore BASENUM is 4)
    Ty = 2 # number of target classes (Ty >= 2)
    save_model_path = '../crons_disease/results/CNNclassifier'  # path to the saved model directory
    
    n_workers = 1
    ## ======= parameters required ======= ##
    
    ## ======= parameters optional ======= ##
    device = "gpu" if tf.test.is_gpu_available() else "cpu"
    
    if_cnn = 1 # if_cnn = {0: no cnn, 1: cnn+resnet     resnet=Residual Neural Network: https://en.wikipedia.org/wiki/Residual_neural_network}
    n_cnn_layer = 1 # number of cnn/resnet blocks                
    n_cnn_filters = 256 # number of output channels 
    cnn_window = 9 # kernel size for convolutional layer

    if_lstm = 2                
    #if_lstm = 1 # if lstm = {0: no lstm, 1: just lstm, 2: bi-lstm}
    n_lstm_node = 128 # number of hidden nodes in LSTM 
                    
    att_n_layer = 1 # number of hidden layers
    att_n_node = 16 # number of hidden nodes
    
    fc_n_layer = 1
    #fc_n_layer = 0 # number of additional fully connected layers
    fc_n_node = 32 # number of hidden nodes in additional fully connected layers 
    drop_out_rate = 0 # dropout rate 
    
    opt_lr = 0.001
    #opt_lr = 0.005 # learning rate
    opt_decay = 0 # learning rate decay
    
    batch_size = 1024 # number of sequences used to train a batch
    epochs = 10
    #epochs = 1 # number of epochs to train
    shuffle = True
    verbose = 1 # {0: silence, 1: show progress}
    ## ======= parameters optional ======= ##