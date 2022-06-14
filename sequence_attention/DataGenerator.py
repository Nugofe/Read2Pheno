import pickle
import keras
import numpy as np

base2vec = {'A': np.array([1,0,0,0]),
            'C': np.array([0,1,0,0]),
            'G': np.array([0,0,1,0]),
            'T': np.array([0,0,0,1]),
            'U': np.array([0,0,0,1]),
            'W': np.array([0.5,0,0,0.5]),
            'S': np.array([0,0.5,0.5,0]),
            'M': np.array([0.5,0.5,0,0]),
            'K': np.array([0,0,0.5,0.5]),
            'R': np.array([0.5,0,0.5,0]),
            'Y': np.array([0,0.5,0,0.5]),
            'B': np.array([0,1/3,1/3,1/3]),
            'D': np.array([1/3,0,1/3,1/3]),
            'H': np.array([1/3,1/3,0,1/3]),
            'V': np.array([0.33,0.33,0.33,0]),
            'N': np.array([0.25,0.25,0.25,0.25]),
            }

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, file_list, sample_to_label, label_dict, batch_size=32, dim=(100, 4), shuffle=True):
        '''Initialization'''
        self.dim = dim
        self.batch_size = batch_size
        self.sample_to_label = sample_to_label
        self.list_IDs = file_list
        self.label_dict = label_dict
        self.n_classes = len(label_dict)
        self.shuffle = shuffle
        self.on_epoch_end()
            
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __seq_mapper(self, seq):
        seq_coded = np.zeros(self.dim)
        for i in range(len(seq)):
            if seq[i] in base2vec:
                seq_coded[i,:] = base2vec[seq[i]]
            else:
                seq_coded[i,:] = base2vec['N']
        return seq_coded
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        if self.n_classes == 2:
            y = np.zeros((self.batch_size,), dtype=int)
        else:
            y = np.zeros((self.batch_size, self.n_classes), dtype=int)
        # Generate data
        for i, file in enumerate(list_IDs_temp):
            # Store sample
            with open(file) as f:
                data = f.read()
                seq = data.split('\n')[1]
            seq_tmp = seq if len(seq) < self.dim[0] else seq[:self.dim[0]]
            seq_coded = self.__seq_mapper(seq_tmp)
            X[i,] = seq_coded
            if self.n_classes == 2:
                y[i] = self.label_dict[self.sample_to_label[file.split('/')[2]]]
            else:
                y[i,self.label_dict[self.sample_to_label[file.split('/')[2]]]] = 1

        return X, y       

# EL NUESTRO
class DataGeneratorPickle(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, file_list, sample_to_label, label_dict, batch_size=32, dim=(100, 4), shuffle=True):
        '''Initialization'''
        self.dim = dim   # dimension:  dim=(opt.SEQLEN,opt.BASENUM)=(100,4)
        self.batch_size = batch_size
        self.sample_to_label = sample_to_label
        self.list_IDs = file_list    # partition['train'] o partition['train']
        self.label_dict = label_dict
        self.n_classes = len(label_dict)
        self.shuffle = shuffle
        self.on_epoch_end() # self.indexes -> índices que identifican las secuencias del file_list
            
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # get all data (NUEVA PARA USAR EL CV)
    def getalldata(self):
        '''Generate all batches of data'''

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        #print('list_IDs_temp !!!!!! ' + str(list_IDs_temp))

        # Generate data containing batch_size samples
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    # get one batch of data (the batch is the index)
    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch (for train or for test - depende de los datos que hayamos pasado)
        #     escoger las secuencias de un batch
        #     se buscan los índices que identifican a las secuencias de ese batch
        # index = 0  =>  indexes = [0*1024:1*1024] = [0:1024]
        # index = 1  =>  indexes = [1*1024:2*1024] = [1024:2048]
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data containing batch_size samples
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        # shuffle de los índices que identifican las secuencias
        # de modo que en cada batch puedan ir secuencias de muestras diferentes
        # (si hubiese shuffle, siempre irían en orden y la gran mayoría de secuencias 
        # del batch serían de la misma muestra)
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __seq_mapper(self, seq):
        'Sequence one-hot encoding'

        seq_coded = np.zeros(self.dim) # solo hace el encoding de la longitud de secuencia indicada
        for i in range(len(seq)):
            if seq[i] in base2vec:
                seq_coded[i,:] = base2vec[seq[i]]
            else:
                seq_coded[i,:] = base2vec['N']
        return seq_coded
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)

        #print("list_IDs_batch !!!!!!!!!!!!!!!!!!!! " + str(list_IDs_temp))

        # Initialization
        X = np.zeros((len(list_IDs_temp), *self.dim)) # array con número de ceros igual al batch_size
        if self.n_classes == 2: # EL NUESTRO
            y = np.zeros((len(list_IDs_temp),), dtype=int)
        else:
            y = np.zeros((len(list_IDs_temp), self.n_classes), dtype=int)
        
        # Generate data
        for i, file in enumerate(list_IDs_temp):
            # Store sample
            sample_file_name = file[0] # ej: ../crohns_disease/results/CNNclassifier/CD/ERR1368879.pkl
            #print(str(i) + " sample_file_name !!!!!!!!!!!!!!!!!!!! " + sample_file_name)
            sample_id = sample_file_name.split('/')[-1][:-4] # ej: ERR1368879
            #print(str(i) + " sample_id !!!!!!!!!!!!!!!!!!!! " + sample_id)
            header = file[1] # ej: ERR1368879.39887 1939.100001_39886 length=175
            #print(str(i) + " header !!!!!!!!!!!!!!!!!!!! " + header)
            read_dict = pickle.load(open(sample_file_name, 'rb'))
            seq = read_dict[header] # la secuencia sin one-hot encode
            #print(str(i) + " seq !!!!!!!!!!!!!!!!!!!! " + seq)
            
            seq_tmp = seq if len(seq) < self.dim[0] else seq[:self.dim[0]]
            seq_coded = self.__seq_mapper(seq_tmp)
            X[i,] = seq_coded
            if self.n_classes == 2: # EL NUESTRO
                y[i] = self.label_dict[self.sample_to_label[sample_id]]
            else:
                y[i,self.label_dict[self.sample_to_label[sample_id]]] = 1

        return X, y # X = secuencias one-hot encoded, y = labels encoded

class DataGeneratorUnlabeled(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, file_list, batch_size=32, dim=(100, 4)):
        '''Initialization'''
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.shuffle = False
        self.on_epoch_end()
            
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __seq_mapper(self, seq):
        seq_coded = np.zeros(self.dim)
        for i in range(len(seq)):
            if seq[i] in base2vec:
                seq_coded[i,:] = base2vec[seq[i]]
            else:
                seq_coded[i,:] = base2vec['N']
        return seq_coded
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        # Generate data
        for i, file in enumerate(list_IDs_temp):
            # Store reads
            with open(file) as f:
                data = f.read()
                seq = data.split('\n')[1]
            seq_tmp = seq if len(seq) < self.dim[0] else seq[:self.dim[0]]
            seq_coded = self.__seq_mapper(seq_tmp)
            X[i,] = seq_coded

        return X   
    
class DataGeneratorUnlabeledPickle(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, file_list, batch_size=32, dim=(100, 4)):
        '''Initialization'''
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.shuffle = False
        self.on_epoch_end()
            
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __seq_mapper(self, seq):
        seq_coded = np.zeros(self.dim)
        for i in range(len(seq)):
            if seq[i] in base2vec:
                seq_coded[i,:] = base2vec[seq[i]]
            else:
                seq_coded[i,:] = base2vec['N']
        return seq_coded
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        # Generate data
        for i, file in enumerate(list_IDs_temp):
            # Store reads
            sample_file_name = file[0]
            sample_id = sample_file_name.split('/')[-1][:-4]
            header = file[1]
            read_dict = pickle.load(open(sample_file_name, 'rb'))
            seq = read_dict[header]
            
            seq_tmp = seq if len(seq) < self.dim[0] else seq[:self.dim[0]]
            seq_coded = self.__seq_mapper(seq_tmp)
            X[i,] = seq_coded

        return X   
