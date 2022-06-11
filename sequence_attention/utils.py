import os
import random
import pandas as pd
import numpy as np
import pickle
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

def fna_to_dict(sample_id, label, in_dir, out_dir):
    '''
    convert fna file to dictionary
    '''
    
    out_name = '{}/{}/{}.pkl'.format(out_dir, label, sample_id)
    filename = '{}/{}.fna'.format(in_dir, sample_id)
    
    meta_data_read_list = []
    read = ''
    f_sample = open(filename)
    read_dict = {}
    for line in f_sample:
        if line[0] == '>':
            if len(read) != 0:
                read_dict[header] = read
                meta_data_read_list.append(header)
                read = ''
            header = line[1:].strip()
        else:
            read += line.strip()
    if len(read) != 0:
        read_dict[header] = read
        meta_data_read_list.append(header)
    f_sample.close()
    pickle.dump(read_dict, open(out_name, 'wb'))
    return meta_data_read_list

# ESTE NO
def split_sample(sample_id, label, in_dir, out_dir):
    '''
    split reads in a sample into seperate files
    '''
    
    file_dir = '{}/{}/{}'.format(out_dir, label, sample_id)
    filename = '{}/{}.fna'.format(in_dir, sample_id)
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    meta_data_read_list = []
    read = ''
    f_sample = open(filename)
    for line in f_sample:
        if line[0] == '>':
            if len(read) != 0:
                with open('{}/{}/{}/{}.fna'.format(out_dir, label, sample_id, header), 'w+') as f_read:
                    f_read.write(read)
                    meta_data_read_list.append(header)
                read = ''
            read += line
            header = line[1:].strip()
        else:
            read += line
    if len(read) != 0:
        with open('{}/{}/{}/{}.fna'.format(out_dir, label, sample_id, header), 'w+') as f_read:
            f_read.write(read)
            meta_data_read_list.append(header)
    f_sample.close()
    return meta_data_read_list

# ESTE YA NO
def train_test_split(meta_data, train_size_per_class):
    '''
    train test split
    '''
    label_list = sorted(meta_data['label'].unique())
    sample_by_class = {label: meta_data[meta_data['label']==label]['sample_id'].tolist() for label in label_list}
    
    train = []
    test = []
    
    for cls in sample_by_class:
        tmp_list = random.sample(sample_by_class[cls], train_size_per_class)
        train.extend(tmp_list)
        test_list = [sid for sid in sample_by_class[cls] if sid not in train]
        test.extend(test_list)
    
    partition = {'train': train, 'test': test}
    
    return partition

# NUEVA
def select_num_samples(meta_data, number_samples_per_class):
    
    label_list = sorted(meta_data['label'].unique()) #ej: ['CD', 'Not-CD']
    sample_by_class = {label: meta_data[meta_data['label']==label]['sample_id'].tolist() for label in label_list} #ej: {'CD': ['ERR1368879', ...], 'Not-CD': ['ERR1368881', ...]}
    
    data = []
    
    for cls in sample_by_class: # cls = CD, Not-CD
        tmp_list = random.sample(sample_by_class[cls], number_samples_per_class) # 'number_samples_per_class' muestras de cada clase
        data.extend(tmp_list)
    
    return data

# ESTE NO
def preprocess_data(opt):
    '''
    preprocessing the data
    '''
    
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    
    meta_data = pd.read_csv(opt.meta_data_file, dtype='str')
    
    label_list = sorted(meta_data['label'].unique())
    
    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx
        
    pickle.dump(label_dict, open('{}/label_dict.pkl'.format(opt.out_dir), 'wb'))
    
    for label in label_list:
        label_dir = '{}/{}'.format(opt.out_dir, label)
        if os.path.exists(label_dir):
            continue
        os.makedirs(label_dir)
        
    read_meta_data = {}
    STEP = meta_data.shape[0] // 10 if meta_data.shape[0] > 10 else 1
    for idx in range(meta_data.shape[0]):
        if idx % STEP == 0:
            logging.info('Processing raw data: {:.1f}% completed.'.format(10 * idx / STEP))
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']

        read_meta_data[sample_id] = split_sample(sample_id, label, opt.in_dir, opt.out_dir)
    
    min_num_sample = min([meta_data[meta_data['label']==label].shape[0] for label in label_list])
    
    num_train_samples_per_cls = opt.num_train_samples_per_cls
    num_train_samples_per_cls = num_train_samples_per_cls if num_train_samples_per_cls < min_num_sample else min_num_sample
    
    partition = train_test_split(meta_data, num_train_samples_per_cls)
    
    
    sample_to_label = {}
    for idx in range(meta_data.shape[0]):
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']
        sample_to_label[sample_id] = label

    pickle.dump([sample_to_label, read_meta_data], open('{}/meta_data.pkl'.format(opt.out_dir), 'wb'))
    
    train_list = []
    for sample_id in partition['train']:
        train_list.extend(['{}/{}/{}/{}.fna'.format(opt.out_dir, sample_to_label[sample_id], sample_id, read_id) for read_id in read_meta_data[sample_id]])
    test_list = []
    for sample_id in partition['test']:
        test_list.extend(['{}/{}/{}/{}.fna'.format(opt.out_dir, sample_to_label[sample_id], sample_id, read_id) for read_id in read_meta_data[sample_id]])
    
    read_partition = {'train': train_list, 'test': test_list}
    pickle.dump(read_partition, open('{}/train_test_split.pkl'.format(opt.out_dir), 'wb'))    
    
# SE DIVIDIÓ LA FUNCIONALIDAD ORIGINAL (ahora dos funciones: preprocess_data_pickle + select_data)
def preprocess_data_pickle(opt):
    '''
    preprocessing the data
    '''
    # si no existe el out_dir, lo creea
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    
    # lee meta_data.csv (las clasificaciones reales por muestra) y las ordena alfabéticamente en función de la label (CD, Not-CD)
    meta_data = pd.read_csv(opt.meta_data_file, dtype='str')
    label_list = sorted(meta_data['label'].unique())
    
    # label_dict:   {'CD': 0, 'Not-CD': 1}
    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx

    pickle.dump(label_dict, open('{}/label_dict.pkl'.format(opt.out_dir), 'wb'))
    
    # crea un directorio para cada label (CD, Not-CD) en el output_dir
    # en él se guardará un .pickle por cada muestra (.fna) clasificada como esa label
    for label in label_list:
        label_dir = '{}/{}'.format(opt.out_dir, label)
        if os.path.exists(label_dir):
            continue
        os.makedirs(label_dir)
    
    read_meta_data = {}
    STEP = meta_data.shape[0] // 10 if meta_data.shape[0] > 10 else 1  # tamaño csv > 10 ? STEP = tamaño del csv / 10 (y se queda con la parte entera de la división) : STEP = 1
    for idx in range(meta_data.shape[0]):
        if idx % STEP == 0: # módulo
            logging.info('Processing raw data: {:.1f}% completed.'.format(10 * idx / STEP))

        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']
        read_meta_data[sample_id] = fna_to_dict(sample_id, label, opt.in_dir, opt.out_dir)
    
    # crear un .pickle con los datos de meta_data.csv y guardarlo en el out_dir
    # read_meta_data:   {'ERR1368879': ['ERR1368879.1 1939.100001_0 length=175', 'ERR1368879.2 1939.100001_1 length=175', ...], ...}
    sample_to_label = {}
    for idx in range(meta_data.shape[0]):
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']
        sample_to_label[sample_id] = label

    pickle.dump([sample_to_label, read_meta_data], open('{}/meta_data.pkl'.format(opt.out_dir), 'wb'))

# NUEVA
def select_data_pickle(opt):
    meta_data = pd.read_csv(opt.meta_data_file, dtype='str')
    label_list = sorted(meta_data['label'].unique()) #ej: ['CD', 'Not-CD']

    #label_dict = pickle.load(open('{}/label_dict.pkl'.format(opt.out_dir), 'rb')) 
    sample_to_label, read_meta_data = pickle.load(open('{}/meta_data.pkl'.format(opt.out_dir), 'rb'))

    # escoger el número de MUESTRAS de cada tipo
    #       escoger el número establecido en la configuración, o si este es muy pequeño, 
    #       el número de muestras de la clase con menos muestras
    min_num_sample = min([meta_data[meta_data['label']==label].shape[0] for label in label_list])
    num_train_samples_per_cls = opt.num_train_samples_per_cls
    num_train_samples_per_cls = num_train_samples_per_cls if num_train_samples_per_cls < min_num_sample else min_num_sample
    
    #data = select_num_samples(meta_data, num_train_samples_per_cls)
    sample_by_class = {label: meta_data[meta_data['label']==label]['sample_id'].tolist() for label in label_list} #ej: {'CD': ['ERR1368879', ...], 'Not-CD': ['ERR1368881', ...]}
    
    data = []
    for cls in sample_by_class: # cls = CD, Not-CD
        tmp_list = random.sample(sample_by_class[cls], num_train_samples_per_cls)
        data.extend(tmp_list)
    
    read_list = []          # todas las secuencias
    read_list_selected = [] # solo las secuencias seleccionadas
    for sample_id in data:
        read_list.extend([('{}/{}/{}.pkl'.format(opt.out_dir, sample_to_label[sample_id], sample_id), read_id) for read_id in read_meta_data[sample_id]])

    pickle.dump(read_list, open('{}/sequence_list.pkl'.format(opt.out_dir), 'wb')) # fichero con todas las secuencias de las muestras escogidas
    print("sequence_list   done")

    # escoger el número de SECUENCIAS de cada muestra    
    for sample_id in read_meta_data:
        num_reads = min(opt.num_train_reads_per_sample, len(read_meta_data[sample_id]))
        tmp_list = random.sample(read_meta_data[sample_id], num_reads)
        filter_set = set(tmp_list)
        read_list_selected.extend([tuple for tuple in read_list if tuple[1] in filter_set])
    
    pickle.dump(read_list_selected, open('{}/sequence_list_selected.pkl'.format(opt.out_dir), 'wb')) # fichero solo con las secuencias escogidas
    print("sequence_list_selected   done")

 ################ LAS MÍAS ################

def save_text_array(file, elements): # file =  path + file_name
    with open(file, mode='wt', encoding='utf-8') as myfile:
        for e in elements:
            myfile.write(e)
            myfile.write('\n')

def save_numpy_array(file, array):
    np.savetxt(file, array, delimiter=",")