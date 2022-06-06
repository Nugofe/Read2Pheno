import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .sequence_attention_model import sequence_attention_model
from .utils import save_text_array
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

font = {'family': 'sans-serif', # Helvetica
        'size'   : 20}
matplotlib.rc('font', **font)

class SeqAttModel:
    '''
    Attention based sequence model
    '''
    def __init__(self, opt):
        '''
        Model initialization
        '''
        self.opt = opt
        self.model = sequence_attention_model(self.opt)
        self.history = None
        logging.info('Model initialized.')

    def tune_and_eval(self, X, y): # cross validation
        skf = StratifiedKFold(n_splits=self.opt.n_folds, shuffle=True) # CV con k = 10

        p_micro=[] # cada elemento es la métrica calculada en un fold
        p_macro=[]
        r_micro=[]
        r_macro=[]
        f1_micro=[]
        f1_macro=[]
        accuracy=[]
        roc_auc=[]

        for train_index, valid_index in skf.split(X, y): # Generate indices to split data into training and validation set
            print ('\n Evaluation on a new fold is now get started ..')
            X_train=X[train_index,:]
            y_train=y[train_index,:]
            y_class_train=y[train_index]

            X_valid=X[valid_index,:]
            y_valid=y[valid_index,:]
            y_class_valid=y[valid_index]
            
            model = sequence_attention_model(self.opt) # modelo con sus capas, pesos, funciones de activación...
            
            # entrenar el modelo y obtener métricas
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=self.opt.batch_size, epochs=self.opt.epochs, verbose=self.opt.verbose, shuffle=True)
            pred = model.predict_classes(X_valid)
            #eval_loss, eval_p_macro, eval_p_micro, eval_r_macro, eval_r_micro, eval_f1_macro, eval_f1_micro, eval_accuracy, eval_roc = self.model.evaluate(X_valid, y_valid, batch_size=self.opt.batch_size, verbose=self.opt.verbose)

            f1_micro.append(f1_score(y_class_valid, pred, average='micro'))
            f1_macro.append(f1_score(y_class_valid, pred, average='macro'))
            p_micro.append(precision_score(y_class_valid, pred, average='micro'))
            p_macro.append(precision_score(y_class_valid, pred, average='macro'))
            r_micro.append(recall_score(y_class_valid, pred, average='micro'))
            r_macro.append(recall_score(y_class_valid, pred, average='macro'))
            accuracy.append(accuracy_score(y_class_valid, pred))
            roc_auc.append(roc_auc_score(y_class_valid, pred))

        # guardar las métricas obtenidas
        # mean values
        f1mac=np.mean(f1_macro)
        f1mic=np.mean(f1_micro)
        prmac=np.mean(p_macro)
        prmic=np.mean(p_micro)
        remac=np.mean(r_macro)
        remic=np.mean(r_micro)
        maccur=np.mean(accuracy)
        mrocauc=np.mean(roc_auc)
        # std values - desviación típica
        sf1mac=np.std(f1_macro)
        sf1mic=np.std(f1_micro)
        sprmac=np.std(p_macro)
        sprmic=np.std(p_micro)
        sremac=np.std(r_macro)
        sremic=np.std(r_micro)
        saccur=np.std(accuracy)
        srocauc=np.std(roc_auc)
        # table
        #latex_line=' & '.join([str(np.round(x,2))+' $\\pm$ '+str(np.round(y,2)) for x,y in [ [prmic, sprmic], [remic, sremic], [f1mic, sf1mic], [prmac, sprmac], [remac, sremac], [f1mac, sf1mac], [maccur, saccur], [mrocauc, srocauc] ]])      
        
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        pickle.dump([p_micro, r_micro, f1_micro, p_macro, r_macro, f1_macro, accuracy, roc_auc, (loss_values, val_loss_values)], 
                    open('{}/results_cnn.pkl'.format(self.opt.out_dir), 'wb'))

        # guardar los datos más importantes en un formato más leíble
        attributes=['mean_f1_macro: ' + str(f1mac),        'mean_f1_micro: ' + str(f1mic), 
                    'mean_precision_macro: ' + str(prmac), 'mean_precision_micro: ' + str(prmic),
                    'mean_recall_macro: ' + str(remac),    'mean_recall_micro: ' + str(remic),
                    'mean_accuracy: ' + str(maccur),       'mean_roc_auc: ' + str(mrocauc),
                    'std_f1_macro: ' + str(sf1mac),        'std_f1_micro: ' + str(sf1mic), 
                    'std_precision_macro: ' + str(sprmac), 'std_precision_micro: ' + str(sprmic),
                    'std_recall_macro: ' + str(sremac),    'std_recall_micro: ' + str(sremic),
                    'std_accuracy: ' + str(saccur),        'std_roc_auc: ' + str(srocauc)]

        save_text_array(self.opt.out_dir + '/best_metrics.txt', attributes)

        
    def train(self, X, y): # train
        '''
        Model training
        '''
        logging.info('Training started: train the model on {} sequences'.format(X.shape[0]))
        self.history = self.model.fit(X, y, batch_size=self.opt.batch_size, epochs=self.opt.epochs, verbose=self.opt.verbose)
        #train_loss, train_acc = self.model.evaluate(X, y, batch_size=self.opt.batch_size, verbose=self.opt.verbose)
        #logging.info('Training completed: training accuracy is {:.4f}.'.format(train_acc))
    
    def train_generator(self, training_generator, n_workers):
        '''
        Model training with a generator
        '''
        logging.info('Training started:')

        # NOTA: .fit_generator y .evaluate_generator están deprecados. ahora se usan .fit y .evaluate

        # entrenar el modelo batch a batch
        # (.fit_generator porque el model es de la clase Model de keras)
        self.history = self.model.fit_generator(generator=training_generator,
                                                use_multiprocessing=True,
                                                workers=n_workers, verbose=True)

        # evaluar la acurracy del train set
        eval_loss, eval_acc = self.model.evaluate_generator(generator=training_generator,
                                                            use_multiprocessing=True,
                                                            workers=n_workers)
        logging.info('Evaluation completed: evaluation accuracy is {:.4f}.'.format(eval_acc))
        
    def predict(self, X):
        '''
        Model predicting
        '''
        logging.info('Predicting started: predict {} sequences'.format(X.shape[0]))
        pred = self.model.predict(X, batch_size=self.opt.batch_size, verbose=self.opt.verbose)
        logging.info('Predicting completed.')
        return pred
    
    def predict_generator(self, pred_generator, n_workers):
        '''
        Model evaluation with a generator
        '''
        logging.info('Predicting started:')
        pred = self.model.predict_generator(generator=pred_generator,
                                            use_multiprocessing=True,
                                            workers=n_workers, verbose=self.opt.verbose)
        logging.info('Predicting completed.')
        return pred
        
    def evaluate(self, X, y): # test
        '''
        Model evaluation
        '''
        logging.info('Evaluation started: evalute {} sequences'.format(X.shape[0]))
        eval_loss, eval_p_macro, eval_p_micro, eval_r_macro, eval_r_micro, eval_f1_macro, eval_f1_micro, eval_accuracy, eval_roc = self.model.evaluate(X, y, batch_size=self.opt.batch_size, verbose=self.opt.verbose)
        #logging.info('Evaluation completed: evaluation accuracy is {:.4f}.'.format(eval_acc))
        return [eval_p_macro, eval_p_micro, eval_r_macro, eval_r_micro, eval_f1_macro, eval_f1_micro, eval_accuracy, eval_roc]
    
    def evaluate_generator(self, test_generator, n_workers):
        '''
        Model evaluation with a generator
        '''
        logging.info('Evaluation started:')

        # evaluar la acurracy del test set
        eval_loss, eval_acc = self.model.evaluate_generator(generator=test_generator,
                                                            use_multiprocessing=True,
                                                            workers=n_workers)
        logging.info('Evaluation completed: evaluation accuracy is {:.4f}.'.format(eval_acc))
        return eval_acc
    
    def save(self, name):
        '''
        Model save
        '''
        self.model.save_weights(name)
        
    def load(self, name):
        '''
        Model load
        '''
        self.model.load_weights(name)
        
    def _extract_intermediate_output(self):
        '''
        helper function to extract intermediate layer output
        '''
        inputs = []
        inputs.extend(self.model.inputs)
        outputs = []
        outputs.extend(self.model.outputs)

        attention = self.model.get_layer('att_weights')
        embedding = self.model.get_layer('att_emb')

        outputs.append(attention.output)
        outputs.append(embedding.output)

        extract_intermediate_layer = K.function(inputs, outputs)
        return extract_intermediate_layer
    
    def extract_weigths(self, X):
        '''
        extract sequence attention weigths and sequence embedding from the model for input sequences, X.
        '''
        logging.info('Weights extraction started: extract weights for {} sequences'.format(X.shape[0]))
        extract_intermediate_layer = self._extract_intermediate_output()
        prediction, attention_weights, sequence_embedding = extract_intermediate_layer([X])
        logging.info('Weights extraction completed.')
        return prediction, attention_weights, sequence_embedding
    
    def visualize_training_history(self):
        '''
        Visualize training history
        '''
        if self.history is None:
            logging.info('Model training history does not exist.')
            return
        plt.figure(figsize=(8, 6))
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['loss'], 'o--', color='k', label='val_loss')
            plt.title('Validation dataset loss')
        else:
            plt.plot(self.history.history['loss'], 'o--', color='k', label='train_loss')
            plt.title('Training dataset loss')

        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()
