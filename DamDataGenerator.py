import numpy as np
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=256, imsize=(266,266,4), shuffle_seed=444, noise_std = 0, data_path='/nfs/nhome/live/glindsay/larger_damCV/images/', test=False,val=True):
 
        self.batch_size = batch_size
        self.test = test
        self.imsize=imsize
        self.noise_std = noise_std
        self.val = False if test else val
   
        self.labels = np.load(data_path+'labels.npy')

        if self.test:
            im_IDs = np.load(data_path+'testims_inds.npy')
            self.im_path = data_path+'test_images/'
        else:
            im_IDs = np.load(data_path+'trainims_inds.npy')
            self.im_path = data_path+'train_images/'

        self.num_ims = len(im_IDs)

        np.random.seed(shuffle_seed)
        np.random.shuffle(im_IDs)
        self.im_IDs = im_IDs


        if self.val:
            self.val_IDs = self.im_IDs[:batch_size]
            self.im_IDs = self.im_IDs[batch_size:]
            self.num_ims = len(self.im_IDs)

        self.batch_per_epoch = self.__len__()
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_ims / self.batch_size))

    def get_valbatch(self):
        X, y = self.__data_generation(self.val_IDs)
        X= X[:,:self.imsize[0],:self.imsize[1],:self.imsize[2]]
        return X,  y

    def get_testbatch(self,index=0):
        if self.test:
          if self.batch_size*(index+1) <= self.num_ims:
            X, y = self.__data_generation(self.im_IDs[self.batch_size*index:self.batch_size*(index+1)])
            return X,  y
          else:
            print('Exceeds number of test images') #todo: make this an error message
            return None
        else:
            print('Test set not loaded') #todo: make this an error message
            return None

    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.im_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        ##self.batch_id = 0
        self.indexes = np.arange(len(self.im_IDs))
        np.random.shuffle(self.indexes)

    def label_convert(self,l):
        if l == 'P' or l == 'I':
            return 1
        elif l == 'A':
            return 0
        else:
            print('Incorrect label'); return None #TODO: make error

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,)+self.imsize)
        y = np.zeros((self.batch_size), dtype=int)

        noise = np.random.randn(X.shape[0],X.shape[1],X.shape[2],X.shape[3])*self.noise_std
        noise[np.random.choice(self.batch_size,size=(int(self.batch_size*.45)),replace=False),:,:,:] = 0

        #training set mean and std
        r = [0.33066305322913947, 0.1417965275302205]
        g = [0.3291309715529797, 0.11527820030273506]
        b = [0.280864934417904, 0.0973566175910353]
        ir = [0.45477920985106823, 0.13050297084409782]


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im = np.load(self.im_path + 'im' + "%05d" % ID + '_l'+str(self.labels[ID]) + '.npy')
            X[i,] = im[:self.imsize[0],:self.imsize[1],:self.imsize[2]]/255. #basic crop because some have extra pixels
            X[i,:,:,0] = (X[i,:,:,0]-r[0])/r[1]
            X[i,:,:,1] = (X[i,:,:,1]-g[0])/g[1]
            X[i,:,:,2] = (X[i,:,:,2]-b[0])/b[1]
            X[i,:,:,3] = (X[i,:,:,3]-ir[0])/ir[1]

            # Store class
            y[i] = self.label_convert(self.labels[ID])

        X += noise
        return X, y
