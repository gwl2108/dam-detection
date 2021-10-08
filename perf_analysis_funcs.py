import tensorflow as tf
import DamDataGenerator as data
#import focalloss_funcs as floss
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os


loss_dir = '/nfs/nhome/live/glindsay/larger_damCV/nets/losses/'
fv_dir = '/nfs/nhome/live/glindsay/larger_damCV/nets/firstvals/'

model_dir='/nfs/nhome/live/glindsay/larger_damCV/nets/'
fig_dir = '/nfs/nhome/live/glindsay/larger_damCV/figs/' 
#act_dir = '/nfs/nhome/live/glindsay/virtual_rodent/activity/'
#export_dir = '/nfs/nhome/live/glindsay/damCV/saved_model' 


def plot_training(net):
    train_loss = []
    val_loss = []
       
    filename = [f for f in os.listdir(loss_dir) if net in f][0]
    net = filename[:-10]
    with open(loss_dir+filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            losses = r['epoch;loss;val_loss'].split(';')
            train_loss.append(float(losses[1]))
            val_loss.append(float(losses[2]))
    plt.title(str(np.load(fv_dir+net+'Firstval.npy')))
    plt.plot(np.arange(1,len(train_loss)+1),train_loss)
    plt.plot(np.arange(1,len(train_loss)+1),val_loss); plt.legend(['train','val'])
    plt.savefig(fig_dir+net+'_train.png'); plt.close()

def do_perf_analysis(net,batch_size=660):

    filename = [f for f in os.listdir(loss_dir) if net in f][0][:-10]
    #alpha = float(filename.split('_al')[1].split('_')[0])
    #gamma = float(filename.split('_gm')[1].split('_')[0])
    reconstructed_model = tf.keras.models.load_model(model_dir+filename) #, custom_objects={ 'binary_focal_loss_fixed': floss.binary_focal_loss(alpha=alpha, gamma=gamma) })
    
    DataGen = data.DataGenerator(batch_size=batch_size,test=True)
    inputs, labels = DataGen.get_testbatch() # #__getitem__(1) #
    loss = reconstructed_model.evaluate((inputs,labels),verbose=0)
    print('Test loss for '+net+' is '+str(loss))

    outputs = reconstructed_model(inputs,training=False).numpy()
    #need to do recall, prec etc here
    #hit rate is number correctly caught positives / total positives in test set
    #selectivity is same for negatives
    #chance accuracy is 85.7% on test set
    print(outputs); print(outputs>.5); #print(labels)
    pos = np.where(outputs>=.5)[0]
    neg = np.where(outputs<.5)[0]
    cpos = np.where(labels==1)[0]
    cneg = np.where(labels==0)[0]
    TP = len(np.intersect1d(pos,cpos))
    TN = len(np.intersect1d(neg,cneg))
    FP = len(np.intersect1d(pos,cneg))
    FN = len(np.intersect1d(neg,cpos))

    print(net)
    print('accuracy: ',(TP+TN)/(TP+FP+TN+FN))
    print('hit rate: ',TP/(TP+FN))
    print('precision: ',TP/(TP+FP))
    print('selectivity: ',TN/(TN+FP))

    accs = []
    HRs = []
    precs = []
    sels = []
    for thresh in range(0,105,5):
        pos = np.where(outputs>=thresh/100)[0]
        neg = np.where(outputs<thresh/100)[0]
        TP = len(np.intersect1d(pos,cpos))
        TN = len(np.intersect1d(neg,cneg))
        FP = len(np.intersect1d(pos,cneg))
        FN = len(np.intersect1d(neg,cpos))
        accs.append((TP+TN)/(TP+FP+TN+FN))
        HRs.append(TP/(TP+FN))
        try:
            precs.append(TP/(TP+FP))
        except: 
            precs.append(np.nan)
        sels.append(TN/(TN+FP))

    
    plt.plot(accs); plt.plot(HRs); plt.plot(precs); plt.plot(sels); plt.legend(['Acc','HR','Prec','Sel']); plt.title(filename);
    plt.xticks(np.arange(0,20,2),np.arange(0,1,.1))
    plt.savefig(fig_dir+'ROC_'+net);  plt.close()

