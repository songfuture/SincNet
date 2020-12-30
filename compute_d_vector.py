# compute_d_vector.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# Feb 2019

# Description: 
# This code computes d-vectors using a pre-trained model
 

import os
import soundfile as sf
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dnn_models import MLP
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf_inp,str_to_bool
import sys

# Model to use for computing the d-vectors
model_file='/home/mirco/sincnet_models/SincNet_TIMIT/model_raw.pkl' # This is the model to use for computing the d-vectors (it should be pre-trained using the speaker-id DNN)
cfg_file='/home/mirco/SincNet/cfg/SincNet_TIMIT.cfg' # Config file of the speaker-id experiment used to generate the model
te_lst='data_lists/TIMIT_test.scp' # List of the wav files to process
out_dict_file='d_vect_timit.npy' # output dictionary containing the a sentence id as key as the d-vector as value
data_folder='/home/mirco/Dataset/TIMIT_norm_nosil'

avoid_small_en_fr=True
energy_th = 0.1  # Avoid frames with an energy that is 1/10 over the average energy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = None

# Reading cfg file
options=read_conf_inp(cfg_file)


#[data]
pt_file=options.pt_file
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te)


# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
    
# loss function
cost = nn.NLLLoss()

  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=128


# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.to(device)



DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.to(device)


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.to(device)


checkpoint_load = torch.load(model_file)
CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])



CNN_net.eval()
DNN1_net.eval()
DNN2_net.eval()
test_flag=1 


d_vector_dim=fc_lay[-1]
d_vect_dict={}

   
with torch.no_grad(): 
    
    for i in range(snt_te):
           
         [signal, fs] = sf.read(data_folder+'/'+wav_lst_te[i])
         
         # Amplitude normalization
         signal=signal/np.max(np.abs(signal))
        
         signal=torch.from_numpy(signal).float().to(device).contiguous()
        
         if avoid_small_en_fr: 
             # computing energy on each frame:
             beg_samp=0
             end_samp=wlen
    
             N_fr=int((signal.shape[0]-wlen)/(wshift)) #一条语句所包含的帧数
             Batch_dev=N_fr
             en_arr=torch.zeros(N_fr).float().contiguous().to(device)
             count_fr=0
             count_fr_tot=0
             while end_samp<signal.shape[0]:
                en_arr[count_fr]=torch.sum(signal[beg_samp:end_samp].pow(2)) #各个frame的能量计算=该frame所有采样点的平方和
                beg_samp=beg_samp+wshift
                end_samp=beg_samp+wlen
                count_fr=count_fr+1
                count_fr_tot=count_fr_tot+1
                if count_fr==N_fr:
                    break
    
             en_arr_bin=en_arr>torch.mean(en_arr)*0.1 #判断语句中符合要求的帧数
             en_arr_bin.to(device)
             n_vect_elem=torch.sum(en_arr_bin)
    
             if n_vect_elem<10:
                 print('only few elements used to compute d-vectors')
                 sys.exit(0)



         # split signals into chunks
         beg_samp=0
         end_samp=wlen
         
         N_fr=int((signal.shape[0]-wlen)/(wshift))
         
        
         sig_arr=torch.zeros([Batch_dev,wlen]).float().to(device).contiguous() #语音（输入）分割成batch个chunck
         dvects=Variable(torch.zeros(N_fr,d_vector_dim).float().to(device).contiguous()) #d_vector（输出）又保存为语音帧数个d_vector
         count_fr=0
         count_fr_tot=0
         while end_samp<signal.shape[0]:
             sig_arr[count_fr,:]=signal[beg_samp:end_samp]
             beg_samp=beg_samp+wshift
             end_samp=beg_samp+wlen
             count_fr=count_fr+1
             count_fr_tot=count_fr_tot+1
             if count_fr==Batch_dev: #当对输入的帧数计数达到batch时，输入网络
                 inp=Variable(sig_arr)
                 dvects[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN1_net(CNN_net(inp)) #dvects[0:batch_dev,:]，dvects[batch_dev:2*batch_dev,:]的d_vects得到
                 count_fr=0 #count_fr重新计数，但是count_fr_tot继续累加
                 sig_arr=torch.zeros([Batch_dev,wlen]).float().to(device).contiguous()
         #这里应该是少考虑了语音总帧数除以batch_dev的余数,要不然就是作者设计舍弃剩余的语音帧,
         #inp=Variable(sig_arr[count_fr_tot:count_fr_tot+count_fr])
         #dvects[count_fr_tot:count_fr_tot+count_fr,:]=DNN1_net(CNN_net(inp)) #此时count_fr_tot+count_fr=N_fr
         #当到达语音最后一个采样点，但还未到达最后一个batch时，进入该if。也就是说，语音帧数除以batch_dev的余数。比如总帧数是200，batch_dev是128，还余72帧
         #下面这个语句可以验证一下，对于有余数的情况，如果用下面的语句，那么余数帧的d_vec和语句开始帧的d_vec应该是重复的
         if count_fr>0: #当总语音帧数小于batch_dev时，进入该if语句，比如总帧数是50，batch_dev
          inp=Variable(sig_arr[0:count_fr]) 
          dvects[count_fr_tot-count_fr:count_fr_tot,:]=DNN1_net(CNN_net(inp))
        
         if avoid_small_en_fr:
             dvects=dvects.index_select(0, (en_arr_bin==1).nonzero().view(-1)) #只选择检测合格帧的d_vects
         
         # averaging and normalizing all the d-vectors
         d_vect_out=torch.mean(dvects/dvects.norm(p=2, dim=1).view(-1,1),dim=0) #一条语音的d_vect为各个帧的d_vect的均值
         
         # checks for nan
         nan_sum=torch.sum(torch.isnan(d_vect_out))

         if nan_sum>0:
             print(wav_lst_te[i])
             sys.exit(0)

         
         # saving the d-vector in a numpy dictionary
         dict_key=wav_lst_te[i].split('/')[-2]+'/'+wav_lst_te[i].split('/')[-1]
         d_vect_dict[dict_key]=d_vect_out.cpu().numpy()
         print(dict_key)

# Save the dictionary
np.save(out_dict_file, d_vect_dict)
         
    
    



