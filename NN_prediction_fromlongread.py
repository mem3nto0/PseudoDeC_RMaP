#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from Bio import SeqIO

"""
///////////////////////////
Run this file to analyse the test long read. After generating the long reads
from the file "Create_fulllength_testdata_from_tombo.py, this code can be used
to create the full prediction of the reads. 
/////////////////////////
"""

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)

# load the NN model

model = tf.keras.models.load_model("model/RNA_2inp-Inception_BCE_challange_3")

#%%

path = os.getcwd()
path1 = path + "/folder_longread_preprocessed" # choose the folder where preprocessed test data were saved
raw_seg_len = 800
files_list = os.listdir(path1)
spacer_pos = 0
max_seq_len = 80
labels = 4 # labels is a reppresentation of "A,C,G,T"
base_dict = { 1:"A", 2:"C", 3:"G", 4:"T"}
N_miss = 0
save_path = path + "/save_prediction" # save path where each prediction will be saved

# ////// read the reference from a .fasta file ////////////

reference_path = "path/your_fasta.fasta"

references = list(SeqIO.parse(reference_path, "fasta"))
reference = references[0].seq

reference_track_mod = np.zeros([len(reference), 2])     

for i in range (len(files_list)): #len(files_list)

    try:

        with np.load(path1 + "/" + files_list[i]) as data:

            Raw_signal =data["raw_signal_tombo"] # raw-signal analysed by tombo
            sequence_tombo = data["seq_tombo"]
            one_hot = data["One_hot_tombo"] # one hot reppresentation of the tombo basecalling
            name_id = data["id_name"] # name of the file
            position_adjusting = data["reference_start"] # name of the file
            
        #normlaize the raw signal
        median = np.median(Raw_signal)
        std = np.median(np.abs(Raw_signal-median))*1.4826 + np.finfo(np.float32).eps
        Raw_signal = (Raw_signal - median)/std

        #generate input data for the Neural netowrk

        N_segments = int(len(Raw_signal)/raw_seg_len)

        Input_1 = np.zeros([N_segments +1,raw_seg_len])            # initialize the first input of the NN
        Input_2 = np.zeros([N_segments +1,max_seq_len,labels])     # initialize the second input of the NN

        for j in range(N_segments):

            start = j*raw_seg_len

            Input_1[j] = Raw_signal[start: start + raw_seg_len]

            window_one_hot = one_hot[start: start + raw_seg_len,:]
            probe = np.argmax(window_one_hot, axis = -1)
            probe = probe[probe != spacer_pos]
            probe = probe - 1 # minus one for rescaling the results between 0-3 (4 labels)

            for kk in range (len(probe)):

                Input_2[j,kk, probe[kk]] = 1 
        
        #find the number of point not overlapping
        not_overlaping_last_seg = len(Raw_signal) - (start + 800)

        # the extention to +1 is for keeping the full dimention of the output
        Input_1[N_segments] = Raw_signal[-800:]

        Additional_window = one_hot[-800:,:]
        probe = np.argmax(Additional_window, axis = -1)
        probe = probe[probe != spacer_pos]
        probe = probe -1

        for kk in range (len(probe)):

            Input_2[N_segments, kk, probe[kk]] = 1 

        #probe the overlapping bases for the last segment
        Window_overlap = one_hot[-800:-not_overlaping_last_seg,:]
        seq_overlap = np.zeros([Window_overlap.shape[0],4])
        probe = np.argmax(Window_overlap, axis = -1)
        probe = probe[probe != spacer_pos]
        probe = probe - 1

        for kk in range (len(probe)):

            seq_overlap[kk, probe[kk]] = 1 

        seq_overlap = np.sum(seq_overlap, axis = 1)
        seq_overlap = np.where(seq_overlap > 0.5)[0] 
        len_overlap = len(seq_overlap)

        X_total ={"Input_1": Input_1, "Input_2": Input_2}

        prediction = model.predict(X_total)

        # reconstruct the final output removing the null part of the predictions
        Final_seq_binary = []
        Final_seq_score = []

        for kk in range(N_segments): #

            full_position = np.sum(prediction[kk], axis = 1)
            full_position = np.where(full_position> 0.5)[0]

            real_part =  np.argmax(prediction[kk,:len(full_position)], axis=-1)
            seq_with_score = np.argmax(prediction[kk,:len(full_position)], axis=-1)
            seq_with_score=seq_with_score.astype("float32")

            position_mod = np.where(seq_with_score == 1)[0]

            for jj in range (len(position_mod)):

                seq_with_score[position_mod[jj]] = prediction[kk,position_mod[jj],1]

            Final_seq_binary = np.concatenate((Final_seq_binary,real_part), axis=0)
            Final_seq_score = np.concatenate((Final_seq_score,seq_with_score), axis=0)

        full_position = np.sum(prediction[N_segments], axis = 1)
        full_position = np.where(full_position> 0.5)[0]

        real_part = np.argmax(prediction[N_segments,:len(full_position)], axis=-1)
        seq_with_score = np.argmax(prediction[N_segments,:len(full_position)], axis=-1)
        seq_with_score=seq_with_score.astype("float32")

        position_mod = np.where(seq_with_score == 1)[0]

        for jj in range (len(position_mod)):

            seq_with_score[position_mod[jj]] = prediction[N_segments,position_mod[jj],1]

        not_overlaping_part = real_part[len_overlap:]
        not_overlaping_part_score = seq_with_score[len_overlap:]

        Final_seq_binary = np.concatenate((Final_seq_binary,not_overlaping_part), axis=0)
        Final_seq_score = np.concatenate((Final_seq_score,not_overlaping_part_score), axis=0)


        if (len(Final_seq_binary) - len(sequence_tombo)) != 0:

            N_miss += 1


        else:

            where_mod = np.where(Final_seq_binary >= 1)[0]
            modific_detec = np.zeros(len(where_mod))

            for j in range(len(where_mod)):

                modific_detec[j] = Final_seq_binary[where_mod[j]]
        
            if len(modific_detec) > 1:

                for n in range(len(modific_detec)):

                    mod_probe_position = where_mod[n]
                    mod_probe_predicted = modific_detec[n]

                    reference_track_mod[int(mod_probe_position) + int(position_adjusting), int(mod_probe_predicted -1)] += 1

            else:

                mod_probe_position = where_mod[0]
                mod_probe_predicted = modific_detec[0]

                reference_track_mod[int(mod_probe_position) + int(position_adjusting), int(mod_probe_predicted -1)] += 1   

    except:

        None

print("not analyzed reads:", N_miss)

file_name = "Final_analysis.npz".format(i) # train_data_Gmd

np.savez_compressed(os.path.join(save_path,file_name),
                    reference_frequency = reference_track_mod)

print("data saved in the currenct analysis folder")

