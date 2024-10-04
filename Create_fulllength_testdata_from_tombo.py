#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import pysam

"""
////////////////////////////////
///////  this program riassign the raw data to tombo analysisi table. /////////
////////////////////////////////
"""

main_folder = "/main_folder" # folder where the single_fast5 is saved
folders_list = os.listdir(main_folder)

path = os.getcwd()
save_path = path + "/saving" #saving path

modified_data = True

# ///////// read multi fasta file after the Tombo analysis /////////////

Multi_fast5_file_path = main_folder + "/single_fast5"
multi_fast5_files = os.listdir(main_folder + "/single_fast5" ) # + folders_list[l]

# //// load the .bam file after basecalling //////////

bamfile_name = "path/your_file.bam"
samfile = pysam.AlignmentFile(bamfile_name, "rb")


for i in range (0,len(multi_fast5_files)): #   len(multi_fast5_files) # i created up to 25 for modified data

    print(i)

    try:

        fast5_files = os.listdir(Multi_fast5_file_path + "/" + multi_fast5_files[i])

        for j in range(len(fast5_files)): # len(fast5_files)

            files = h5.File(Multi_fast5_file_path + "/" + multi_fast5_files[i]  + "/" + fast5_files[j], "r")

            try:

                probe = files["Analyses/RawGenomeCorrected_000/BaseCalled_template/Events"]

                raw_signal = files["Raw/Reads"]
                name = np.array(raw_signal)
                raw_signal = files["Raw/Reads" + "/" + name[0] + "/Signal"]
                template = files["Raw/Reads" + "/" + name[0]]

                raw_signal = np.array(raw_signal)
                raw_signal = np.flip(raw_signal)

                # this allows to obtain the starting of the tombo analysis
                attributes = probe.attrs
                name_start = "read_start_rel_to_raw"
                start_tombo = int(attributes[name_start])

                # take the id name of a read
                attributes = template.attrs
                name_start = "read_id"
                name_id = attributes[name_start]

                for read in samfile.fetch():
                    if read.query_name == name_id:
                        reference_pos = read.reference_start
                        break

                tombo_signal = [];
                probe = np.array(probe)
                sequence = np.empty(len(probe), dtype= str)
                collect_legnth_event = np.zeros(len(probe))

                for k in range(len(probe)):

                    length = probe[k][3]

                    event = np.ones(length)*probe[k][0]
                    tombo_signal = np.concatenate((tombo_signal,event), axis=0)
                    sequence[k] = probe[k][-1]
                    collect_legnth_event[k] = length
                
                labels = 3

                seq_one_hot = np.zeros([len(tombo_signal),3])
                seq_one_hot2 = np.zeros([len(tombo_signal),5])

                if modified_data == True:

                    base_dict_1 = { "A":1, "C":1, "G":1, "T":2}

                else:
                    base_dict_1 = { "A":1, "C":1, "G":1, "T":1}

                base_dict_2 = { "A":1,"C":2, "G":3, "T":4}

                spacer = 1

                for k in range(len(probe)):
                            
                    start = probe[k][2] 
       
                    seq_one_hot2[start,base_dict_2[sequence[k]]] = 1
                    seq_one_hot[start,base_dict_1[sequence[k]]] = 1


                # save the full length datasets

                file_name = "train_data_{}_{}.npz".format(i,j) # train_data_Gmd

                np.savez_compressed(os.path.join(save_path,file_name),
                                    raw_signal_tombo = raw_signal[start_tombo: start_tombo + len(tombo_signal)],
                                    One_hot_tombo = seq_one_hot2, 
                                    seq_tombo = sequence,
                                    id_name = name_id,
                                    reference_start = reference_pos)        

            except:

                print("empty: index ",j)

    except:

        print("can't open file")
