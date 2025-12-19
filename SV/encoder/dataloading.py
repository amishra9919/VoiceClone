# import numpy as np
# #create new numpy files
# a = np.load("D:\\Mini Project\\Real-Time-Voice-Cloning-master\\SV\\encoder\\clean_150\\*.npy")
# print(np.shape(a))    

# # for i in range(2):  #speaker
# #     for j in range(10): #npy
# #         for k in range(640): #datapoint
            
# ground_truth = np.repeat(np.arange(64), 10)
# print(ground_truth)
import os
import numpy as np
for filename in os.listdir("files"):
   with open(os.path.join("files", filename), 'r') as f:
       a = np.load(filename)
       a