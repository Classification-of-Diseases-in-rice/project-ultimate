"""
This for splitting our Dataset into training & testing & validation

The dataset must be in same directory of this script
"""

import splitfolders
input_folder = "E:\PROJECT\Scripts\Data_Set\Input_Data" #Path of input dataset
output = "E:\PROJECT\Scripts\Data_Set/Processed_Data" #Path for output dataset
splitfolders.ratio(input_folder, output ,seed=1337 ,ratio=(.7,.15,.15))

"""
Ratio takes four inputs inputfolder,outputfolder,seed,ratio(training%,validation%,testing%)
Seed is random sampling of images
"""
