# example for reading data from file

import numpy as np                    # get the file reader

# extract data from the output file
data = np.genfromtxt("/home/steven-wallace/Documents/asu/eee-419/notes/M9/m9_4_4_data.csv",comments="$",skip_header=3)
val_type_1 = data["type1"]                 # extract the values
val_type_2 = data["type2"]
val_type_3 = data["type3"]
print(val_type_1,val_type_2,val_type_3)    # print the results
