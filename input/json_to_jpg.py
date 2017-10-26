import numpy as np
import pandas as pd
from scipy import misc

import argparse
import os, errno

def json_to_jpg(input_json, output, label_file_name, is_test):
    input_data=pd.read_json(input_json)
    label_file = open(label_file_name, 'w')

    for i in range(input_data.shape[0]):
        if not (i) % 100:
            print("converting " + str(i) + "th imgs") 

        band1 = np.array(input_data.band_1[i]).reshape(75,75)
        band2 = np.array(input_data.band_2[i]).reshape(75,75)
        band3 = band1 * band2 / 100
        img = np.stack((band1, band2, band3), axis = -1)
        misc.imsave(output+ "/" + input_data.id[i]+".jpg", img)
        if is_test:
            label_file.write(str(i) + '\t' + str(0) + '\t' + output + '/' + input_data.id[i]+".jpg\n")
        else:
            label_file.write(str(i) + '\t' + str(input_data.is_iceberg[i]) + '\t' + output + '/' + input_data.id[i]+".jpg\n")

    label_file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert json file to pngs",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json_input', type=str, help='json file')
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--label_file', type=str, help='output label file')
    parser.add_argument('--is_test', type=bool, help='is test')
    parser.set_defaults(is_test=False)

    args = parser.parse_args()

    try:
        os.makedirs(args.output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    is_test =  (args.is_test == 'True')

    json_to_jpg(args.json_input, args.output, args.label_file, is_test)

