from cmath import log
import csv
import os
from tkinter import END
import pandas as pd

from config import nvprof_paths, metrics
from config import C as C
from config import HW as HW
from config import RS as RS
from config import K as K

if __name__ == '__main__':
    for j in range(0, len(nvprof_paths)):
        output_file = 'csv/'+nvprof_paths[j] + '_sum.csv'
        with open(output_file, 'r') as out:
            reader = csv.reader(out)
            headers = next(reader)

        for i in range(0, len(C)):
            log_file = nvprof_paths[j] + '/nvprof_comp_' + \
                str(C[i]) + '_' + str(HW[i]) + \
                '_' + str(K[i]) + '_' + str(RS[i]) + '.csv'
            myDict = {}
            myDict['Configuration'] = str(C[i]) + '_' + str(HW[i]) + '_' + str(K[i]) + '_' + str(RS[i])
            with open(log_file, 'r') as log:
                    while True:
                        pos = log.tell()
                        line = log.readline()
                        if not line:  # End of file
                            break
                        if "Kernel Name" in line:
                            log.seek(pos)  # rewind to beginning of the header line
                            break

                    reader = csv.DictReader(log)
                    for row in reader:
                        kernel_name = row.get("Kernel Name", "").strip()
                        invocation = row.get("Invocations", "").strip()
                        avg = row.get("Average", "").strip()
                        
                        invocation = invocation.replace(',', '')
                        avg = avg.replace(',', '')
                        time = float(avg) * int(invocation)
                        
                        unit = row.get("Metric Unit", "").strip()
                        if unit == "ns":
                            time = time / 1000000
                        else:
                            raise NotImplementedError("Unsupported unit: " + unit)
                        
                        myDict[kernel_name] = time
                    
                    with open(output_file, 'a') as out:
                        extra_keys = set(myDict.keys()) - set(headers)
                        if extra_keys:
                            print(f"Warning: These kernel names are not in header and will be ignored: {extra_keys}")
                        writer = csv.DictWriter(out, fieldnames=headers)
                        writer.writerow(myDict)
        print(nvprof_paths[j] + " parsing finished")
