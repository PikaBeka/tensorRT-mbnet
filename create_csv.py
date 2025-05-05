import csv
import os
from config import nvprof_paths, metrics, C, HW, K, RS

AreMetrics = False

if __name__ == '__main__':

    for i in range(0, len(nvprof_paths)):  # for each kernel method
        method = 'csv/'+nvprof_paths[i] + '_sum.csv'
        headers = []
        kernel_names = set()
        with open(method, 'w', newline='') as fopen:  # create csv file
            files = os.listdir(nvprof_paths[i])
            headers = ['Configuration']  # first column to indicate configuration

            for j in range(0, len(C)):  # for every file in folders
                file = "nvprof_comp_" + str(C[j]) + "_" + str(HW[j]) +"_" + str(K[j]) + "_" + str(RS[j]) + ".csv"
                # print(file)
                if not os.path.exists(nvprof_paths[i] + '/' + file):
                    print("Error: " + file + " not found")
                    continue
                with open(nvprof_paths[i]+'/'+file, 'r') as log:
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
                        if kernel_name:
                            kernel_names.add(kernel_name)

            # print(kernel_names)
            headers += sorted(kernel_names)
            writer = csv.writer(fopen)
            headers.append('Total_time')  # column for total time
            headers.append('Kernel_time')  # column for kernel time (convolution time)
            # print(headers)
            writer.writerow(headers)
        print('Created csv file ' + method)


    # for metric in metrics:  # we create csv file for every metrics

    #     out_file = 'csv/'+ metric +'_sum.csv'  # csv file created

    #     header = ['Configuration']  # column to store conf information

    #     with open(out_file, 'w', newline='') as fopen:  # open csv file to write
    #         for path in nvprof_paths:  # traverse each metrics txt file
    #             files = os.listdir('metrics/'+metric+'/'+path)  # list directories
    #             for file in files:
    #                 with open('metrics/'+path+'/'+file, 'r') as log:  # open each file
    #                     data = log.readlines()  # read their lines
    #                     header = find_headers_metric(data, header, metrics)
    #         writer = csv.writer(fopen)
    #         writer.writerow(header)
    #     print('Created csv file ' + out_file)
