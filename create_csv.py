import csv
import os
from config import nvprof_paths, metrics, C, HW, K, RS

AreMetrics = False

# this function finds a name of the kernel
def takeName(possible_header):
    if possible_header == '[CUDA memcpy DtoH]':
        return 'DtoH'
    if possible_header == '[CUDA memcpy HtoD]':
        return 'HtoD'
    if possible_header == '[CUDA memset]':
        return 'memset'
    
    if possible_header[:5] == "void ":
        possible_header = possible_header[5:]
    
    word = ''
    for ch in possible_header:
        # if ch == ' ':  # only one word required
        #     word = ''
        #     continue
        if ch == '(':  # in case we find parameters we stop
            break
        word += ch
    return word


# function looks for all unique column names
def find_headers_time(data, headers):
    toStart = 0
    for elem in data:
        elem = elem.split()  # split lines to words

        if elem[0] == 'API':  # if we reach the API, we already looked GPU kernels
            break

        if elem[0] == 'GPU':  # after this kernel starts
            toStart = 1

        if toStart == 1:
            possibleHeader = ''
            # go from backward since the kernel name is the last columns
            for word in reversed(elem):
                if word[0].isdigit():
                    break
                possibleHeader = word + " " + possibleHeader
            # function retrieves only name without conf
            header = takeName(possibleHeader.strip())
            if header not in headers:  # add only if unique
                headers.append(header)
    return headers

def find_headers_metric(data, headers):
    toStart = 0
    for elem in data:
        elem = elem.split()

        if elem[0] == 'Kernel:':
            possibleHeader = ''
            # go from backward since the kernel name is the last columns
            for word in reversed(elem):
                if word[0].isdigit():
                    break
                possibleHeader = word + " " + possibleHeader
            # function retrieves only name without conf
            header = takeName(possibleHeader.strip())
            if header not in headers:  # add only if unique
                headers.append(header)
    return headers

if __name__ == '__main__':

    for i in range(0, len(nvprof_paths)):  # for each kernel method
        method = 'csv/'+nvprof_paths[i] + '_sum.csv'
        headers = []
        with open(method, 'w', newline='') as fopen:  # create csv file
            files = os.listdir(nvprof_paths[i])
            headers = ['Configuration']  # first column to indicate configuration

            for j in range(0, len(C)):  # for every file in folders
                file = "nvprof_comp_" + str(C[j]) + "_" + str(HW[j]) +"_" + str(K[j]) + "_" + str(RS[j]) + ".txt"
                # print(file)
                if not os.path.exists(nvprof_paths[i] + '/' + file):
                    print("Error: " + file + " not found")
                    continue
                with open(nvprof_paths[i]+'/'+file, 'r') as log:
                    data = log.readlines()
                    # function returns all found headers
                    headers = find_headers_time(data, headers)
                    # break

            writer = csv.writer(fopen)
            headers.append('Total_time')  # column for total time
            headers.append('Kernel_time')  # column for kernel time (convolution time)
            print(headers)
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
