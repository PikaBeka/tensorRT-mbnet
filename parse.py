from cmath import log
import csv
import os
from tkinter import END
import pandas as pd

from create_csv import takeName
from config import nvprof_paths, metrics
from config import C as C
from config import HW as HW
from config import RS as RS
from config import K as K

# parsr class
class Parse:
    def __init__(self, log_file, C, HW, K, RS, o_file):
        self.log_file = log_file
        self.C = C
        self.HW = HW
        self.K = K
        self.RS = RS
        self.sum = o_file

    # function to work with time
    def parse_time(self, timestamp):
        unit = timestamp[-1]  # retreive first unit, mostly (s)
        time = timestamp[0:-1]  # remove the unit
        if not time[-1].isdigit():  # if there is still unit like (m, n, u)
            unit = time[-1] + unit  # remove it and add to unit
            time = time[0:-1]
        # transform time to ms
        if unit == "s":
            time = float(time) * 1000
        if unit == "us":
            time = round(float(time) * 0.001, 4)
        if unit == "ns":
            time = float(time) * 1e-6
        return float(time)

    def takeRecord(self, elem, dict, total_time):
        # First need to find value
        # then find a name of the field
        possibleHeader = ''

        for word in reversed(elem):
            if word[0].isdigit():
                break
            possibleHeader = word + " " + possibleHeader

        header = takeName(
            possibleHeader.strip())  # look for kernel name

        if elem[0] == 'GPU':  # this case different since time will be in later columns
            time = self.parse_time(elem[3])
        else:
            time = self.parse_time(elem[1])  # obtains time

        if header in dict:
            dict[header] = dict[header] + time
        else:
            dict[header] = time  # add it to header
        total_time = total_time + time  # sum all times
        return total_time

    def parse_file(self):
        if not os.path.exists(self.log_file):
            print('Path to log file is invalid\n')
            return

        df = pd.read_csv(self.sum)

        headers = df.columns

        # with open(self.sum, 'r') as csv_file:
        #     headers = csv_file.readlines()[0].split(',')  # get all headers

        with open(self.sum, 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
            record = {
                'Configuration': '(' + str(self.C) + '_' + str(self.HW) + '_' + str(self.K) + '_' + str(self.RS) + ')'}  # add first configuration value
            with open(self.log_file, 'r') as log:
                data = log.readlines()
                toStart = 0
                total_time = 0.0
                for elem in data:
                    elem = elem.split()
                    if elem[0] == 'API':
                        break

                    if elem[0] == 'GPU':
                        toStart = 1

                    if toStart == 1:
                        # takes a record of the kernel time
                        total_time = self.takeRecord(elem, record, total_time)

            record['Total_time'] = "{:.3f}".format(total_time)
            record['Kernel_time'] = "{:.3f}".format(total_time)

            if "HtoD" in record:
                record['Kernel_time'] = "{:.3f}".format(
                    float(record['Kernel_time']) - float(record['HtoD']))

            if "DtoH" in record:
                record['Kernel_time'] = "{:.3f}".format(
                    float(record['Kernel_time']) - float(record['DtoH']))

            if "memset" in record:
                record['Kernel_time'] = "{:.3f}".format(
                    float(record['Kernel_time']) - float(record['memset']))

            if "DtoD]" in record:
                record['Kernel_time'] = "{:.3f}".format(
                    float(record['Kernel_time']) - float(record['DtoD]']))

            csv_writer.writerow(record)
            csv_file.close()


class metricsParse(Parse):
    def __init__(self, log_file, C, HW, K):
        super(metricsParse, self).__init__(log_file, C, HW, K, 'simple')

    def parse_file(self, dict, metric):
        if not os.path.exists(self.log_file):
            print('Path to log file is invalid\n')
            return

        with open("metrics/" + metric + "_sum.csv", 'a') as csv_file:
            with open(self.log_file, 'r') as log:
                data = log.readlines()

                for elem in data:
                    elem = elem.split()

                    if elem[0] == 'Kernel:':

                        if elem[1] == 'void':
                            header = elem[2]
                        else:
                            header = elem[1]

                        end = 0
                        for ch in header:
                            end += 1
                            if ch == '(' or ch == '<':
                                break
                        header = header[0:end]

                        if header[-1] == '(' or header[-1] == '<':
                            header = header[0:-1]
                        if header == 'kernel_conv_filter':
                            header += '\n'

                    if elem[1] == metric:
                        val = elem[-1]
                        if not val[-1].isdigit():
                            val = val[0:-1]
                        if not val[0].isdigit():
                            val = val[1:]
                        dict[header] = val
        return dict


if __name__ == '__main__':
    for j in range(0, len(nvprof_paths)):
        for i in range(0, len(C)):
            log_file = nvprof_paths[j] + '/nvprof_comp_' + \
                str(C[i]) + '_' + str(HW[i]) + \
                '_' + str(K[i]) + '_' + str(RS[i]) + '.txt'
            parser = Parse(log_file, int(C[i]), int(
                HW[i]), int(K[i]), int(RS[i]), 'csv/'+nvprof_paths[j]+'_sum.csv')
            parser.parse_file()
        print(nvprof_paths[j] + " parsing finished")

    # for metric in metrics:
    #     for i in range(0, len(C)):

    #         with open('metrics/'+metric+'_sum.csv', 'a') as output:
    #            	with open("metrics/" + metric + "_sum.csv", 'r') as file:
    #                  headers = file.readlines()[0].split(
    #                      ',')  # get all headers
    #             csv_writer = csv.DictWriter(output, fieldnames=headers)
    #             dict = {'Configuration': '(' + str(C[i])+'_'+str(HW[i])+'_'+str(K[i]) + ')'}
                
    #             for path in nvprof_paths:
    #                 if path == 'direct_shared':
    #                     log_file = 'metrics/'+path+'/nvprof_comp_' + \
    #                         str(C[i])+'_'+str(HW[i])+'_'+str(K[i])+'.txt'

    #                     parser = metricsParse(log_file, int(
    #                         C[i]), int(HW[i]), int(K[i]))

    #                     dict = parser.parse_file(dict, metric)

    #             csv_writer.writerow(dict)
    #         output.close()
    #     print(metric + " parsing finished")
