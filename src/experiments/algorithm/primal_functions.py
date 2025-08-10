import numpy as np

class Primal:
    @staticmethod
    def add_column(data, times):
        for i in range(1, times + 1):
            new = np.zeros((len(data), 1), dtype=float)
            data = np.append(data, new, axis=1)
        return data

    @staticmethod
    def delete_column(data,index,times):
        for i in range(1,times+1):
            data = np.delete(data,index, axis=1)
        return data

    @staticmethod
    def add_row(data,times):
        for i in range(1,times+1):
            columns = np.shape(data)[1]
            new = np.zeros((1,columns),dtype = float)
            data = np.append(data,new,axis = 0)
        return data

    @staticmethod
    def delete_row(data,number):
        data = data[number:,]
        return data

    @staticmethod
    def rounding(data, how_far):
        data = data.round(decimals=how_far)
        return data
