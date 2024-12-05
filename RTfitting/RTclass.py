import numpy as np
import latex
import pandas as pd
import matplotlib.pyplot as plt
from fit_models import Quadra_model, PLL_model, Linear_model
import warnings
import inspect

# def has_return_value(func):
#     '''
#     checks if the function 'func' has a returned value
#     '''
#     signature = inspect.signature(func)
#     return_annotation = signature.return_annotation
#     return return_annotation is not inspect._empty
#
# def check_results(func):
#     '''
#     This wrapper function checks if the self.results list is empty, and sends out a warning if there is no results
#     '''
#     def wrapped_function(self, *args, **kwargs):
#         if not self.results:
#             warnings.warn("Whoops, you don't have any results yet. Try a group fitting?", UserWarning)
#         elif has_return_value(func):
#             returned_value = func(self, *args, **kwargs)
#             return returned_value
#         else:
#             func(*args, **kwargs)
#     return wrapped_function

class RT:
    def __init__(self, doping, type):
        self.doping = str(doping)  #Doping level
        self.type = type    #One of the three classes of data: "Pristine", "Bi24", "Pb"
        self.results = []   #A list to hold the group-fitting results
        self.temperatures = [] #The temperature list generated during the group-fitting
        self.model = None #A record of the fit_model used
        self.notes = ''  #String to hold the notes about the group-fitting applied

    @staticmethod
    def nearest_index(data, value):
        '''
        returns the integer index of where 'data' has the closest value to 'value'
        '''
        abs_diff = np.abs(data - value)
        index = np.argmin(abs_diff)
        return index

    def check_results(self):
        '''
        sends a warning message if the object does not have any fitting results stored
        '''
        if not self.results:
            warnings.warn("Whoops, you don't have any results yet. Try a group fitting?", UserWarning)
            print(f"This is for p={self.doping} of {self.type}")
    @staticmethod
    def sort_RT(Twave, Rwave):
        '''
        this function sorts the Rwave and Twave in ascending order, while keeping the original (R,T) pairs
        '''
        sort_indices = np.argsort(Twave)
        return Twave[sort_indices], Rwave[sort_indices]
    @staticmethod
    def igor_sort(Twave, Rwave):
        '''
        the sort method used in Igor, only for cross-checking
        '''
        if Twave[0]>Twave[10]:
            sorted_Twave = np.flip(Twave)
            sorted_Rwave = np.flip(Rwave)
            return sorted_Twave, sorted_Rwave
        else:
            return Twave, Rwave

    def plot(self):
        '''
        plots the R(T) data
        '''
        Twave = self.Tdata
        Rwave = self.Rdata
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(Twave, Rwave, 'o--', color='r', label="Resistivity")
        ax.set_xlabel("Temperature(K)", fontsize=14)
        ax.set_ylabel(r'$\rho_{ab}(T)$', fontsize=14)
        ax.legend(fontsize=13)
        plt.show()

    def plot_fit(self, temp):
        '''
        This function plots the data and residuals of the fitting for the "temp" slice, i.e., for fit from [temp, T_end]
        '''
        T_index = RT.nearest_index(self.temperatures, temp)
        result = self.results[T_index]
        fig,_ = result.plot()  #Plot the fit results and residuals with the ModelResult.plot() function of lmfit
        return fig

    # 1. Loading and cropping the RT data
    def load_RT(self, file):
        raw_data = np.loadtxt(file, skiprows=1)
        T_raw = raw_data[:, 0]
        R_raw = raw_data[:, 1]
        self.Tdata, self.Rdata = RT.sort_RT(T_raw, R_raw)
        print(f"The shape of T_data and R_data is: {self.Tdata.shape}, {self.Rdata.shape}")
        print(f"Data of {self.type} p={self.doping} loaded successfully!")

    def load_RT_Bi2201(self, file):
        raw_data = np.loadtxt(file, skiprows=1, delimiter=",")
        T_raw = raw_data[:, 0]
        R_raw = raw_data[:, 1]
        self.Tdata, self.Rdata = RT.sort_RT(T_raw, R_raw)
        print(f"The shape of T_data and R_data is: {self.Tdata.shape}, {self.Rdata.shape}")
        print(f"Data of {self.type} p={self.doping} loaded successfully!")

    def load_RT_LSCO(self, file):
        raw_data = np.loadtxt(file, skiprows=0, delimiter=",")
        T_raw = raw_data[:, 0]
        R_raw = raw_data[:, 1]
        self.Tdata, self.Rdata = RT.sort_RT(T_raw, R_raw)
        print(f"The shape of T_data and R_data is: {self.Tdata.shape}, {self.Rdata.shape}")
        print(f"Data of {self.type} p={self.doping} loaded successfully!")

    def igor_load(self, Tpath, Rpath):
        T_raw = np.loadtxt(Tpath, skiprows=1)
        R_raw = np.loadtxt(Rpath, skiprows=1)
        self.Tdata, self.Rdata = RT.igor_sort(T_raw, R_raw)
        print(f"The shape of T_data and R_data is: {self.Tdata.shape}, {self.Rdata.shape}")
        print(f"Data of {self.type} p={self.doping} loaded successfully!")

    def crop_RT(self, T_start):
        '''
        This function crops both the Twave and the Rwave to the corresponding cutoff-temperature(T_start), and returns the cropped data (Tdata, Rdata)
        '''
        T_end = self.Tdata.max()
        absolute_diff = np.abs(self.Tdata - T_start)
        n_start = np.argmin(absolute_diff)
        return self.Tdata[n_start:], self.Rdata[n_start:]

    # 2. Single-fit and group-fit functionalities
    def single_fit(self, fit_model, ydata, params, xdata, method='least_squares'):
        '''
        :return: The ModelResult object of the fit
        '''
        result = fit_model.fit(ydata, params, x=xdata, method=method)
        return result

    def prep_temps(self, T_i, T_f):
        '''
        returns an ndarray of temperatures
        T_i is the starting-point of the cutoff temperature, i.e., cropping the RT data to [T_i, T_end]
        T_f is the final-point of the cutoff temperature, i.e., cropping RT to [T_f, T_end]
        Custom: T_i > T_f
        '''
        i_index = RT.nearest_index(self.Tdata, T_i)
        f_index = RT.nearest_index(self.Tdata, T_f)
        temp_arr = self.Tdata[f_index:(i_index + 1)]
        temp_arr = np.flip(temp_arr)
        return temp_arr

    def group_fit(self, fit_model, params, T_i, T_f, method='least_squares'):
        '''
        this function fits a group of y and x data, and returns the results of each fit as a lis of the ModelResult objects
        :param fit_model: the model to be used for the fit
        :param params: parameters for the first slice of fitting
        :param T_i: initial cutoff temperature, i.e., fitting the data of R[T_i,T_end] vs T[T_i, T_end]
        :param T_f: the last cutoff temperature, i.e., fitting the data of R[T_f, T_end] vs T[T_f, T_end]
        :results: the generated temperatures are stored in "self.temperatures", fitting results stored in self.results
        '''
        temperatures = self.prep_temps(T_i, T_f)
        results = []
        for temp in temperatures:
            print(temp, end='\r')
            start_index = RT.nearest_index(self.Tdata, temp)
            ydata = self.Rdata[start_index:]
            xdata = self.Tdata[start_index:]
            result = fit_model.fit(ydata, params, x=xdata, method=method)
            params = result.params
            results.append(result)
        self.model = fit_model
        self.temperatures = temperatures
        self.results = results
        self.notes = f"The group fit was applied for {T_i} to {T_f}K, with the fitting model: {self.model}"
        print(self.notes)

    # 3. Tools to assist on analyzing the fit results
    def df_fitreport(self, attr_list):
        '''
        :param 'attr_list' contains any wanted attributes of the ModelResult object
        This function generates a DataFrame report of the fitting attributes listed in 'attr_list'
        and indexes each row with the corresponding cutoff temperature
        '''
        self.check_results()
        results = self.results
        temperatures = self.temperatures
        dict = {}
        for attr_name in attr_list:
            attr_T = []  # the list to hold attr_vs_temperature values
            for i in range(len(temperatures)):
                attr = getattr(results[i], attr_name)
                attr_T.append(attr)
            dict[attr_name] = attr_T
        df = pd.DataFrame(dict, index=temperatures)
        return df

    def df_params(self):
        '''
        This function generates a DataFrame collection of the best_values for each fitted parameter
        '''
        self.check_results()
        temperatures = self.temperatures
        results = self.results
        dict = {}
        param_names = self.model.param_names
        for param in param_names:
            param_T = []  # the list to hold attr_vs_temperature values
            for i in range(len(temperatures)):
                best_value = getattr(results[i], 'best_values').get(param)
                param_T.append(best_value)
            dict[param] = param_T
        df = pd.DataFrame(dict, index=temperatures)
        return df

    def get_params(self, T):
        '''
        This functions returns the best_values of the fitting parameters for the fit at T
        :param T: Target temperature of the fit
        :return: the fitted parameters
        '''
        self.check_results()
        temperatures = self.temperatures
        results = self.results
        T_index = RT.nearest_index(temperatures, T)
        return getattr(results[T_index],'best_values')

    def get_params_init(self, T):
        '''
        This functions returns the init_values of the fitting parameters for the fit at T
        :param T: Target temperature of the fit
        :return: the fitted parameters
        '''
        self.check_results()
        temperatures = self.temperatures
        results = self.results
        T_index = RT.nearest_index(temperatures, T)
        return getattr(results[T_index], 'init_values')

    def get_result(self, T):
        '''
        returns the ModelResult object for fit at T
        '''
        temperatures = self.temperatures
        results = self.results
        T_index = RT.nearest_index(temperatures, T)
        return results[T_index]

    def get_y(self, T):
        '''
        returns the ModelResult object for fit at T
        '''
        Rdata = self.Rdata
        temperatures = self.temperatures
        T_index = RT.nearest_index(temperatures, T)
        return Rdata[T_index]

    def get_Tdata(self, T):
        '''
        returns the temperature data for fit at T, i.e., the xdata for fit at T
        '''
        index = RT.nearest_index(self.Tdata, T)
        Tdata = self.Tdata[index:]
        return Tdata