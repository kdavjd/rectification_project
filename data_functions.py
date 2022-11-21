import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class DataFunctions():
    def delete_hyphens(frame, exclude_list) -> pd.DataFrame:
        """Некоторые таблицы имеют пропуски в виде дефиса вместо значений. 
        Функция меняет все нечисловые значения на NaN. 

        Args:
            frame (pd.DataFrame): Таблица которую необходимо преобразовать
            exclude_list (list): список с именами столбцов, которые не нужно трогать

        Returns:
            pd.DataFrame: Новая таблица
        """
         

        new_frame = pd.DataFrame()
        
        for column in range(len(frame.columns)):                       
            if frame.columns[column] in exclude_list:
                new_frame[frame.columns[column]] = frame[frame.columns[column]]
            else:
                new_frame[frame.columns[column]] = frame[frame.columns[column]].apply(lambda x: np.nan if type(x) == str else x)
        return new_frame

    def get_coeffs(x, y) -> np.array:
        """Функция возвращает уравнение полинома, по которому можно будет найти значение
        аргумента.

        Args:
            x (pd.Series): Значения аргумента, по которому строится полином.
            y (pd.Series): Значения функции аргумента, по которому строится полином.

        Returns:
            np.array: Коэффициенты уравнения полинома.
        """
        rmse = []
        for i in range(7):
            if i >= len(x):
                break    
            coefficient = np.polyfit(x, y, i)
            y_i = np.polyval(coefficient, x)  
            rmse.append((np.square(y - y_i )).mean())
        n_poly = rmse.index(min(np.array(rmse)))
        y_coeff = np.polyfit(x,y,n_poly)
        
        return y_coeff

    def get_fit(x, y) -> np.array:
        """Функция возвращает аппроксимизированные значение
        кривой полиномом 0-6 порядка

        Args:
            x (pd.Series): Значения аргумента, по которому строится полином
            y (pd.Series): Значения функции аргумента, по которому строится полином

        Returns:
            np.array: Значения полинома функции аргумента кривой
        """
        rmse = []
        for i in range(7):
            if i >= len(x):
                break    
            coefficient = np.polyfit(x, y, i)
            y_i = np.polyval(coefficient, x)  
            rmse.append((np.square(y - y_i )).mean())
        n_poly = rmse.index(min(np.array(rmse)))
        y_coeff = np.polyfit(x,y,n_poly)
        y_n = np.polyval(y_coeff, x)
        return y_n
    