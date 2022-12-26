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

    def get_coeffs(x, y, n=7) -> np.array:
        """Функция возвращает уравнение полинома, по которому можно будет найти значение
        аргумента.

        Args:
            x (pd.Series): Значения аргумента, по которому строится полином.
            y (pd.Series): Значения функции аргумента, по которому строится полином.

        Returns:
            np.array: Коэффициенты уравнения полинома.
        """
        rmse = []
        for i in range(n):
            if i >= len(x):
                break    
            coefficient = np.polyfit(x, y, i)
            y_i = np.polyval(coefficient, x)  
            rmse.append((np.square(y - y_i )).mean())
        n_poly = rmse.index(min(np.array(rmse)))
        y_coeff = np.polyfit(x,y,n_poly)
        
        return y_coeff

    def get_fit(x, y, n=7) -> np.array:
        """Функция возвращает аппроксимизированные значение
        кривой полиномом 0-6 порядка

        Args:
            x (pd.Series): Значения аргумента, по которому строится полином
            y (pd.Series): Значения функции аргумента, по которому строится полином

        Returns:
            np.array: Значения полинома функции аргумента кривой
        """
        rmse = []
        for i in range(n):
            if i >= len(x):
                break    
            coefficient = np.polyfit(x, y, i)
            y_i = np.polyval(coefficient, x)  
            rmse.append((np.square(y - y_i )).mean())
        n_poly = rmse.index(min(np.array(rmse)))
        y_coeff = np.polyfit(x,y,n_poly)
        y_n = np.polyval(y_coeff, x)
        return y_n
    
    def delete_experiment_errors(frame, col) -> pd.DataFrame:
        """Функция преобразует серию данных, удаляя те точки, в которых значение аргумента \n
        уменьшается при увеличении индекса. В условиях последовательного эксперимента это ошибка.

        Args:
            frame (pd.DataFrame): Таблица со всеми данными экспериментов бинарной смеси.
            col (_type_): Серия экспериментов, по которой проводится проверка.

        Returns:
            pd.DataFrame: Таблица с данными без ошибок экспериментов.
        """
        x_ = np.arange(len(frame)).astype('float64')
        values = frame[col].values
        for i in range(len(frame)):
            if i < len(frame)-1:
                x_[i] = values[i+1] - values[i]
            else:
                x_[i] = 0
            frame['xi'] = x_
        return frame.drop(index = frame[frame['xi'] < 0].index)
    
    def prepare_data(data: pd.DataFrame, filter_by: str, function_of: str, var_list: list, x_min = 0, x_max = 100, x_range = 100) -> pd.DataFrame:
        """Функция возвращает две таблицы. В первой исходные данные преобразуются так что: \n
        кривые экспериментов апроксиммируются и аргумент принимает значения от x_min до x_max, в колличестве x_range\n
        значения функции находятся для всего диапазона значений аргумента в каждой точке var_list  от function_of. 
        Вторая таблица хранит в себе коэффициенты функций кривых.
        

        Args:
            data (pd.DataFrame): Исходная таблица с экспериментами бинарного раствора.
            filter_by (str): 't' либо 'p' - переменная по оси Z.
            function_of (str): 't' либо 'p' - переменная, по оси Y. 
            var_list (list): ОТСОРТИРОВАННЫЙ список значений переменной по оси Z, в которых она определена.
            x_min (int, optional): Минимальный мольный процент интервала интерполяции. Defaults to 0.
            x_max (int, optional): Максимальный мольный процент интервала интерполяции. Defaults to 100.
            x_range (int, optional): Колличество шагов. Defaults to 100.

        Returns:
            pd.DataFrame: таблица с интерполированными данными
            pd.DataFrame: таблица с функциями кривых 
        """

        df = pd.DataFrame()
        f_= pd.DataFrame()
        for t in var_list:
            x = np.linspace(x_min,x_max,x_range) 
            t_ = data[data[filter_by] == t]
            t_ = t_.sort_values('x')
            t_ = DataFunctions.delete_experiment_errors(t_, 'x')
            f_x = DataFunctions.get_coeffs(t_['x'],t_[function_of])
            t_ = t_.sort_values('y')
            t_ = DataFunctions.delete_experiment_errors(t_, 'y')
            f_y = DataFunctions.get_coeffs(t_['y'],t_[function_of])
            df['x',t] = np.poly1d(f_x)(x)
            df['y',t] = np.poly1d(f_y)(x)
            f_['x',t] = f_x
            f_['y',t] = f_y
        return df, f_
    
    def get_surface_data(data, low, high, f, dx, x_min = 0, x_max= 100, x_range = 100) -> pd.DataFrame:
        """Функция соединяет прямыми точки с одинаковым индексом от 'low' к 'high. В каждой точке 'x_range' находится производная; \n
        в случае линейной функции- тангенс 'tg' угла наклона прямой. Функция 'fk' показывает как меняется \n
        коэфициент k в уравнении прямой (y = kx+b) от 'х'. Занчение координаты 'z' в точке 'y' находится \n
        из уравнения прямой где k= fk(x), x = (y - y0), b= f(x)

        Args:
            data (_type_): Таблица с подготовленными данными после prepare_data
            low (_type_): Нижнее значение интерполируемой области
            high (_type_): Верхнее значение интерполируемой области
            f (_type_): Коэфициенты полинома функции кривой нижнего значения области. Берется из prepare_data
            dx (_type_): Длина прилежащего катета. 
            x_min (int, optional): Минимальный мольный процент интервала интерполяции. Defaults to 0.
            x_max (int, optional): Максимальный мольный процент интервала интерполяции. Defaults to 100.
            x_range (int, optional): Колличество шагов. Defaults to 100.

        Returns:
            pd.DataFrame: _description_
        """
        x = np.linspace(x_min,x_max,x_range) 
        tg = (data[high] - data[low])/(dx)
        y = np.linspace(list(low)[1],list(high)[1],100)
        fk = DataFunctions.get_coeffs(x,tg)
        x_, y_ = np.meshgrid(x,y)
        z = np.poly1d(fk)(x_)*(y_-y_[0])+np.poly1d(f)(x)
        
        return x_,y_,z
    
    def get_surface(prepared_data: pd.DataFrame, data_coefs: pd.DataFrame) -> pd.DataFrame:
        """Функция возвращает коодинаты двух поверхностей- значения жидкой и паровой фазы из \n 
        преобразоанной таблицы с исходными экспериментами функцией prepare_data.

        Args:
            prepared_data (pd.DataFrame): Таблица с преобразованными данными функцией prepare_data
            data_coefs (pd.DataFrame): коэфициенты функций кривых, полученных функцией prepare_data

        Returns:
            pd.DataFrame: четыре таблицы с координатами поверхностей
        """
        surface_x = pd.DataFrame()
        surface_y = pd.DataFrame()
        surface_zx = pd.DataFrame()
        surface_zy = pd.DataFrame()
        liquid_list = list(prepared_data.columns)[::2]
        vapor_list = list(prepared_data.columns)[1::2]
        for var in liquid_list:
            if len(liquid_list) == 1:
                break
            else:
                low_liquid = liquid_list[0]
                low_vapor = vapor_list[0]
                high_liquid = liquid_list[1]
                high_vapor = vapor_list[1]
                
                _x, _y, _zx = DataFunctions.get_surface_data(
                    data=prepared_data, low=low_liquid, high= high_liquid, 
                    dx= (liquid_list[1][1] - liquid_list[0][1]), 
                    f = data_coefs[low_liquid])
                
                _x, _y, _zy = DataFunctions.get_surface_data(
                    data=prepared_data, low=low_vapor, high= high_vapor,
                    dx= (vapor_list[1][1] - vapor_list[0][1]),
                    f = data_coefs[low_vapor])
                
                liquid_list = liquid_list[1::]
                vapor_list = vapor_list[1::]
                surface_x = pd.concat([surface_x, pd.DataFrame(_x)])
                surface_y = pd.concat([surface_y, pd.DataFrame(_y)])
                surface_zx = pd.concat([surface_zx, pd.DataFrame(_zx)])
                surface_zy = pd.concat([surface_zy, pd.DataFrame(_zy)])
                
        return surface_x, surface_y, surface_zx, surface_zy