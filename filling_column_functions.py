import pandas as pd

#from engineering_projects.data_functions import DataFunctions
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from data_functions import DataFunctions as dfc


class Calculations():
    
    def material_balance(F, xf, xp, xw, diagram, Substance) -> pd.DataFrame:
        """Получает основные переменные материального баланса для дальнейших расчетов

        Args:
            F (np.double): Производительность по исходной смеси кг/с
            xf (np.double): Содержние в исходной смеси %масс Ллт 
            xp (np.double): Содержние в дистилляте(ректификате) %масс
            xw (np.double): Содержние в кубовом остатке %масс ллт
            diagram (numpy.ndarray): Уравнение полинома диаграмы равновесия по абсциссе жидкость по ординате пар
            Substance (Component): Список из двух объектов класса Компонент в которых хранятся свойства разделяемых веществ

        Returns:
            pd.DataFrame: основные переменные материального баланса для дальнейших расчетов
        """
        
        balance = pd.Series(dtype=float)
        balance['Ma'] = Substance['A'].ph_organic['molar_mass'].values
        balance['Mb'] = Substance['B'].ph_organic['molar_mass'].values
        balance['массовый расход в питателе'] = F
        balance['массовый расход в кубовом остатке'] = F*(xp - xf)/(xp - xw)
        balance['массовый расход в дефлегматоре'] = F - balance['массовый расход в кубовом остатке']    
        balance['xf'] =(xf/balance['Ma'])/((xf/balance['Ma'])+((1-xf)/balance['Mb']))#Мольная доля легколетучего компонента в исходной смеси
        balance['xp'] =(xp/balance['Ma'])/((xp/balance['Ma'])+((1-xp)/balance['Mb']))#Мольная доля легколетучего компонента в диистилляте
        balance['xw'] =(xw/balance['Ma'])/((xw/balance['Ma'])+((1-xw)/balance['Mb']))#Мольная доля легколетучего компонента в кубовом остатке
        balance['yf'] = np.poly1d(diagram)(balance['xf'])
        balance['Rmin'] = (balance['xp'] - balance['yf'])/(balance['yf'] - balance['xf'])
           
                
        
        return balance

    def get_value(component,attribute,temperature):
        """Возвращшает значение искомого параметра

        Args:
            component (): вщество класса компонент
            attribute (): искомый параметр, например вязкозть паров
            temperature (): температура при которой нужно найти параметр

        Returns:
            : значение параметра
        """
        
        attr = getattr(component, attribute)
        coeff = dfc.get_coeffs(list(attr.columns),list(*attr.values))
        value = np.poly1d(coeff)(temperature)
        
        return value
    
    def get_range_phlegm_number(yf, xw, xf, xp, Rmin, xy_diagram, diagram, Bt_range: int, plot_lines = 'True'):
        """Для поиска оптимального флегмового числа "Ropt" необходимо задаться "Bt_range", умножив "Rmin"
        на который, получают точку рабочего флегмового числа -"R". Для всех "R" находится новая "yf" и строится 
        линия рабочего флегмового числа по коодинатам (xw,yw)-(xf,yf) и (xf,yf)-(xp,yp).
        Разделение веществ считают ступенчатым. Ступени разделения имеют ширину равную разнице значений xy_diagram и линии "R". 

        Args:
            yf: Значение содержания компонента ЛЛТ в паровой фазе при Rmin в точке xf
            xw: Доля легколетучего компонента в кубовом остатке
            xf: Доля легколетучего компонента в исходной смеси
            xp: Доля легколетучего компонента в диистилляте
            Rmin: Минимальное флегмовое число
            xy_diagram: Уравнение полинома диаграмы равновесия по абсциссе жидкость по ординате пар
            diagram: Таблица с данными жидкость/пар/температура для данной смеси
            Bt_range (int): Количество точек коэфициента избытка флегмы, которым задается функция при поиске Ropt
            plot_lines (str, optional): Нужно ли строить графики или просто делать рсчет. Defaults to 'True'.

        Returns:
            R: Массив с набором рабочих флегмовых чисел
            Ngraf: Массив с набором чисел ступеней разделения
        """
        
        _ = [0, 1]
        _x = float(xw), float(xf)
        x_ = float(xf), float(xp)
        x_y = dfc.get_fit(diagram['x'], diagram['y'])
        Bt = np.linspace(1.05,3,Bt_range)
        R = Bt*Rmin
        yf_ = R/(R+1)*xf+xp/(R+1)
        step = []
        platform = []
        N = pd.Series(dtype=float)
        if plot_lines == 'True':
            fig = plt.figure(figsize=(15,45))
        i = 1
        
        def biuld_phlegm_lines(yf):
            _y = float(xw), float(yf)
            RW_function = np.polyfit(_x,_y, 1)
            y_ = float(yf), float(xp)
            RP_function = np.polyfit(x_,y_, 1)
            return RW_function, RP_function

        
        for yf in yf_:
            if len(step) == 0:
                step.append(np.poly1d(xy_diagram)(xw))
                
            while step[-1] <= yf:
                RW, RP = biuld_phlegm_lines(yf)
                RW[1] = RW[1] - step[-1]
                platform.append(np.roots(RW))
                step.append(np.poly1d(xy_diagram)(platform[-1]))
                
            while step[-1] <= xp:
                RW, RP = biuld_phlegm_lines(yf)
                RP[1] = RP[1] - step[-1]
                platform.append(np.roots(RP))
                step.append(np.poly1d(xy_diagram)(platform[-1]))
                
            if step[-1] > xp:            
                N[yf] = len(step) #Считаем кол-во ступеней            
                platform = [xw] + platform 
                outside_corners = list(zip(platform,step)) #Получаем координаты вершин внешних углов лестницы
                platform = platform[1::]
                inside_corners = list(zip(platform,step)) #Получаем координаты вершин внутренних углов лестницы
                stair = list(zip(outside_corners,inside_corners)) 
                sort_corners = []
                for x,y in stair:
                    sort_corners.append(x)
                    sort_corners.append(y)
                stair_line_x,stair_line_y = zip(*sort_corners) #Получаем значения ломаной линии по абсцисе и ординате
                W_line = float(xw), float(yf)
                P_line = float(yf), float(xp)
                
                if plot_lines == 'True':                    
                    axes = fig.add_subplot(10,5,i)
                    axes.plot(stair_line_x,stair_line_y, 'o--', lw=1, ms=1)
                    axes.plot(_, _, color='black', lw=0.5)
                    axes.plot(diagram['x'], x_y, color='red', lw=1, ms=2)
                    axes.plot(_x, W_line, color='green', lw=1, ms=2)
                    axes.plot(x_, P_line, color='green', lw=1, ms=2)
                    axes.set_title(f"N = {N[yf]}, R = {round(R[i-1],2)}, Yf = {round(yf,2)}")     
                step = []            
                platform = []
                i += 1
        Ngraf = N.values *(R+1)       
        return R,Ngraf
    
    def get_optimal_phlegm_number(R,Ngraf, plot_lines = "True"):
        """Оптимальное флегмовое число "Ropt" находят по минимуму функции R от N(R+1).
        

        Args:
            R: Массив с набором рабочих флегмовых чисел
            Ngraf: Массив с набором чисел ступеней разделения
            plot_lines (str, optional):нужно ли строить графики или просто проводить расчет. Defaults to "True".

        Returns:
            Ropt: Оптимальное флегмовое число
        """
        Nfit = dfc.get_fit(R,Ngraf)
        optimal = []
        for value in Nfit:
            optimal.append(value == Nfit.min())
        Ropt = np.round(R[optimal],2)
        if plot_lines == 'True':
            fig = plt.figure(figsize=(7,7))
            axes = fig.add_subplot()
            axes.plot(R,Ngraf)
            axes.plot(R,Nfit, '--')
            axes.set_xlabel(r'R')
            axes.set_ylabel(r'N(R+1)')
            axes.set_title(f"Оптимальное флегмовое число = {np.round(R[optimal],2)}") 
            print(f"Оптимальное флегмовое число = {np.round(R[optimal],2)}")
        return Ropt
    
    def calculate_properties_slice(liquid_fraction, vapor_fraction, temperature, Substance, Ma, Mb, slice_type = 'DataFrame'):
        
        thermal_conductivity_a = Calculations.get_value(component= Substance['A'], 
                    attribute='thermal_conductivity_organic_liquid', temperature=temperature)
        #Теплопроводность компонента А [Вт/(м*K)]
        
        thermal_conductivity_b = Calculations.get_value(component= Substance['B'],
                    attribute='thermal_conductivity_organic_liquid', temperature=temperature)
        #Теплопроводность компонента Б  [Вт/(м*K)]
        
        thermal_expansion_a = Calculations.get_value(component= Substance['A'],
                    attribute='thermal_expansion_organic_liquid', temperature=temperature)
        #Коэффициенты объемного теплового расширения компонента А b*10^3, K^-1
        
        thermal_expansion_b = Calculations.get_value(component= Substance['B'],
                    attribute='thermal_expansion_organic_liquid', temperature=temperature)
        #Коэффициенты объемного теплового расширения компонента Б b*10^3, K^-1
        
        vapor_pressure_a = Calculations.get_value(component= Substance['A'],
                    attribute='vapor_pressure_organic_liquid', temperature=temperature)
        #Давление насыщенного пара [мм.рт.ст.] компонента А
        
        vapor_pressure_b = Calculations.get_value(component= Substance['B'],
                    attribute='vapor_pressure_organic_liquid', temperature=temperature)
        #Давление насыщенного пара [мм.рт.ст.] компонента Б
        
        sigma_a = Calculations.get_value(component= Substance['A'],
                    attribute='interfactial_tension_organic_liquid', temperature=temperature)
        #Поверхностное натяжение [мДж/м^2] компонента А
        
        sigma_b = Calculations.get_value(component= Substance['B'],
                    attribute='interfactial_tension_organic_liquid', temperature=temperature)
        #Поверхностное натяжение [мДж/м^2] компонента Б 
        
        Cp_a = Calculations.get_value(component= Substance['A'],
                    attribute='heat_capacity_organic_liquid', temperature=temperature)
        #Удельная теплоемкость [Дж/(кг*K)] компонента А
        
        Cp_b = Calculations.get_value(component= Substance['B'],
                    attribute='heat_capacity_organic_liquid', temperature=temperature)
        #Удельная теплоемкость [Дж/(кг*K)] компонента Б
        
        Qv_a = Calculations.get_value(component= Substance['A'],
                    attribute='heat_vaporization_organic_liquid', temperature=temperature)
        #Теплота парообразования компонента А [кДж/кг]
        
        Qv_b = Calculations.get_value(component= Substance['B'],
                    attribute='heat_vaporization_organic_liquid', temperature=temperature)
        #Теплота парообразования компонента Б [кДж/кг]
        
        p_a = Calculations.get_value(component= Substance['A'],
                    attribute='density_organic_liquid', temperature=temperature)
        #Плотность [кг/м^3] компонента А
        
        p_b = Calculations.get_value(component= Substance['B'],
                    attribute='density_organic_liquid', temperature=temperature)
        #Плотность [кг/м^3] компонента Б 
        
        u_a = Calculations.get_value(component= Substance['A'],
                    attribute='vicosity_organic_liquid', temperature=temperature) 
        #Динамическая вязкость [мПа*с] компонента А 
        
        u_b = Calculations.get_value(component= Substance['B'],
                    attribute='vicosity_organic_liquid', temperature=temperature)
        #Динамическая вязкость [мПа*с] компонента Б 
        
        ug_a = Calculations.get_value(component= Substance['A'],
                    attribute='vicosity_organic_vapor', temperature=temperature)
        #Вязкость паров [мкПа*с] компонента А 
        
        ug_b = Calculations.get_value(component= Substance['B'],
                    attribute='vicosity_organic_vapor', temperature=temperature)
        #Вязкость паров [мкПа*с] компонента Б 


        def calculate_mixture_value(a,b,fraction):
            value = fraction*a + (1-fraction)*b
            return value

        if slice_type == 'DataFrame':
            slice = pd.DataFrame(dtype=np.float64)
        else:
            slice = pd.Series(dtype=np.float64)
        
        slice['температура'] = temperature
        slice['содержание легколетучего в жидкости'] = liquid_fraction
        slice['содержание легколетучего в паре'] = vapor_fraction
        slice['плотность жидкости'] = calculate_mixture_value(p_a,p_b,liquid_fraction)
        slice['теплопроводность жидкости'] = calculate_mixture_value(thermal_conductivity_a,thermal_conductivity_b,liquid_fraction)
        slice['теплота парообразования жидкости'] = calculate_mixture_value(Qv_a,Qv_b,liquid_fraction)
        slice['удельная теплоемкость жидкости'] = calculate_mixture_value(Cp_a,Cp_b,liquid_fraction)
        slice['поверхностное натяжение жидкости'] = calculate_mixture_value(sigma_a,sigma_b,liquid_fraction)
        slice['давление насыщенного пара жидкости'] = calculate_mixture_value(vapor_pressure_a,vapor_pressure_b,liquid_fraction)
        slice['коэффициент объемного расширения жидкости'] = calculate_mixture_value(thermal_expansion_a,thermal_expansion_b,liquid_fraction)
        slice['молярный объем газа'] = 22.4*(273.15+temperature)/273.15#Где 22.4 - молярный объем при н.у, 273.15 температура в Кельвинах
        slice['молярная масса жидкости'] = calculate_mixture_value(Ma,Mb,liquid_fraction)
        slice['молярная масса газа'] = calculate_mixture_value(Ma,Mb,vapor_fraction)
        slice['плотность пара'] = slice['молярная масса газа'] / slice['молярный объем газа']
        slice['вязкость пара']=slice['молярная масса газа']/((vapor_fraction*Ma/(ug_a/1000))+((1-vapor_fraction)*Mb/(ug_b/1000)))
        slice['вязкость жидкости'] = 10.0**(liquid_fraction*np.log10(u_a)+(1-liquid_fraction)*np.log10(u_b))
        slice['молярный объем жидкости'] = slice['молярная масса жидкости']/slice['плотность жидкости']*1000
        return slice

    def calculate_properties(diagram, balance, Substance):
        properties = pd.DataFrame(dtype=float)

        def slice_values(liquid_fraction, balance):
            fx = dfc.get_coeffs(diagram['x'], diagram['t'])
            fy = dfc.get_coeffs(diagram['y'], diagram['t'])
            
            temperature = np.poly1d(fx)(liquid_fraction)
            
            yt=[]
            x = np.linspace(0,1,100)
            yt = np.poly1d(fy)(x)
            
            vapor_fraction = np.array([len(yt[yt>temperature])/100])#Мольная доля ллт в паровой фазе при искомой температуре
            
            return liquid_fraction, vapor_fraction, temperature, balance['Ma'], balance['Mb']

        fraction_list = np.array([balance['xw'], (balance['xw']+balance['xf'])/2, balance['xf'], (balance['xf']+balance['xp'])/2, balance['xp']])

        for fraction in fraction_list:
            liquid_fraction, vapor_fraction, temperature, Ma, Mb = slice_values(fraction, balance)
            slice = Calculations.calculate_properties_slice(liquid_fraction, vapor_fraction, temperature, Substance, Ma, Mb)
            properties = pd.concat([properties, slice])
        
        properties.index = ['куба', 'низа','питания','верха','дистиллята']
        return properties
    
    def get_transfer_numbers(balance, Ropt, xy_diagram, plot_lines = 'True'):
        
        #Готовим значения функций для графиков
        yf = Ropt/(Ropt+1)*balance['xf']+balance['xp']/(Ropt+1)
        _x = float(balance['xw']), float(balance['xf'])
        x_ = float(balance['xf']), float(balance['xp'])
        W_line = float(balance['xw']), float(yf)
        P_line = float(yf), float(balance['xp'])

        def biuld_phlegm_lines(yf):        
                _y = float(balance['xw']), float(yf)
                RW_function = np.polyfit(_x,_y, 1)        
                y_ = float(yf), float(balance['xp'])
                RP_function = np.polyfit(x_,y_, 1)
                return RW_function, RP_function

        #Находим все функции для графиков
        w, p = biuld_phlegm_lines(yf)

        def fw(x): return 1/(np.poly1d(xy_diagram)(x) - np.poly1d(w)(x))
        def fp(x): return 1/(np.poly1d(xy_diagram)(x) - np.poly1d(p)(x))
        def fxy(x): return np.poly1d(xy_diagram)(x)
        def fxyw(x): return np.poly1d(w)(x)
        def fxyp(x): return np.poly1d(p)(x)

        w_x = np.linspace(float(balance['xw']), float(balance['xf']), 100)
        p_x = np.linspace(float(balance['xf']), float(balance['xp']), 100)
        wi_x = np.poly1d(xy_diagram)(np.linspace(float(balance['xw']), float(balance['xf']), 100))
        pi_x = np.poly1d(xy_diagram)(np.linspace(float(balance['xf']), float(balance['xp']), 100))
        xy = np.linspace(0, 1, 100)

        #Считаем интегралы
        bottom_values = np.poly1d(xy_diagram)(np.linspace(float(balance['xw']), float(balance['xf']), 100))
        distillate_values = np.poly1d(xy_diagram)(np.linspace(float(balance['xf']), float(balance['xp']), 100))
        
        bottom_function = dfc.get_coeffs(bottom_values,fw(w_x))
        distillate_function = dfc.get_coeffs(distillate_values,fp(p_x)) 
         
        def fbottom(x): return np.poly1d(bottom_function)(x)
        def fdistillate(x): return np.poly1d(distillate_function)(x)
        
        bottom = round(integrate.quad(fbottom, bottom_values[0],bottom_values[-1])[0],2)
        top = round(integrate.quad(fdistillate, distillate_values[0],distillate_values[-1])[0],2)

        if plot_lines == 'True':
            #Строим графики
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
            _ = [0, 1]
            axes[0].plot(wi_x, fw(w_x), label = (r'$\int {низ}$= ' + f'{bottom}'))
            axes[0].fill_between(wi_x, fw(w_x), where=[(w_x >= float(balance['xw'])) and (w_x <= float(balance['xf'])) for w_x in w_x],
                            color = 'blue', alpha = 0.4)
            axes[0].plot(pi_x, fp(p_x), label = (r'$\int {верх}$= ' + f'{top}'))
            axes[0].fill_between(pi_x, fp(p_x), where=[(p_x >= float(balance['xf'])) and (p_x <= float(balance['xp'])) for p_x in p_x],
                            color = 'green', alpha = 0.4)
            axes[0].set_ylabel(r'$ \frac {1}{y* - y}$',  fontsize=15)
            axes[0].set_xlabel(f'Мольная доля ллт в паре', fontsize=10)
            axes[0].legend(loc='upper center')


            axes[1].plot(xy, fxy(xy), color='black', lw=1, ms=2)
            axes[1].plot(_, _, color='black', lw=0.5)
            axes[1].plot(_x, W_line, color='black', lw=1, ms=2)
            axes[1].plot(x_, P_line, color='black', lw=1, ms=2)
            axes[1].fill_between(w_x, fxy(w_x), fxyw(w_x),  where=[(w_x >= float(balance['xw'])) and (w_x <= float(balance['xf'])) for w_x in w_x],
                            color = 'blue', alpha = 0.4)
            axes[1].fill_between(p_x, fxy(p_x), fxyp(p_x), where=[(p_x >= float(balance['xf'])) and (p_x <= float(balance['xp'])) for p_x in p_x],
                            color = 'green', alpha = 0.4)
            axes[1].set_ylabel(f'Мольная доля ллт в паре', fontsize=10)
            axes[1].set_xlabel(f'Мольная доля ллт в жидкости', fontsize=10)
        return bottom, top
    
    def calculate_diameter(balance, Ropt, properties, filling_name: str):
        diameter = pd.Series(dtype=float)
        standart_list = np.array([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.6, 3.0])
        
        filling = pd.DataFrame(columns = ['удельная поверхность','свободный объем','насыпная плотность'], dtype=float)
        filling.loc['25x25x3'] = [200, 0.74, 530]
        filling.loc['35x35x4'] = [140, 0.78, 530]
        filling.loc['50x50x5'] = [87.5, 0.785, 530]
        filling = filling.loc[filling_name]
        
        diameter['массовая нагрузка жидкости верха'] = (balance['массовый расход в дефлегматоре'] * Ropt * properties['молярная масса жидкости']['верха'] 
                                                        / properties['молярная масса жидкости']['дистиллята'])
        
        diameter['массовая нагрузка жидкости низа'] = ((balance['массовый расход в дефлегматоре'] * Ropt * properties['молярная масса жидкости']['низа'] 
                                                        / properties['молярная масса жидкости']['дистиллята']) 
                + balance['массовый расход в питателе'] * properties['молярная масса жидкости']['низа'] / properties['молярная масса жидкости']['питания'])
        
        diameter['массовый поток пара верха'] = (balance['массовый расход в дефлегматоре'] * (Ropt+1) * properties['молярная масса газа']['верха'] 
                                                / properties['молярная масса газа']['дистиллята'])
        
        diameter['массовый поток пара низа'] = (balance['массовый расход в дефлегматоре'] * (Ropt+1) * properties['молярная масса газа']['низа'] 
                                                / properties['молярная масса газа']['дистиллята'])
        
        diameter['предельная скорость пара верха'] = np.sqrt(
        1.2*np.exp(-4 * (diameter['массовая нагрузка жидкости верха'] / diameter['массовый поток пара верха'])**0.25 
                            *(properties['плотность пара']['верха'] / properties['плотность жидкости']['верха'])**0.125)
                                                    * (9.8 * filling['свободный объем']**3 * properties['плотность жидкости']['верха'])
            /(filling['удельная поверхность'] * properties['плотность пара']['верха'] * properties['вязкость жидкости']['верха']**0.16))
        
        diameter['предельная скорость пара низа'] = np.sqrt(
        1.2*np.exp(-4 * (diameter['массовая нагрузка жидкости низа'] / diameter['массовый поток пара низа'])**0.25
                                *(properties['плотность пара']['низа'] / properties['плотность жидкости']['низа'])**0.125)
                                                    * (9.8 * filling['свободный объем']**3 * properties['плотность жидкости']['низа'])
            /(filling['удельная поверхность'] * properties['плотность пара']['низа'] * properties['вязкость жидкости']['низа']**0.16))
        
        diameter['рабочая скорость пара верха'] = diameter['предельная скорость пара верха'] * 0.7
        
        diameter['рабочая скорость пара низа'] = diameter['предельная скорость пара низа'] * 0.7
        
        diameter['диаметр верха'] = np.sqrt(4*diameter['массовый поток пара верха']
                                    / (np.pi * diameter['рабочая скорость пара верха'] * properties['плотность пара']['верха']))
        
        diameter['диаметр низа'] = np.sqrt(4*diameter['массовый поток пара низа']
                                    / (np.pi * diameter['рабочая скорость пара низа'] * properties['плотность пара']['низа']))
        
        diameter['стандартный размер обечайки'] = np.array([standart_list[standart_list > diameter['диаметр низа']].min(),
                                                            standart_list[standart_list > diameter['диаметр верха']].min()]).max()
        
        diameter['действительная рабочая скорость верха'] = diameter['рабочая скорость пара верха'] *(diameter['диаметр верха'] 
                                                                                                    /diameter['стандартный размер обечайки'])**2
        
        diameter['действительная рабочая скорость низа'] = diameter['рабочая скорость пара низа'] *(diameter['диаметр низа'] 
                                                                                                    /diameter['стандартный размер обечайки'])**2
        
        diameter['% от предельной скорости верха'] =  diameter['действительная рабочая скорость верха'] / diameter['предельная скорость пара верха'] * 100
        
        diameter['% от предельной скорости низа'] =  diameter['действительная рабочая скорость низа'] / diameter['предельная скорость пара низа'] * 100
        
        return diameter
    
    def calculate_hight(balance, properties, diameter, xy_diagram, bottom, top, Substance, Ropt, PRESSURE, filling_name: str):
        
        filling = pd.DataFrame(columns = ['удельная поверхность','свободный объем','насыпная плотность'], dtype=float)
        filling.loc['25x25x3'] = [200, 0.74, 530]
        filling.loc['35x35x4'] = [140, 0.78, 530]
        filling.loc['50x50x5'] = [87.5, 0.785, 530]
        
        filling_phi = pd.DataFrame()
        filling_phi['x'] = 10**np.array([3,3.4,3.6,3.8,4,4.2,4.5,4.7])
        filling_phi['25x25x3'] = 10**np.array([-1.61,-1.61,-1.61,-1.6,-1.58,-1.55,-1.35,-1.3])
        filling_phi['35x35x4'] = 10**np.array([-1.8,-1.6,-1.55,-1.5,-1.4,-1.3,-1.2,-1.1])
        filling_phi['50x50x5'] = 10**np.array([-1.4,-1.3,-1.25,-1.2,-1.17,-1.1,-1.08,-1.05])

        filling_psi = pd.DataFrame()
        filling_psi['x'] = [10, 20, 30, 40, 50, 60, 70, 80]
        filling_psi['25x25x3'] = [40, 75, 90, 100, 95, 90, 85, 75]
        filling_psi['35x35x4'] = [80, 120, 140, 150, 150, 155, 160, 160]
        filling_psi['50x50x5'] = [120, 175, 190, 200, 205, 210, 210, 210]
        filling_psi['c'] = [1, 1, 1, 1, 0.95, 0.9, 0.8, 0.6]
        
        f_phi = dfc.get_coeffs(filling_phi['x'],filling_phi[filling_name])
        f_psi = dfc.get_coeffs(filling_psi['x'],filling_psi[filling_name])
        f_c = dfc.get_coeffs(filling_psi['x'],filling_psi['c'])
        
        phi_top = np.poly1d(f_phi)(np.log10(diameter['массовая нагрузка жидкости верха']*3600))
        phi_bottom = np.poly1d(f_phi)(np.log10(diameter['массовая нагрузка жидкости низа']*3600))
        
        psi_top = np.poly1d(f_psi)(diameter['% от предельной скорости верха'])
        psi_bottom = np.poly1d(f_psi)(diameter['% от предельной скорости низа'])
        
        c_top = np.poly1d(f_c)(diameter['% от предельной скорости верха'])
        c_bottom = np.poly1d(f_c)(diameter['% от предельной скорости низа'])
        
        variables = pd.Series(dtype=float)
        
        u_a = Calculations.get_value(component= Substance['A'], attribute='vicosity_organic_liquid', temperature=20)
        u_b = Calculations.get_value(component= Substance['B'], attribute='vicosity_organic_liquid', temperature=20)
        variables['вязкость жидкости верха при 20°С'] = (u_a*properties['содержание легколетучего в жидкости']['верха'] 
                                                        + u_b*(1-properties['содержание легколетучего в жидкости']['верха']))
        
        variables['вязкость жидкости низа при 20°С'] = (u_a*properties['содержание легколетучего в жидкости']['низа'] 
                                                        + u_b*(1-properties['содержание легколетучего в жидкости']['низа']))
        
        p_a = Calculations.get_value(component= Substance['A'], attribute='density_organic_liquid', temperature=20)
        p_b = Calculations.get_value(component= Substance['B'], attribute='density_organic_liquid', temperature=20)
        variables['плотность жидкости верха при 20°С'] = (p_a*properties['содержание легколетучего в жидкости']['верха'] 
                                                        + p_b*(1-properties['содержание легколетучего в жидкости']['верха']))
        
        variables['плотность жидкости низа при 20°С'] = (p_a*properties['содержание легколетучего в жидкости']['низа'] 
                                                        + p_b*(1-properties['содержание легколетучего в жидкости']['низа']))
        
        def get_m(x): return np.poly1d(xy_diagram)(x)/x
        x_bottom = np.linspace(balance['xw'],balance['xf'],100)
        x_top = np.linspace(balance['xf'],balance['xp'],100)
        
        hight = pd.Series(dtype=float)
        hight['отношение нагрузок пар/жидкость верха'] = (Ropt+1)/Ropt

        hight['отношение нагрузок пар/жидкость низа'] = (Ropt+1)/(Ropt +(balance['массовый расход в питателе'] * properties['молярная масса жидкости']['дистиллята']
                                                            /(balance['массовый расход в дефлегматоре'] * properties['молярная масса жидкости']['питания'])))

        hight['коэффициент диффузии жидкости верха при 20°С'] = np.double([10**(-6)]) / (np.sqrt(variables['вязкость жидкости верха при 20°С'])
        *(properties['молярный объем жидкости']['дистиллята']**(1/3) + properties['молярный объем жидкости']['куба']**(1/3))**2) *np.sqrt(
            1/properties['молярная масса жидкости']['дистиллята'] + 1/properties['молярная масса жидкости']['куба'])
        
        hight['коэффициент диффузии жидкости низа при 20°С'] = np.double([10**(-6)]) / (np.sqrt(variables['вязкость жидкости низа при 20°С'])
        *(properties['молярный объем жидкости']['дистиллята']**(1/3) + properties['молярный объем жидкости']['куба']**(1/3))**2) *np.sqrt(
            1/properties['молярная масса жидкости']['дистиллята'] + 1/properties['молярная масса жидкости']['куба'])
        
        hight['температурный коэффициент верха'] = (0.2 * np.sqrt(variables['вязкость жидкости верха при 20°С'])
                                                    / variables['плотность жидкости верха при 20°С']**(1/3))
        
        hight['температурный коэффициент низа'] = (0.2 * np.sqrt(variables['вязкость жидкости низа при 20°С'])
                                                    / variables['плотность жидкости низа при 20°С']**(1/3))
        
        hight['коэффициент диффузии жидкости низа'] = (hight['коэффициент диффузии жидкости низа при 20°С'] 
                            *(1 + hight['температурный коэффициент низа']*(properties['температура']['низа'] - 20)))
                                                                                                                
        hight['коэффициент диффузии жидкости верха'] = (hight['коэффициент диффузии жидкости верха при 20°С'] 
                            *(1 + hight['температурный коэффициент верха']*(properties['температура']['верха'] - 20)))
        
            
        hight['коэффициент диффузии пара верха'] = (np.double(4.22*10**(-2)) * (np.double(273) + properties['температура']['верха'])**(3/2)
            /(PRESSURE * (properties['молярный объем жидкости']['дистиллята']**(1/3) + properties['молярный объем жидкости']['куба']**(1/3))**2)*np.sqrt(
                1/properties['молярная масса жидкости']['дистиллята'] + 1/properties['молярная масса жидкости']['куба']))
        
        hight['коэффициент диффузии пара низа'] = (np.double(4.22*10**(-2)) * (np.double(273) + properties['температура']['низа'])**(3/2)
            /(PRESSURE * (properties['молярный объем жидкости']['дистиллята']**(1/3) + properties['молярный объем жидкости']['куба']**(1/3))**2)*np.sqrt(
                1/properties['молярная масса жидкости']['дистиллята'] + 1/properties['молярная масса жидкости']['куба']))
        
        hight['средний коэффициент распределения верха'] = get_m(x_top).mean()
        hight['средний коэффициент распределения низа'] = get_m(x_bottom).mean()
        
        hight['критерий Прандтля жидости верха'] = (properties['вязкость жидкости']['верха']*10**-3/(properties['плотность жидкости']['верха'] 
                                                                                    *hight['коэффициент диффузии жидкости верха']))
        
        hight['критерий Прандтля жидкости низа'] = properties['вязкость жидкости']['низа']*10**-3/(properties['плотность жидкости']['низа'] 
                                                                                    *hight['коэффициент диффузии жидкости низа'])
        
        hight['высота единицы переноса жидкости верха'] = 0.258 * phi_top * c_top * hight['критерий Прандтля жидости верха']**(1/2) * 3**0.15
        hight['высота единицы переноса жидкости низа'] = 0.258 * phi_bottom * c_bottom * hight['критерий Прандтля жидкости низа']**(1/2) * 3**0.15
        
        if diameter['стандартный размер обечайки'] > 0.8:
            d = 1.24
        else:
            d = 1
            
        hight['критерий Прандтля пара верха'] = properties['вязкость пара']['верха']*10**-3/(properties['плотность пара']['верха'] 
                                                                                    *hight['коэффициент диффузии пара верха'])
        hight['критерий Прандтля пара низа'] = properties['вязкость пара']['низа']*10**-3/(properties['плотность пара']['низа'] 
                                                                                    *hight['коэффициент диффузии пара низа'])
        
        hight['массовая плотность орошения верха'] = diameter['массовая нагрузка жидкости верха'] / (0.785 * diameter['стандартный размер обечайки']**2)
        hight['массовая плотность орошения низа'] = diameter['массовая нагрузка жидкости низа'] / (0.785 * diameter['стандартный размер обечайки']**2)
        
        hight['высота единицы переноса пара верха'] = ((0.0175 * psi_top * hight['критерий Прандтля пара верха'] 
                                                        *diameter['стандартный размер обечайки']**d * 3**0.33)
            /((hight['массовая плотность орошения верха'] * properties['вязкость жидкости']['верха']**0.16
                *(1000/properties['плотность жидкости']['верха'])**1.25 * (((72.8*10**-3)
                /(properties['поверхностное натяжение жидкости']['верха']/1000))**0.8)))**0.6)
                                                    
        hight['высота единицы переноса пара низа'] = ((0.0175 * psi_bottom * hight['критерий Прандтля пара низа'] 
                                                        *diameter['стандартный размер обечайки']**d * 3**0.33)
            /((hight['массовая плотность орошения низа'] * properties['вязкость жидкости']['низа']**0.16
                *(1000/properties['плотность жидкости']['низа'])**1.25 * (((72.8*10**-3)
                /(properties['поверхностное натяжение жидкости']['низа']/1000))**0.8)))**0.6)
        
        hight['общая высота единицы переноса верха'] = (hight['высота единицы переноса пара верха'] + hight['средний коэффициент распределения верха']
                                                        *hight['отношение нагрузок пар/жидкость верха'] * hight['высота единицы переноса жидкости верха'])
        
        hight['общая высота единицы переноса низа'] = (hight['высота единицы переноса пара низа'] + hight['средний коэффициент распределения низа']
                                                       *hight['отношение нагрузок пар/жидкость низа'] * hight['высота единицы переноса жидкости низа'])
        
        hight['высота насадки верха'] = hight['общая высота единицы переноса верха'] * top
        hight['высота насадки низа'] = hight['общая высота единицы переноса низа'] * bottom
        hight['общая высота насадки в колонне'] = hight['высота насадки верха'] + hight['высота насадки низа']
        
        if diameter['стандартный размер обечайки'] <= 1:
            Zv = 0.6
            Zn = 1.5
        elif diameter['стандартный размер обечайки'] >= 2.4:
            Zv = 1.4
            Zn = 2.5
        else:
            Zv = 1
            Zn = 2
        
        n =   np.round(hight['высота насадки верха']/3) + np.round(hight['высота насадки низа']/3)
        hight['общая высота колонны'] = 3*n +(n-1)*0.5 +Zv +Zn
    
        return hight
    
    def calculate_thermal_balance(balance, properties, Ropt):
        thermal_balance = pd.Series(dtype='float64')
        thermal_balance['теплота забираемая водой в дефлегматоре'] = (balance['массовый расход в дефлегматоре'] 
                                                        *(1 + Ropt) * properties['теплота парообразования жидкости']['дистиллята'])

        thermal_balance['теплота передаваемая паром от испарителя'] = (balance['массовый расход в питателе'] 
                                                                       * properties['теплота парообразования жидкости']['питания'])


        thermal_balance['теплота исходной смеси'] = (balance['массовый расход в питателе'] 
                              *properties['температура']['питания'] * properties['удельная теплоемкость жидкости']['питания']/1000)

        thermal_balance['теплота кубовой жидкости'] = (balance['массовый расход в кубовом остатке'] 
                                    *properties['температура']['куба'] * properties['удельная теплоемкость жидкости']['куба']/1000)

        thermal_balance['теплота дистиллята'] = (balance['массовый расход в дефлегматоре'] 
                        * properties['температура']['дистиллята'] * properties['удельная теплоемкость жидкости']['дистиллята']/1000)

        thermal_balance['теплота получаемая кипящей жидкостью'] = (thermal_balance['теплота забираемая водой в дефлегматоре']
                                                                +thermal_balance['теплота дистиллята'] + thermal_balance['теплота кубовой жидкости']
                                                                -thermal_balance['теплота исходной смеси'])
        return thermal_balance
