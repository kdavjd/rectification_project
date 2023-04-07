import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from .data_functions import DataFunctions as dfc
from .column_functions import Calculations as clc


class Calculations():
    
    def get_heater(
        row,
        name,
        aqua_vapor_saturation_by_pressure,
        aqua_liquid_saturation,
        aqua_vapor_saturation,
        balance,
        properties,
        FEED_TEMPERATURE,
        Ropt,
        HEATER_NAME = 'подогреватель',
        ORIENTACION = 'вертикальный',
        AQ_PRESSURE = 3,
        call = 'auto'):
                    
                    
        heat = pd.Series(dtype=float)
        calc = pd.Series(dtype=float)

        x=pd.Series([0.008, 0.008, 0.023])
        y=pd.Series([0.9, 0.9, 0.8])
        Re=pd.Series([0.0, 2299.0, 9999.0])

        t = aqua_vapor_saturation_by_pressure[aqua_vapor_saturation_by_pressure['pressure_e-5'] >= AQ_PRESSURE]['temperature'].min()

        aq = aqua_liquid_saturation.loc[aqua_liquid_saturation[aqua_liquid_saturation['temperature'] >= t].index.min()]
        vap = aqua_vapor_saturation[aqua_vapor_saturation['temperature'] >= t]['specific_heat_vaporization'].min()

        heat['диаметр кожуха'] = row['D кожуха, мм']
        heat['внутренний диаметр труб'] = row['d труб, мм']
        heat['внешний диаметр труб'] = row['d труб, мм'] + np.double(0.004)
        heat['число ходов'] = row['Число ходов*']
        heat['число труб'] = row['Общее число труб, шт.']
        heat['длинна труб'] = np.double(name)
        if call == 'auto':
            heat['поверхность теплообмена'] = row[row.keys() == name]
        elif call =='app':
            heat['поверхность теплообмена'] = row[row.keys() == name].values
        else:
            heat['поверхность теплообмена'] = row[row.keys() == name].values

        if HEATER_NAME == 'подогреватель':
            heater_point = 'питания'
            heater_consumption = balance['массовый расход в питателе']
        elif HEATER_NAME == 'дефлегматор':
            heater_point = 'дистиллята'
            heater_consumption = float(balance['массовый расход в дефлегматоре'] * (Ropt+1))
            
            
        calc['cредняя движущая сила теплопередачи'] = (
            ((t-FEED_TEMPERATURE)-(t-properties['температура'][heater_point]))
            /np.log((t-FEED_TEMPERATURE)/(t-properties['температура'][heater_point])))
        
        calc['тепловой поток в подогревателе'] = (
            heater_consumption
            *(properties['температура'][heater_point]-FEED_TEMPERATURE)
            *properties['удельная теплоемкость жидкости'][heater_point])

        calc['расход пара на подогрев'] = calc['тепловой поток в подогревателе']/vap

        calc['критерий Рейнольдса'] = (
            (4 * heater_consumption * heat['число ходов'])
            /(np.pi*(properties['вязкость жидкости'][heater_point]/1000)
              *heat['внутренний диаметр труб']*heat['число труб']))

        xpd = x[calc['критерий Рейнольдса'] > Re].max()
        ypd = y[calc['критерий Рейнольдса'] > Re].max()
        
        calc['критерий Прандтля'] = (
            properties['удельная теплоемкость жидкости'][heater_point]
            *(properties['вязкость жидкости'][heater_point]/1000)
            /properties['теплопроводность жидкости'][heater_point])

        calc['коэффициент теплоотдачи в трубах']= (
            (properties['теплопроводность жидкости'][heater_point]
             /heat['внутренний диаметр труб'])
            *(xpd*(calc['критерий Рейнольдса']**ypd)
              *calc['критерий Прандтля']**0.43))

        if ORIENTACION == 'вертикальный':
            calc['коэффициент теплопередачи в межтрубном']=(
                3.78*aq['thermal_conductivity']
                *(((aq['density']**2)
                   *heat['внешний диаметр труб']
                   *heat['число труб'])
                  /(aq['viscosity_kilo']/1000*calc['расход пара на подогрев']))**(1/3))
            
        elif ORIENTACION == 'горизонтальный':
                        
            Ppd = 0.7 if heat['число труб'] <= 100 else 0.6
            
            calc['коэффициент теплопередачи в межтрубном'] = (
                2.02*Ppd*aq['thermal_conductivity']
                *(((aq['density']**2)*heat['длинна труб']*heat['число труб'])
                  /(aq['viscosity_kilo']/1000*calc['расход пара на подогрев']))**(1/3))

        else:
            print("теплообменник либо 'вертикальный' либо 'горизонтальный', посчитан как вертикальный")
            calc['коэффициент теплопередачи в межтрубном']=(
                3.78*aq['thermal_conductivity']
                *(((aq['density']**2)
                   *heat['внешний диаметр труб']
                   *heat['число труб'])
                  /(aq['viscosity_kilo']/1000
                    *calc['расход пара на подогрев']))**(1/3))

        calc['сумма термических сопротивлений'] = (0.002/17.5)+1/5800+1/5800
        calc['коэффициент теплопередачи']=(
            1/(1/calc['коэффициент теплоотдачи в трубах']
               +calc['сумма термических сопротивлений']
               +1/calc['коэффициент теплопередачи в межтрубном']))

        calc['требуемая поверхность теплообмена'] = (
            calc['тепловой поток в подогревателе']
            /(calc['коэффициент теплопередачи']*calc['cредняя движущая сила теплопередачи']))

        calc['запас поверхности, %']=(
            (np.double(heat['поверхность теплообмена'])-calc['требуемая поверхность теплообмена'])
            /heat['поверхность теплообмена']
            *100)
        
        if call == 'auto':
            return calc['запас поверхности, %']
        else:
            return calc
        
        
    def get_capacitor(
        row,
        name,        
        aqua_liquid_saturation,        
        thermal_balance,
        balance,
        properties,        
        Ropt,
        Tn=15,
        Tk=30,
        ORIENTACION = 'вертикальный',
        call = 'auto'):
        
        x=pd.Series([0.008, 0.008, 0.023])
        y=pd.Series([0.9, 0.9, 0.8])
        Re=pd.Series([0.0, 2299.0, 9999.0])
        heat = pd.Series(dtype=float)
        calc = pd.Series(dtype=float)
        
        heat['диаметр кожуха'] = row['D кожуха, мм']
        heat['внутренний диаметр труб'] = row['d труб, мм']
        heat['внешний диаметр труб'] = row['d труб, мм'] + np.double(0.004)
        heat['число ходов'] = row['Число ходов*']
        heat['число труб'] = row['Общее число труб, шт.']
        heat['длинна труб'] = np.double(name)
        if call == 'auto':
            heat['поверхность теплообмена'] = row[row.keys() == name]
        else:
            heat['поверхность теплообмена'] = row[row.keys() == name].values
        
        heater_point = 'дистиллята'
        heater_consumption = float(balance['массовый расход в дефлегматоре'] * (Ropt+1))
        
        t = (Tn + Tk)/2
        aq = aqua_liquid_saturation.loc[aqua_liquid_saturation[aqua_liquid_saturation['temperature'] >= t].index.min()]
        
        calc['расход воды'] = (
            thermal_balance['теплота забираемая водой в дефлегматоре']*1000
            /properties['удельная теплоемкость жидкости'][heater_point]
            /t)
        
        calc['cредняя движущая сила теплопередачи'] = (
                ((properties['температура'][heater_point]-Tn)-(properties['температура'][heater_point]-Tk))
                /np.log((properties['температура'][heater_point]-Tn)/(properties['температура'][heater_point]-Tk)))
        
        if ORIENTACION == 'вертикальный' or ORIENTACION != 'горизонтальный':
            if ORIENTACION != 'вертикальный':
                print("теплообменник либо 'вертикальный' либо 'горизонтальный', посчитан как вертикальный")
            calc['коэффициент теплопередачи в межтрубном']=(
                3.78*properties['теплопроводность жидкости'][heater_point]
                *(((properties['плотность жидкости'][heater_point]**2)
                    *heat['внешний диаметр труб']
                    *heat['число труб'])
                    /(properties['вязкость жидкости'][heater_point]/1000*heater_consumption))**(1/3))
                    
        elif ORIENTACION == 'горизонтальный':
            Ppd = 0.7 if heat['число труб'] <= 100 else 0.6
            calc['коэффициент теплопередачи в межтрубном'] = (
                2.02*Ppd*properties['теплопроводность жидкости'][heater_point]
                *(((properties['плотность жидкости'][heater_point]**2)
                *heat['длинна труб']
                *heat['число труб'])
                    /(properties['вязкость жидкости'][heater_point]/1000*heater_consumption))**(1/3))    
            
        calc['критерий Рейнольдса'] = (
            (4 * calc['расход воды'] * heat['число ходов'])
            /(np.pi*(aq['viscosity_kilo']/1000)
            *heat['внутренний диаметр труб']*heat['число труб']))
        
        xpd = x[float(calc['критерий Рейнольдса']) > Re].max()
        ypd = y[float(calc['критерий Рейнольдса']) > Re].max()
        
        calc['критерий Прандтля'] = (
            aq['specific_heat_capacity']
            *(aq['viscosity_kilo']/1000)
            /aq['thermal_conductivity'])
        
        calc['коэффициент теплоотдачи в трубах']= (
                (aq['thermal_conductivity']
                /heat['внутренний диаметр труб'])
                *(xpd*(calc['критерий Рейнольдса']**ypd)
                *calc['критерий Прандтля']**0.43))
        
        calc['сумма термических сопротивлений'] = (0.002/17.5)+1/5800+1/5800
        
        calc['коэффициент теплопередачи']=(
            1/(1/calc['коэффициент теплоотдачи в трубах']
                +calc['сумма термических сопротивлений']
                +1/calc['коэффициент теплопередачи в межтрубном']))

        calc['требуемая поверхность теплообмена'] = (
            thermal_balance['теплота забираемая водой в дефлегматоре']*1000
            /(calc['коэффициент теплопередачи']*calc['cредняя движущая сила теплопередачи']))

        calc['запас поверхности, %']=(
            (heat['поверхность теплообмена']-calc['требуемая поверхность теплообмена'])
            /heat['поверхность теплообмена']
            *100)
        
        if call == 'auto':
                return calc['запас поверхности, %']
        else:
            return calc
        
        
    def get_diameter(col):
        new_row = []
                
        for i in col:
            if i == '20×2':
                new_row.append(0.016)
            elif i == '25×2':
                new_row.append(0.021)
            else:
                new_row.append(np.NaN)
                
        return new_row
    
    
    def calculate_equipment(
        heaters_table,
        aqua_vapor_saturation_by_pressure, 
        aqua_liquid_saturation, 
        aqua_vapor_saturation, 
        balance,
        properties,
        FEED_TEMPERATURE,
        thermal_balance,
        Ropt,
        EQ_NAME,                        
        ORIENTACION = 'вертикальный',
        AQ_PRESSURE = 3,
        pipes = 'продукт',
        COOLER_NAME = 'дистиллята',
        aq_t = 20,
        tk = 30,
        Tn = 15,
        Tk = 30
        ):
        
        def get_heater_index(row):
            i = (str(int(row[0]))
                    + ' ' + str(row[1] + np.double(0.004))
                    + ' ' + str(int(row[2]))
                    + ' ' + str(int(row[3])))
            return i
        
        def get_evaporator_index(row):
            i = (str(int(row['D кожуха, мм']))
                    + ' ' + str(row['d труб, мм'] + np.double(0.004))
                    + ' ' + str(int(row['Число ходов']))
                    + ' ' + str(int(row['Общее число труб, шт'])))
            return i
        
        exclude_list = ['d труб, мм']
        heaters_table = dfc.delete_hyphens(heaters_table, exclude_list)
        
        pipes_names = ['1', '1.5', '2', '3', '4', '6', '9']
        evaporator_pipes_names = [2,3,4,6]
        
        if EQ_NAME == 'испаритель':
            heaters = pd.DataFrame(columns=evaporator_pipes_names)
            heaters['name'] = heaters_table.apply(get_evaporator_index, axis=1)
        else:
            heaters = pd.DataFrame(columns=pipes_names)
            heaters['name'] = heaters_table.apply(get_heater_index, axis=1)        
        
        if EQ_NAME == 'подогреватель':
            for name in pipes_names:
                heaters[name] = heaters_table.apply(Calculations.get_heater, axis = 1, args=(
                    name,
                    aqua_vapor_saturation_by_pressure, 
                    aqua_liquid_saturation, 
                    aqua_vapor_saturation, 
                    balance,
                    properties,
                    FEED_TEMPERATURE,
                    Ropt),
                    HEATER_NAME = EQ_NAME,
                    ORIENTACION = ORIENTACION,
                    AQ_PRESSURE = AQ_PRESSURE)
                
        elif EQ_NAME == 'дефлегматор':
            for name in pipes_names:
                heaters[name] = heaters_table.apply(Calculations.get_capacitor, axis = 1, args=(
                    name,        
                    aqua_liquid_saturation,        
                    thermal_balance,
                    balance,
                    properties,        
                    Ropt),
                    Tn=Tn,
                    Tk=Tk,
                    ORIENTACION = ORIENTACION,
                    )
            pass
                
        elif EQ_NAME == 'испаритель':
            for name in evaporator_pipes_names:
                heaters[name] = heaters_table.apply(Calculations.get_evaporator, axis = 1, args=(
                    name,
                    aqua_vapor_saturation_by_pressure, 
                    aqua_liquid_saturation,
                    aqua_vapor_saturation,
                    properties,
                    thermal_balance),
                    AQ_PRESSURE = AQ_PRESSURE)
                
        elif EQ_NAME == 'холодильник':
            for name in pipes_names:
                heaters[name] = heaters_table.apply(Calculations.get_cooler, axis = 1, args=(
                    name,
                    aqua_liquid_saturation,
                    aqua_vapor_saturation,
                    properties,balance),
                    COOLER_NAME = COOLER_NAME,
                    pipes = pipes,
                    aq_t = aq_t,
                    tk = tk,)
                
        heaters.index = heaters['name']
        heaters = heaters.drop('name', axis=1)
        return heaters
    
    
    def get_evaporator(
        row,
        name,
        aqua_vapor_saturation_by_pressure,
        aqua_liquid_saturation,
        aqua_vapor_saturation,
        properties,
        thermal_balance,
        AQ_PRESSURE = 3,
        call = 'auto'):
    
        t = aqua_vapor_saturation_by_pressure[aqua_vapor_saturation_by_pressure['pressure_e-5'] >= AQ_PRESSURE]['temperature'].min()
        aq = aqua_liquid_saturation.loc[aqua_liquid_saturation[aqua_liquid_saturation['temperature'] >= t].index.min()]
        vap = aqua_vapor_saturation.loc[aqua_vapor_saturation[aqua_vapor_saturation['temperature'] >= t].index.min()]
            
        calc = pd.Series(dtype=float)
        calc['диаметр кожуха'] = row['D кожуха, мм']
        calc['внутренний диаметр труб'] = row['d труб, мм']
        calc['внешний диаметр труб'] = row['d труб, мм'] + np.double(0.004)        
        calc['число ходов'] =  row['Число ходов']        
        calc['число труб'] = row['Общее число труб, шт']
        calc['длинна труб'] = np.double(name)
        if call == 'auto':
            calc['поверхность теплообмена'] = row[row.keys() == name]
        else:
            calc['поверхность теплообмена'] = row[row.keys() == name].values
            
        evaporator = pd.Series(dtype=float)
            
        evaporator['необходимый расход пара'] = (
            thermal_balance['теплота получаемая кипящей жидкостью']
            /vap['specific_heat_vaporization'])
        
        evaporator['cредняя движущая сила теплопередачи'] = t - properties['температура']['питания']
                
        evaporator['коэффициент теплоотдачи от пара к трубам'] = 1.21*aq['thermal_conductivity']*(
            ((aq['density']**2)*vap['specific_heat_vaporization']*1000*9.81)/(aq['viscosity_kilo']/1000*calc['длинна труб']))**(1/3)
        
        evaporator['коэффициент теплоотдачи от труб к жидкости в трубах'] = (
            780*((properties['теплопроводность жидкости']['куба']**1.3)
                *(properties['плотность жидкости']['куба']**0.5)
                *(properties['плотность пара']['куба']**0.06))
            /(((properties['поверхностное натяжение жидкости']['куба']/1000)**0.5)
            *((properties['теплота парообразования жидкости']['куба']*1000)**0.6)
            *(properties['плотность пара']['куба']**0.66)
            *(properties['удельная теплоемкость жидкости']['куба']**0.3)
            *((properties['вязкость жидкости']['куба']/1000)**0.3)))
        
        evaporator['сумма термических сопротивлений'] = (0.002/17.5)+1/5800+1/11600
        
        def f(x,a,b,c,d): return 1/a * x**(4/3) + b*x + 1/c*x**(0.4) - d

        x = np.linspace(0,100000,25000)
        y = f(
            x,
            evaporator['коэффициент теплоотдачи от пара к трубам'],
            evaporator['сумма термических сопротивлений'],
            evaporator['коэффициент теплоотдачи от труб к жидкости в трубах'],
            evaporator['cредняя движущая сила теплопередачи'])
        
        evaporator['удельная тепловая нагрузка'] = x[y == y[y>0].min()]
        
        evaporator['требуемая поверхность теплообмена'] = (
            thermal_balance['теплота получаемая кипящей жидкостью']*1000
            /evaporator['удельная тепловая нагрузка'])
        
        evaporator['запас поверхности, %'] = (
            (calc['поверхность теплообмена']-evaporator['требуемая поверхность теплообмена'])
            /calc['поверхность теплообмена']*100)
        
        if call == 'auto':
            return evaporator['запас поверхности, %']
        else:
            return evaporator
        
        
    def get_cooler(
        row,
        name,        
        aqua_liquid_saturation,
        aqua_vapor_saturation,
        properties,
        balance,
        COOLER_NAME = 'дистиллята',
        pipes = 'продукт',
        aq_t = 20,
        tk = 30,
        call = 'auto'):
        
        t = aq_t #средняя температура воды в течение года
        aq = aqua_liquid_saturation.loc[aqua_liquid_saturation[aqua_liquid_saturation['temperature'] >= t].index.min()]
        vap = aqua_vapor_saturation.loc[aqua_vapor_saturation[aqua_vapor_saturation['temperature'] >= t].index.min()]
        
        calc = pd.Series(dtype=float)
        calc['диаметр кожуха'] = row['D кожуха, мм']
        calc['внутренний диаметр труб'] = row['d труб, мм']
        calc['внешний диаметр труб'] = row['d труб, мм'] + np.double(0.004)
        calc['число ходов'] = row['Число ходов*']
        calc['число труб'] = row['Общее число труб, шт.']
        calc['длинна труб'] = np.double(name)
        if call == 'auto':
            calc['поверхность теплообмена'] = row[row.keys() == name]
        else:
            calc['поверхность теплообмена'] = row[row.keys() == name].values
        
        if COOLER_NAME == 'дистиллята':
            cooler_consumption = balance['массовый расход в дефлегматоре']
        elif COOLER_NAME == 'куба':            
            cooler_consumption = balance['массовый расход в кубовом остатке']
        else:
            print("COOLER_NAME может быть только 'дистиллята' или 'куба', посчитан для 'дистиллята' ")
            COOLER_NAME == 'дистиллята'
            cooler_consumption = balance['массовый расход в дефлегматоре']
            
        cooler = pd.Series(dtype=float)
        
        cooler['тепловой поток в холодильнике'] = (
            properties['удельная теплоемкость жидкости'][COOLER_NAME]
            *cooler_consumption
            *(properties['температура'][COOLER_NAME] - tk))
        
        cooler['расход воды'] = (
            cooler['тепловой поток в холодильнике']
            /aq['specific_heat_capacity']
            /((tk+1) - aq_t))
        
        cooler['cредняя движущая сила теплопередачи'] = (
            ((properties['температура'][COOLER_NAME]-tk)-((tk+1)-aq_t))
            /np.log((properties['температура'][COOLER_NAME]-tk)/((tk+1)-aq_t)))
        
        if calc['диаметр кожуха'] < 299:
            cooler['площадь сечения межтруб'] = 0.3*((np.pi*(calc['диаметр кожуха']/1000)**2)/4)
        else:
            cooler['площадь сечения межтруб'] = 0.16*((np.pi*(calc['диаметр кожуха']/1000)**2)/4)
                
        #В трубы можно пустить либо воду либо продукт
        if pipes == 'продукт' or pipes != 'вода':
            
            cooler['критерий Рейнольдса межтруб'] = (
                (cooler['расход воды']*calc['внешний диаметр труб'])
                /((aq['viscosity_kilo']/1000)*cooler['площадь сечения межтруб']))
            
            cooler['критерий Прандтля межтруб'] = (
                aq['specific_heat_capacity']
                *(aq['viscosity_kilo']/1000)
                /aq['thermal_conductivity'])
            
            if cooler['критерий Рейнольдса межтруб'] < 1000:
                cooler['коэффициент теплопередачи в межтрубном'] = (
                    (aq['thermal_conductivity']/calc['внешний диаметр труб'])
                    *0.4*0.6*(cooler['критерий Рейнольдса межтруб']**0.6)
                    *cooler['критерий Прандтля межтруб']**0.36)
            else:
                cooler['коэффициент теплопередачи в межтрубном'] = (
                    (aq['thermal_conductivity']/calc['внешний диаметр труб'])
                    *0.56*0.6*(cooler['критерий Рейнольдса межтруб']**0.5)
                    *cooler['критерий Прандтля межтруб']**0.36)
            
            x=pd.Series([0.008, 0.008, 0.023])
            y=pd.Series([0.9, 0.9, 0.8])
            Re=pd.Series([0.0, 2299.0, 9999.0])
            
            cooler['критерий Рейнольдса в трубах']=(
                (4*cooler_consumption*calc['число ходов'])
                /(np.pi*(properties['вязкость жидкости'][COOLER_NAME]/1000)
                  *calc['внутренний диаметр труб']*calc['число труб']))
            
            cooler['критерий Прандтля в трубах'] = (
                properties['удельная теплоемкость жидкости'][COOLER_NAME]
                *(properties['вязкость жидкости'][COOLER_NAME]/1000)
                /properties['теплопроводность жидкости'][COOLER_NAME])
            
            xclr = x[cooler['критерий Рейнольдса в трубах'] > Re].max()
            yclr = y[cooler['критерий Рейнольдса в трубах'] > Re].max()
            
            cooler['коэффициент теплопередачи в трубах'] = (
                (properties['теплопроводность жидкости'][COOLER_NAME]
                 /calc['внутренний диаметр труб'])
                *xclr*(cooler['критерий Рейнольдса в трубах']**yclr)
                *cooler['критерий Прандтля в трубах']**0.43)
            
            cooler['сумма термических сопротивлений']=(0.002/17.5)+1/5800+1/5800
            
            cooler['коэффициент теплоотдачи'] = (
                1/(1/cooler['коэффициент теплопередачи в трубах']
                   +cooler['сумма термических сопротивлений']
                   +1/cooler['коэффициент теплопередачи в межтрубном']))
            
            cooler['требуемая поверхность теплообмена'] = (
                cooler['тепловой поток в холодильнике']
                /(cooler['коэффициент теплоотдачи']
                  *cooler['cредняя движущая сила теплопередачи']))
            
            cooler['запас поверхности, %'] = (
                (calc['поверхность теплообмена']-cooler['требуемая поверхность теплообмена'])
                /calc['поверхность теплообмена']*100)
        
        elif pipes == 'вода':
            cooler['критерий Рейнольдса межтруб'] = (
                (cooler_consumption*calc['внешний диаметр труб'])
                /((properties['вязкость жидкости'][COOLER_NAME]/1000)
                  *cooler['площадь сечения межтруб']))
            
            cooler['критерий Прандтля межтруб'] = (
                properties['удельная теплоемкость жидкости'][COOLER_NAME]
                *(properties['вязкость жидкости'][COOLER_NAME]/1000)
                /properties['теплопроводность жидкости'][COOLER_NAME])
            
            if cooler['критерий Рейнольдса межтруб'] < 1000:
                cooler['коэффициент теплопередачи в межтрубном'] = (
                    (properties['теплопроводность жидкости'][COOLER_NAME]/calc['внешний диаметр труб'])
                    *0.4*0.6*(cooler['критерий Рейнольдса межтруб']**0.6)
                    *cooler['критерий Прандтля межтруб']**0.36)
            else:
                cooler['коэффициент теплопередачи в межтрубном'] = (
                    (properties['теплопроводность жидкости'][COOLER_NAME]/calc['внешний диаметр труб'])
                    *0.56*0.6*(cooler['критерий Рейнольдса межтруб']**0.5)
                    *cooler['критерий Прандтля межтруб']**0.36)
            
            x=pd.Series([0.008, 0.008, 0.023])
            y=pd.Series([0.9, 0.9, 0.8])
            Re=pd.Series([0.0, 2299.0, 9999.0])
            
            cooler['критерий Рейнольдса в трубах']=(
                (4*cooler['расход воды']*calc['число ходов'])
                /(np.pi*(aq['viscosity_kilo']/1000)
                  *calc['внутренний диаметр труб']*calc['число труб']))
            
            cooler['критерий Прандтля в трубах'] = (
                aq['specific_heat_capacity']
                *(aq['viscosity_kilo']/1000)
                /aq['thermal_conductivity'])
            
            xclr = x[cooler['критерий Рейнольдса в трубах'] > Re].max()
            yclr = y[cooler['критерий Рейнольдса в трубах'] > Re].max()
            
            cooler['коэффициент теплопередачи в трубах'] = (
                (aq['thermal_conductivity']/calc['внутренний диаметр труб'])
                *xclr*(cooler['критерий Рейнольдса в трубах']**yclr)
                *cooler['критерий Прандтля в трубах']**0.43)
            
            cooler['сумма термических сопротивлений']=(0.002/17.5)+1/5800+1/5800
            
            cooler['коэффициент теплоотдачи'] = (
                1/(1/cooler['коэффициент теплопередачи в трубах']
                   +cooler['сумма термических сопротивлений']
                   +1/cooler['коэффициент теплопередачи в межтрубном']))
            
            cooler['требуемая поверхность теплообмена'] = (
                cooler['тепловой поток в холодильнике']
                /(cooler['коэффициент теплоотдачи']
                  *cooler['cредняя движущая сила теплопередачи']))
            
            cooler['запас поверхности, %'] = (
                (calc['поверхность теплообмена']-cooler['требуемая поверхность теплообмена'])
                /calc['поверхность теплообмена']*100)
            
        if call == 'auto':
            return cooler['запас поверхности, %']
        else:
            return cooler
