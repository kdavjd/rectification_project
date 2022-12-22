import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import integrate
from data_functions import DataFunctions as dfc


class Calculations():
    
    def get_plate_column_height(Ropt, balance, kinetic_frame, diagram, diameter, plot_type='matplotlib'):
        kinematic_diagram=dfc.get_coeffs(
            kinetic_frame['значение кинетической кривой'].index,
            kinetic_frame['значение кинетической кривой'].values)

        kinetic_xy = dfc.get_fit(
            kinetic_frame['значение кинетической кривой'].index,
            kinetic_frame['значение кинетической кривой'].values)
        
        diagram_xy = dfc.get_fit(diagram['x'], diagram['y'])
        
        yf = float(Ropt/(Ropt+1)*balance['xf']+balance['xp']/(Ropt+1))
        xw = float(balance['xw'])
        xf = float(balance['xf'])
        xp = float(balance['xp'])
        
        _x = [xw, xf]
        x_ = [xf, xp]
        _y = [xw, yf]
        y_ = [yf, xp]
        
        def biuld_phlegm_lines():
                nonlocal _x
                nonlocal _y
                nonlocal x_
                nonlocal y_
                return np.polyfit(_x,_y, 1), np.polyfit(x_,y_, 1)
        
        step = []
        step.append(np.poly1d(kinematic_diagram)(float(balance['xw'])))
        platform = []

        while step[-1] <= yf:
            bottom_work_line, top_work_line = biuld_phlegm_lines()
            bottom_work_line[1] = bottom_work_line[1] - step[-1]
            platform.append(np.roots(bottom_work_line))
            step.append(np.poly1d(kinematic_diagram)(platform[-1]))
            
        while step[-1] <= xp:
            bottom_work_line, top_work_line = biuld_phlegm_lines()
            top_work_line[1] = top_work_line[1] - step[-1]
            platform.append(np.roots(top_work_line))
            step.append(np.poly1d(kinematic_diagram)(platform[-1]))
            
        if step[-1] > xp:        
            platform = [float(xw)] + platform 
            outside_corners = list(zip(platform,step)) #Получаем координаты вершин внешних углов лестницы
            platform = platform[1::]
            inside_corners = list(zip(platform,step)) #Получаем координаты вершин внутренних углов лестницы
            stair = list(zip(outside_corners,inside_corners)) 
            sort_corners = []
            for x,y in stair:
                sort_corners.append(x)
                sort_corners.append(y)
            stair_line_x,stair_line_y = zip(*sort_corners) #Получаем значения ломаной линии по абсцисе и ординате
            W_line = xw, yf
            P_line = yf, xp
            
            if diameter['стандартный размер обечайки'] <= 1:
                Zv = 0.6
                Zn = 1.5
            elif diameter['стандартный размер обечайки'] >= 2.4:
                Zv = 1.4
                Zn = 2.5
            else:
                Zv = 1
                Zn = 2
                
        _ = [0, 1]
        if plot_type == 'matplotlib':
            
            fig = plt.figure(figsize=(8,8))
            axes = fig.add_subplot()    
            axes.plot(stair_line_x,stair_line_y, 'o--', lw=1, ms=1)
            axes.plot(_, _, color='black', lw=0.5)
            axes.plot(kinetic_frame['значение кинетической кривой'].index, kinetic_xy, color='red', lw=1, ms=2)
            axes.plot(diagram['x'], diagram_xy, 'o--', color='black', lw=0.5, ms=0.5)
            axes.plot(_x, W_line, color='green', lw=1, ms=2)
            axes.plot(x_, P_line, color='green', lw=1, ms=2)
            axes.set_title(f"N = {len(step)} высота колонны = {((len(step) - 1)*0.5+Zv+Zn)}")
            
            return pd.Series(
                {'общее число действительных тарелок':len(step),
                 'высота колонны':((len(step) - 1)*0.5+Zv+Zn)})
        
        if plot_type=='plotly':
            
            fig = Figures.plot_plate_column_height(stair_line_x, stair_line_y, diagram, diagram_xy, kinetic_frame, kinetic_xy,
                            _x, W_line, x_, P_line, _)
            
            fig.update_layout(
                autosize=False,
                width=500,
                height=500,
                margin=dict(l=20, r=5, t=30, b=2),
                showlegend=False,
                plot_bgcolor='white',
                title_text=f"N = {len(step)} высота колонны = {((len(step) - 1)*0.5+Zv+Zn)} метра",
                title_x=0.5)
            
            return fig, pd.Series(
                {'общее число действительных тарелок':len(step),
                 'высота колонны':((len(step) - 1)*0.5+Zv+Zn)})
    
    def calculate_kinetic_slice(x, xy_diagram, plate_coeffs, balance, properties, plate, Ropt, diameter):
        
        #найдем производную функции как тангенс угла наклона касательной к функции
        def tg(x, dx=0.01):
            nonlocal xy_diagram
            return (np.polyval(xy_diagram, (x + dx)) - np.polyval(xy_diagram, x))/dx
        
        def mass_transfer_factor():
            nonlocal Ropt
            nonlocal values
            return values['производная функции равновестной кривой'] * (Ropt + 1)/Ropt
        
        #подробно в источнике [8] стр 202
        def bypass_fraction(loc = 'низа'):
            nonlocal properties
            nonlocal plate 
            Fs = plate['скорость пара в рабочем сечении тарелки'] * properties['плотность пара'][loc]**0.5
            O = dfc.get_coeffs([1, 1.5, 2, 2.5, 3],
                               [0.1, 0.1, 0.1, 0.15, 0.2])
            return np.polyval(O, Fs)
        
        #подробно в [1] стр 241-242
        def mixing_cells():
            nonlocal diameter
            Lt = np.sqrt(diameter['стандартный размер обечайки']**2 - 1.05**2)
            return Lt/0.35
        
        #подробно в [8] стр 194-195
        def liquid_entrainment(loc = 'верха'):
            nonlocal properties
            nonlocal diameter
            nonlocal plate_coeffs
            nonlocal plate
            e = dfc.get_coeffs([0.7, 0.8, 1.5, 2, 3, 4, 6, 10],
                            [1e-3, 1e-2, 0.8e-1, 0.9e-1, 1.2e-1, 1.3e-1, 1.4e-1, 1.5e-1])
            
            if loc == 'верха' or loc != 'низа':
                m = (
                    1.15/1000
                    *((properties['поверхностное натяжение жидкости']['верха']/1000)
                    /properties['плотность пара']['верха'])**0.295 
                    *((properties['плотность жидкости']['верха']-properties['плотность пара']['верха'])
                    /(properties['вязкость пара']['верха']/1000))**0.425)

                H = 0.3 if diameter['стандартный размер обечайки'] < 1.2 else 0.5
                hp =  plate_coeffs['высота светлого слоя жидкости верха']/(1 - plate_coeffs['паросодержание барботажного слоя верха'])
                Hc = H - hp
                return np.polyval(e, plate['скорость пара в рабочем сечении тарелки']/m/Hc)
                
            if loc == 'низа':
                m = (
                    1.15/1000
                    *((properties['поверхностное натяжение жидкости']['низа']/1000)
                    /properties['плотность пара']['низа'])**0.295 
                    *((properties['плотность жидкости']['низа']-properties['плотность пара']['низа'])
                    /(properties['вязкость пара']['низа']/1000))**0.425)

                H = 0.3 if diameter['стандартный размер обечайки'] < 1.2 else 0.5
                hp =  plate_coeffs['высота светлого слоя жидкости низа'] / (1 - plate_coeffs['паросодержание барботажного слоя низа'])
                Hc = H - hp
                return np.polyval(e, plate['скорость пара в рабочем сечении тарелки']/m/Hc)
            
        def get_kinetic_y(x, Emy, loc = 'низа'):
            nonlocal Ropt
            nonlocal balance
            nonlocal xy_diagram
            yf = Ropt/(Ropt+1)*balance['xf']+balance['xp']/(Ropt+1)
            _x = float(balance['xw']), float(balance['xf'])
            x_ = float(balance['xf']), float(balance['xp'])
            _y = float(balance['xw']), float(yf)
            y_ = float(yf), float(balance['xp'])
            
            RW_function = np.polyfit(_x,_y, 1)
            RP_function = np.polyfit(x_,y_, 1)
            
            if loc == 'низа' or loc != 'верха':
                return np.polyval(RW_function, x) + Emy*(np.polyval(xy_diagram, x) - np.polyval(RW_function, x))
            else:
                return np.polyval(RP_function, x) + Emy*(np.polyval(xy_diagram, x) - np.polyval(RP_function, x))
        
        
        values = pd.DataFrame(dtype=float)
        
        if x < balance['xf']:
            values['коэффициент массопередачи'] = (
                1/((1/plate_coeffs['коэффициент массоотдачи пара низа на кмоль'])
                +tg(x)/plate_coeffs['коэффициент массоотдачи жидкости низа на кмоль']))
            
            values['производная функции равновестной кривой'] = tg(x)
            
            values['общее число единиц переноса'] = (
                values['коэффициент массопередачи']
                *properties['молярная масса газа']['низа']
                /(plate['скорость пара в рабочем сечении тарелки']*properties['плотность пара']['низа']))
            
            values['локальная эффективность по пару'] = 1 - np.exp(-values['общее число единиц переноса'])
            
            values['B'] = (
                mass_transfer_factor()
                *(values['локальная эффективность по пару']+liquid_entrainment(loc='низа')/tg(x))
                /(1-bypass_fraction(loc='низа'))
                /(1+liquid_entrainment(loc='низа')/tg(x)*mass_transfer_factor()))
            
            values["Мёрфи 2"] = (
                values['локальная эффективность по пару']
                /values['B']
                *((1+values['B']/mixing_cells())**mixing_cells() - 1))
            
            values["Мёрфи 1"] = (
                values["Мёрфи 2"]
                /(1+mass_transfer_factor()*bypass_fraction(loc='низа')*values["Мёрфи 2"]
                /(1-bypass_fraction(loc='низа'))))
            
            values['эффективность по Мёрфи'] = (
                values["Мёрфи 1"]
                /(1+liquid_entrainment(loc='низа')*mass_transfer_factor()*values["Мёрфи 1"]
                /(tg(x)*(1-bypass_fraction(loc='низа')))))
            
            values['значение кинетической кривой'] = get_kinetic_y(x, values['эффективность по Мёрфи'], loc='низа')
            
        else:
            
            values['коэффициент массопередачи'] = (
                1/((1/plate_coeffs['коэффициент массоотдачи пара верха на кмоль'])
                +tg(x)/plate_coeffs['коэффициент массоотдачи жидкости верха на кмоль']))
            
            values['производная функции равновестной кривой'] = tg(x)
            
            values['общее число единиц переноса'] = (
                values['коэффициент массопередачи']
                *properties['молярная масса газа']['верха']
                /(plate['скорость пара в рабочем сечении тарелки']*properties['плотность пара']['верха']))
            
            values['локальная эффективность по пару'] = 1 - np.exp(-values['общее число единиц переноса'])
            
            values['B'] = (
                mass_transfer_factor()
                *(values['локальная эффективность по пару']+liquid_entrainment(loc='верха')/tg(x))
                /(1-bypass_fraction(loc='верха'))
                /(1+liquid_entrainment(loc='верха')/tg(x)*mass_transfer_factor()))
            
            values["Мёрфи 2"] = (
                values['локальная эффективность по пару']
                /values['B']
                *((1+values['B']/mixing_cells())**mixing_cells() - 1))
            
            values["Мёрфи 1"] = (
                values['Мёрфи 2']
                /(1+mass_transfer_factor() * bypass_fraction(loc='верха') * values["Мёрфи 2"]
                /(1-bypass_fraction(loc='верха'))))
            
            values['эффективность по Мёрфи'] = (
                values["Мёрфи 1"]
                /(1+liquid_entrainment(loc='верха')*mass_transfer_factor()*values["Мёрфи 1"]
                /(tg(x) * (1-bypass_fraction(loc='верха')))))
            
            values['значение кинетической кривой'] = get_kinetic_y(x, values['эффективность по Мёрфи'], loc='верха')
                
        return values
    
    def get_plate_coeffs(aqua_liquid_saturation, diameter, plate, properties, Substance, PRESSURE):
        bubble_layer = pd.Series(dtype=float)

        def get_water_interfacial_tension(temperature):
        
            aq = aqua_liquid_saturation.loc[aqua_liquid_saturation[aqua_liquid_saturation['temperature'] >= temperature].index.min()]
            return aq['interfacial_tension_kilo']

        bubble_layer['высота светлого слоя жидкости верха'] = (
            0.787 * (diameter['массовая нагрузка жидкости верха']
                    /(properties['плотность жидкости']['верха']
                    *plate['ширина переливного порога']))**0.2
            *(plate['высота переливного порога']**0.56)
            *(plate['скорость пара в рабочем сечении тарелки']**(0.05-4.6*plate['высота переливного порога']))
            *(1-0.31*np.exp(-0.11*properties['вязкость жидкости']['верха']))
            *(properties['поверхностное натяжение жидкости']['верха']
                /get_water_interfacial_tension(properties['температура']['верха']))**0.09)
        
        bubble_layer['высота светлого слоя жидкости низа'] = (
            0.787 * (diameter['массовая нагрузка жидкости низа']
                    /(properties['плотность жидкости']['низа']
                    *plate['ширина переливного порога']))**0.2
            *(plate['высота переливного порога']**0.56)
            *(plate['скорость пара в рабочем сечении тарелки']**(0.05-4.6*plate['высота переливного порога']))
            *(1-0.31*np.exp(-0.11*properties['вязкость жидкости']['низа']))
            *(properties['поверхностное натяжение жидкости']['низа']
                /get_water_interfacial_tension(properties['температура']['низа']))**0.09)
        
        bubble_layer['коэффициент Фруда верха'] = (
            plate['скорость пара в рабочем сечении тарелки']
            /(9.8*bubble_layer['высота светлого слоя жидкости верха']))
        
        bubble_layer['коэффициент Фруда низа'] = (
            plate['скорость пара в рабочем сечении тарелки']
            /(9.8*bubble_layer['высота светлого слоя жидкости низа']))
        
        bubble_layer['паросодержание барботажного слоя верха'] = (
            np.sqrt(bubble_layer['коэффициент Фруда верха'])
            /(1+np.sqrt(bubble_layer['коэффициент Фруда верха'])))
        
        bubble_layer['паросодержание барботажного слоя низа'] = (
            np.sqrt(bubble_layer['коэффициент Фруда низа'])
            /(1+np.sqrt(bubble_layer['коэффициент Фруда низа'])))
        
        #находим коэффициенты диффузии как для насадочной колонны
        variables = pd.Series(dtype=float)
            
        u_a = Calculations.get_value(component= Substance['A'], attribute='vicosity_organic_liquid', temperature=20)
        u_b = Calculations.get_value(component= Substance['B'], attribute='vicosity_organic_liquid', temperature=20)
        
        variables['вязкость жидкости верха при 20°С'] = (
            u_a*properties['содержание легколетучего в жидкости']['верха']
            +u_b*(1-properties['содержание легколетучего в жидкости']['верха']))
        
        variables['вязкость жидкости низа при 20°С'] = (
            u_a*properties['содержание легколетучего в жидкости']['низа']
            +u_b*(1-properties['содержание легколетучего в жидкости']['низа']))
        
        p_a = Calculations.get_value(component= Substance['A'], attribute='density_organic_liquid', temperature=20)
        p_b = Calculations.get_value(component= Substance['B'], attribute='density_organic_liquid', temperature=20)
        
        variables['плотность жидкости верха при 20°С'] = (
            p_a*properties['содержание легколетучего в жидкости']['верха']
            +p_b*(1-properties['содержание легколетучего в жидкости']['верха']))
        
        variables['плотность жидкости низа при 20°С'] = (
            p_a*properties['содержание легколетучего в жидкости']['низа']
            +p_b*(1-properties['содержание легколетучего в жидкости']['низа']))
        
        bubble_layer['коэффициент диффузии жидкости верха при 20°С'] = (
            np.double([10**(-6)])
            /(np.sqrt(variables['вязкость жидкости верха при 20°С'])
            *(properties['молярный объем жидкости']['дистиллята']**(1/3)
                + properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
            
        bubble_layer['коэффициент диффузии жидкости низа при 20°С'] = (
            np.double([10**(-6)])
            /(np.sqrt(variables['вязкость жидкости низа при 20°С'])
            *(properties['молярный объем жидкости']['дистиллята']**(1/3)
                +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
        
        bubble_layer['температурный коэффициент верха'] = (
            0.2*np.sqrt(variables['вязкость жидкости верха при 20°С'])
            /variables['плотность жидкости верха при 20°С']**(1/3))
        
        bubble_layer['температурный коэффициент низа'] = (
            0.2*np.sqrt(variables['вязкость жидкости низа при 20°С'])
            /variables['плотность жидкости низа при 20°С']**(1/3))
        
        bubble_layer['коэффициент диффузии жидкости низа'] = (
            bubble_layer['коэффициент диффузии жидкости низа при 20°С']
            *(1 + bubble_layer['температурный коэффициент низа']
            *(properties['температура']['низа'] - 20)))

        bubble_layer['коэффициент диффузии жидкости верха'] = (
            bubble_layer['коэффициент диффузии жидкости верха при 20°С']
            *(1+bubble_layer['температурный коэффициент верха']
            *(properties['температура']['верха'] - 20)))
        
            
        bubble_layer['коэффициент диффузии пара верха'] = (
            np.double(4.22*10**(-2))
            *(np.double(273)+properties['температура']['верха'])**(3/2)
            /(PRESSURE*(properties['молярный объем жидкости']['дистиллята']**(1/3)
                        +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
        
        bubble_layer['коэффициент диффузии пара низа'] = (
            np.double(4.22*10**(-2))
            *(np.double(273) + properties['температура']['низа'])**(3/2)
            /(PRESSURE * (properties['молярный объем жидкости']['дистиллята']**(1/3)
                        +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
        
        #находим коэффициенты массоотдачи
        bubble_layer['коэффициент массоотдачи жидкости верха'] = (
            6.24*10**5 * bubble_layer['коэффициент диффузии жидкости верха']**0.5
            *(diameter['массовая нагрузка жидкости верха']
            /(properties['плотность жидкости']['верха']*plate['рабочее сечение тарелки']
                *(1-bubble_layer['паросодержание барботажного слоя верха'])))**0.5
            *bubble_layer['высота светлого слоя жидкости верха']
            *(properties['вязкость пара']['верха']
            /(properties['вязкость жидкости']['верха'] + properties['вязкость пара']['верха']))**0.5)
        
        bubble_layer['коэффициент массоотдачи жидкости низа'] = (
            6.24*10**5 * bubble_layer['коэффициент диффузии жидкости низа']**0.5
            *(diameter['массовая нагрузка жидкости низа']
            /(properties['плотность жидкости']['низа']*plate['рабочее сечение тарелки']
                *(1-bubble_layer['паросодержание барботажного слоя низа'])))**0.5
            *bubble_layer['высота светлого слоя жидкости низа']
            *(properties['вязкость пара']['низа']
            /(properties['вязкость жидкости']['низа'] + properties['вязкость пара']['низа']))**0.5)
            
        bubble_layer['коэффициент массоотдачи пара верха'] = (
            6.24*10**5 * bubble_layer['коэффициент диффузии пара верха']**0.5
            *plate['относительное свободное сечение тарелки']/100
            *(plate['скорость пара в рабочем сечении тарелки']/bubble_layer['паросодержание барботажного слоя верха'])**0.5
            *bubble_layer['высота светлого слоя жидкости верха']
            *(properties['вязкость пара']['верха']
            /(properties['вязкость жидкости']['верха'] + properties['вязкость пара']['верха']))**0.5)
        
        bubble_layer['коэффициент массоотдачи пара низа'] = (
            6.24*10**5 * bubble_layer['коэффициент диффузии пара низа']**0.5
            *plate['относительное свободное сечение тарелки']/100
            *(plate['скорость пара в рабочем сечении тарелки']/bubble_layer['паросодержание барботажного слоя низа'])**0.5
            *bubble_layer['высота светлого слоя жидкости низа']
            *(properties['вязкость пара']['низа']
            /(properties['вязкость жидкости']['низа'] + properties['вязкость пара']['низа']))**0.5)
        
        bubble_layer['коэффициент массоотдачи жидкости верха на кмоль'] = (
            bubble_layer['коэффициент массоотдачи жидкости верха']
            *properties['плотность жидкости']['верха']
            /properties['молярная масса жидкости']['верха'])
        
        bubble_layer['коэффициент массоотдачи жидкости низа на кмоль'] = (
            bubble_layer['коэффициент массоотдачи жидкости низа']
            *properties['плотность жидкости']['низа']
            /properties['молярная масса жидкости']['низа'])
        
        bubble_layer['коэффициент массоотдачи пара верха на кмоль'] = (
            bubble_layer['коэффициент массоотдачи пара верха']
            *properties['плотность пара']['верха']
            /properties['молярная масса газа']['верха'])
        
        bubble_layer['коэффициент массоотдачи пара низа на кмоль'] = (
            bubble_layer['коэффициент массоотдачи пара низа']
            *properties['плотность пара']['низа']
            /properties['молярная масса газа']['низа'])
        
        return bubble_layer
    
    def get_plate(diameter, plate_type = 'ТС—Р'):
    
        table_plate = pd.read_excel('tables/Техническая характеристика ситчатых тарелок типа ТС.xlsx')
        hole_list=np.array([3,4,5,8])
        
        type_mask = table_plate['Тип тарелки'] == plate_type
        d_mask = table_plate['Диаметр колонны D, мм'] == diameter['стандартный размер обечайки']*1000
        plate = table_plate[type_mask & d_mask]
        
        free_section_list = np.array([float(str(*plate[hole].values)[0:str(*plate[hole].values).find('—')]) for hole in hole_list])
        free_section = free_section_list[free_section_list == free_section_list.max()][-1]
        
        vapor_speed = np.array([diameter['скорость пара верха'], diameter['скорость пара низа']]).max()
        vapor_section_speed = vapor_speed*0.785*diameter['стандартный размер обечайки']**2/plate['Рабочее сечение тарелки, м2'].values
        
        return pd.Series({'скорость пара в рабочем сечении тарелки':vapor_section_speed,
                        'рабочее сечение тарелки':plate['Рабочее сечение тарелки, м2'].values,
                        'тип тарелки':plate_type,
                        'относительное свободное сечение тарелки':free_section,
                        'высота переливного порога':0.03,
                        'ширина переливного порога':plate['Периметр слива Lc, м'].values})
    
    def calculate_plate_diameter(balance, Ropt, properties, plate_type = 'ситчатая'):
        standart_list = np.array([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.6, 3.0])
        diameter = pd.Series(dtype = float)
        
        diameter['массовая нагрузка жидкости верха'] = (
            balance['массовый расход в дефлегматоре']
            *Ropt*properties['молярная масса жидкости']['верха']
            /properties['молярная масса жидкости']['дистиллята'])

        diameter['массовая нагрузка жидкости низа'] = (
            (balance['массовый расход в дефлегматоре']
             *Ropt*properties['молярная масса жидкости']['низа']
             /properties['молярная масса жидкости']['дистиллята'])
            +balance['массовый расход в питателе']
            *properties['молярная масса жидкости']['низа']
            /properties['молярная масса жидкости']['питания'])
        
        diameter['массовый поток пара верха'] = (
            balance['массовый расход в дефлегматоре']
            *(Ropt+1)*properties['молярная масса газа']['верха']
            /properties['молярная масса газа']['дистиллята'])
                
        diameter['массовый поток пара низа'] = (
            balance['массовый расход в дефлегматоре']
            *(Ropt+1)*properties['молярная масса газа']['низа']
            /properties['молярная масса газа']['дистиллята'])
        
        if plate_type == 'ситчатая':
            
            diameter['скорость пара верха'] = 0.05*np.sqrt(
                properties['плотность жидкости']['верха']
                /properties['плотность пара']['верха'])
            
            diameter['скорость пара низа'] = 0.05*np.sqrt(
                properties['плотность жидкости']['низа']
                /properties['плотность пара']['низа'])
        
        diameter['диаметр верха'] = np.sqrt(
            4*diameter['массовый поток пара верха']
            /(np.pi*diameter['скорость пара верха']
              *properties['плотность пара']['верха']))
            
        diameter['диаметр низа'] = np.sqrt(
            4*diameter['массовый поток пара низа']
            /(np.pi*diameter['скорость пара низа']
              *properties['плотность пара']['низа']))
        
        diameter['стандартный размер обечайки'] = np.array(
            [standart_list[standart_list > diameter['диаметр низа']].min(),
             standart_list[standart_list > diameter['диаметр верха']].min()]).max()
        
        diameter['действительная рабочая скорость верха'] = (
            diameter['скорость пара верха']
            *(diameter['диаметр верха']
              /diameter['стандартный размер обечайки'])**2)
            
        diameter['действительная рабочая скорость низа'] = (
            diameter['скорость пара низа']
            *(diameter['диаметр низа']
              /diameter['стандартный размер обечайки'])**2)

        return diameter
    
    
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
        attr = attr.dropna(axis=1)
        coeff = dfc.get_coeffs(list(attr.columns),list(*attr.values))
        value = np.poly1d(coeff)(temperature)
        
        return value
    
    
    def get_range_phlegm_number(yf, xw, xf, xp, Rmin, xy_diagram, diagram, Bt_range: int, plot_type = 'matplotlib'):
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
            plot_lines (str, optional): Нужно ли строить графики или просто делать рсчет. Defaults to 'matplotlib'.

        Returns:
            R, Ngraf: if plot_type != 'plotly' Массив с набором рабочих флегмовых чисел, массив с набором чисел ступеней разделения
            
            fig, R, Ngraf: if plot_type == 'plotly' график и массивы            
            """
        if Bt_range > 50:
            Bt_range = 50
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
        
        if plot_type == 'matplotlib':
            fig = plt.figure(figsize=(15,45))
            plt.style.use(['science', 'no-latex', 'notebook', 'grid'])
            SMALL_SIZE = 6
            MEDIUM_SIZE = 10
            BIGGER_SIZE = 12
            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
            plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            
        elif plot_type == 'plotly':
            fig = make_subplots(rows=int(np.ceil(Bt_range/5)), cols=5)#в одном ряду будет 5 графиков
        else:
            pass
            
        
        def biuld_phlegm_lines(yf):
            _y = float(xw), float(yf)
            bottom_work_line = np.polyfit(_x,_y, 1)
            y_ = float(yf), float(xp)
            top_work_line = np.polyfit(x_,y_, 1)
            return bottom_work_line, top_work_line

        i = 1
        for yf in yf_:
            if len(step) == 0:
                step.append(np.poly1d(xy_diagram)(xw))
                
            while step[-1] <= yf:
                bottom_work_line, top_work_line = biuld_phlegm_lines(yf)
                bottom_work_line[1] = bottom_work_line[1] - step[-1]
                platform.append(np.roots(bottom_work_line))
                step.append(np.poly1d(xy_diagram)(platform[-1]))
                
            while step[-1] <= xp:
                bottom_work_line, top_work_line = biuld_phlegm_lines(yf)
                top_work_line[1] = top_work_line[1] - step[-1]
                platform.append(np.roots(top_work_line))
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
                
                if plot_type == 'matplotlib' or plot_type == 'plotly':
                    fig = Figures.plot_range_phlegm_number(fig, i, diagram, stair_line_x, stair_line_y, x_y, _x, x_,
                                                           W_line, P_line, N, yf, R, plot_type)
                else:
                    pass
                step = []
                platform = []
                i += 1
        Ngraf = N.values *(R+1)
        
        if plot_type == 'plotly':
            fig.update_layout(
                    autosize=False,
                    width=1000,
                    height=300*int(np.ceil(Bt_range/5)),
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
            return fig, R, Ngraf
        return R,Ngraf
    
    def get_optimal_phlegm_number(R,Ngraf, plot_type = "matplotlib"):
        """Оптимальное флегмовое число "Ropt" находят по минимуму функции R от N(R+1).
        
        Args:
            R: Массив с набором рабочих флегмовых чисел
            Ngraf: Массив с набором чисел ступеней разделения
            plot_lines (str, optional):нужно ли строить графики или просто проводить расчет. Defaults to "True".

        Returns:
            Ropt: Оптимальное флегмовое число
            fig: Если выбран 'plot_type' plotly
        """
        Nfit = dfc.get_fit(R,Ngraf)
        optimal = []
        for value in Nfit:
            optimal.append(value == Nfit.min())
        Ropt = np.round(R[optimal],2)
        
        if plot_type == 'matplotlib' or plot_type == 'plotly':
            fig = Figures.plot_optimal_phlegm_number(R, Ngraf, Nfit, optimal, Ropt, plot_type)
            
        if plot_type == 'plotly':
            return fig, Ropt
        
        return Ropt
    
    def calculate_properties_slice(liquid_fraction, vapor_fraction, temperature, Substance, Ma, Mb, slice_type = 'DataFrame'):
        
        #Теплопроводность компонента А [Вт/(м*K)]
        thermal_conductivity_a = Calculations.get_value(
            component= Substance['A'],
            attribute='thermal_conductivity_organic_liquid', 
            temperature=temperature)
        
        #Теплопроводность компонента Б  [Вт/(м*K)]
        thermal_conductivity_b = Calculations.get_value(
            component= Substance['B'],
            attribute='thermal_conductivity_organic_liquid', 
            temperature=temperature)
        
        #Коэффициенты объемного теплового расширения компонента А b*10^3, K^-1
        thermal_expansion_a = Calculations.get_value(
            component= Substance['A'],
            attribute='thermal_expansion_organic_liquid', 
            temperature=temperature)
        
        #Коэффициенты объемного теплового расширения компонента Б b*10^3, K^-1
        thermal_expansion_b = Calculations.get_value(
            component= Substance['B'],
            attribute='thermal_expansion_organic_liquid', 
            temperature=temperature)
        
        #Давление насыщенного пара [мм.рт.ст.] компонента А
        vapor_pressure_a = Calculations.get_value(
            component= Substance['A'],
            attribute='vapor_pressure_organic_liquid', 
            temperature=temperature)
        
        #Давление насыщенного пара [мм.рт.ст.] компонента Б
        vapor_pressure_b = Calculations.get_value(
            component= Substance['B'],
            attribute='vapor_pressure_organic_liquid', 
            temperature=temperature)
        
        #Поверхностное натяжение [мДж/м^2] компонента А
        sigma_a = Calculations.get_value(
            component= Substance['A'],
            attribute='interfactial_tension_organic_liquid', 
            temperature=temperature)
        
        #Поверхностное натяжение [мДж/м^2] компонента Б
        sigma_b = Calculations.get_value(
            component= Substance['B'],
            attribute='interfactial_tension_organic_liquid', 
            temperature=temperature)
         
        #Удельная теплоемкость [Дж/(кг*K)] компонента А
        Cp_a = Calculations.get_value(
            component= Substance['A'],
            attribute='heat_capacity_organic_liquid', 
            temperature=temperature)
        
        #Удельная теплоемкость [Дж/(кг*K)] компонента Б
        Cp_b = Calculations.get_value(
            component= Substance['B'],
            attribute='heat_capacity_organic_liquid', 
            temperature=temperature)
        
        #Теплота парообразования компонента А [кДж/кг]
        Qv_a = Calculations.get_value(
            component= Substance['A'],
            attribute='heat_vaporization_organic_liquid', 
            temperature=temperature)
        
        #Теплота парообразования компонента Б [кДж/кг]
        Qv_b = Calculations.get_value(
            component= Substance['B'],
            attribute='heat_vaporization_organic_liquid', 
            temperature=temperature)
        
        #Плотность [кг/м^3] компонента А
        p_a = Calculations.get_value(
            component= Substance['A'],
            attribute='density_organic_liquid', 
            temperature=temperature)
        
        #Плотность [кг/м^3] компонента Б
        p_b = Calculations.get_value(
            component= Substance['B'],
            attribute='density_organic_liquid', 
            temperature=temperature)
         
        #Динамическая вязкость [мПа*с] компонента А
        u_a = Calculations.get_value(
            component= Substance['A'],
            attribute='vicosity_organic_liquid', 
            temperature=temperature) 
         
        #Динамическая вязкость [мПа*с] компонента Б 
        u_b = Calculations.get_value(
            component= Substance['B'],
            attribute='vicosity_organic_liquid', 
            temperature=temperature)
        
        #Вязкость паров [мкПа*с] компонента А 
        ug_a = Calculations.get_value(
            component= Substance['A'],
            attribute='vicosity_organic_vapor', 
            temperature=temperature)
        
        #Вязкость паров [мкПа*с] компонента Б
        ug_b = Calculations.get_value(
            component= Substance['B'],
            attribute='vicosity_organic_vapor', 
            temperature=temperature)

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
    
    def get_transfer_numbers(balance, Ropt, xy_diagram, plot_type = 'matplotlib'):
        """Функция возвращает число единиц переноса для модифицированного уравнения массопередачи. Подробнее в [1] стр 232

        Args:
            balance (pd.Series): результат функции material_balance
            Ropt (float): Оптимальное флегмовое число
            xy_diagram (np.array): аппроксимация диаграммы жидкость-пар полиномом
            plot_type (str, optional): если plotly, то функция вернет fig, bottom, top. Defaults to 'matplotlib'.

        Returns:
            bottom, top: if plot_type != 'plotly'. Число единиц переноса вверху колонны и внизу
            fig, bottom, top: if plot_type == 'plotly'. График и число единиц переноса
        """
    
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

        _ = [0, 1]
        if plot_type == 'matplotlib':
            #Строим графики
            SMALL_SIZE = 6
            MEDIUM_SIZE = 10
            BIGGER_SIZE = 12
            plt.style.use(['science', 'no-latex', 'notebook', 'grid'])
            plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
            plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
            
            axes[0].plot(wi_x, fw(w_x), label = (r'$\int {низ}$= ' + f'{bottom}'))
            axes[0].fill_between(wi_x, fw(w_x), where=[(w_x >= float(balance['xw'])) and (w_x <= float(balance['xf'])) for w_x in w_x],
                            color = 'blue', alpha = 0.4)
            axes[0].plot(pi_x, fp(p_x), label = (r'$\int {верх}$= ' + f'{top}'))
            axes[0].fill_between(pi_x, fp(p_x), where=[(p_x >= float(balance['xf'])) and (p_x <= float(balance['xp'])) for p_x in p_x],
                            color = 'green', alpha = 0.4)
            axes[0].set_ylabel(r'$ \frac {1}{y* - y}$',  fontsize=10)
            axes[0].set_xlabel(f'Мольная доля ллт в паре', fontsize=10)
            axes[0].legend(loc='upper center')


            axes[1].plot(xy, fxy(xy), color='black', lw=1, ms=2)
            axes[1].plot(_, _, color='black', lw=0.5)
            axes[1].plot(_x, W_line, color='black', lw=1, ms=2)
            axes[1].plot(x_, P_line, color='black', lw=1, ms=2)
            axes[1].fill_between(
                w_x, fxy(w_x), fxyw(w_x),  
                where=[(w_x >= float(balance['xw'])) and (w_x <= float(balance['xf'])) for w_x in w_x],
                color = 'blue', alpha = 0.4)
            axes[1].fill_between(
                p_x, fxy(p_x), fxyp(p_x), 
                where=[(p_x >= float(balance['xf'])) and (p_x <= float(balance['xp'])) for p_x in p_x],
                color = 'green', alpha = 0.4)
            axes[1].set_ylabel(f'Мольная доля ллт в паре', fontsize=10)
            axes[1].set_xlabel(f'Мольная доля ллт в жидкости', fontsize=10)
            
        if plot_type == 'plotly':
            
            fig = make_subplots(rows=1, cols=2)

            fig.add_trace(go.Scatter(x=wi_x, y=fw(w_x),
                                        line=dict(
                                            color='rgb(0, 77, 153)',
                                            width=3),
                                        mode='lines',
                                        name='',
                                        fill='tozeroy'
                                        ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=pi_x, y=fp(p_x),
                                        line=dict(
                                            color='green',
                                            width=3),
                                        mode='lines',
                                        name='',
                                        fill='tozeroy'
                                        ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=xy, y=fxy(xy),
                                        line=dict(
                                            color='rgb(0, 77, 153)',
                                            width=3),
                                        mode='lines',
                                        name='',
                                        fill=None
                                        ), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=_x, y=W_line,
                                        line=dict(
                                            color='rgb(0, 77, 153)',
                                            width=2),
                                        mode='lines',
                                        name='исчерпывающая рабочая линия',
                                        fill=None
                                        ), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=x_, y=P_line,
                                        line=dict(
                                            color='green',
                                            width=2),
                                        mode='lines',
                                        name='укрепляющая рабочая линия',
                                        fill=None
                                        ), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=_, y=_,
                                        line=dict(
                                            color='grey',
                                            width=1),
                                        mode='lines+markers',
                                        name='линия нулевого разделения'
                                        ), row=1, col=2)
            
            fig.update_xaxes(range=[-0.01, 1.01],
                                    row=1, col=1,
                                    showline=True, linewidth=2, linecolor='black',
                                    mirror=True,
                                    ticks='inside',
                                    gridcolor='rgb(105,105,105)',
                                    griddash='1px',
                                    zeroline=False,
                                    title_text='мольная доля ллт в паре')
            
            fig.update_xaxes(range=[-0.01, 1.01],
                                    row=1, col=2,
                                    showline=True, linewidth=2, linecolor='black',
                                    mirror=True,
                                    ticks='inside',
                                    gridcolor='rgb(105,105,105)',
                                    griddash='1px',
                                    zeroline=False,
                                    title_text='мольная доля ллт в жидкости')
            
            fig.update_yaxes( 
                        row=1, col=1,
                        showline=True, linewidth=2, linecolor='black',
                        mirror=True,
                        ticks='inside',
                        gridcolor='rgb(105,105,105)',
                        griddash='1px',
                        zeroline=False,
                        title_text=r'$ \frac {1}{y* - y}$')
            
            fig.update_yaxes( 
                        row=1, col=2,
                        showline=True, linewidth=2, linecolor='black',
                        mirror=True,
                        ticks='inside',
                        gridcolor='rgb(105,105,105)',
                        griddash='1px',
                        zeroline=False,
                        title_text='мольная доля ллт в паре')
            
            fig.update_layout(
                        autosize=False,
                        width=1000,
                        height=500,
                        margin=dict(l=20, r=5, t=20, b=2),
                        showlegend=False,
                        plot_bgcolor='white')
            
            return fig, bottom, top
        
        return bottom, top
    
    def calculate_diameter(balance, Ropt, properties, filling_name: str):
        diameter = pd.Series(dtype=float)
        standart_list = np.array([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.6, 3.0])
        
        filling = pd.DataFrame(
            columns = ['удельная поверхность','свободный объем','насыпная плотность'],
            dtype=float)
        filling.loc['25x25x3'] = [200, 0.74, 530]
        filling.loc['35x35x4'] = [140, 0.78, 530]
        filling.loc['50x50x5'] = [87.5, 0.785, 530]
        filling = filling.loc[filling_name]
        
        diameter['массовая нагрузка жидкости верха'] = (
            balance['массовый расход в дефлегматоре']*Ropt 
            *properties['молярная масса жидкости']['верха'] 
            /properties['молярная масса жидкости']['дистиллята'])
        
        diameter['массовая нагрузка жидкости низа'] = (
            (balance['массовый расход в дефлегматоре']*Ropt
             *properties['молярная масса жидкости']['низа'] 
             /properties['молярная масса жидкости']['дистиллята'])
            +balance['массовый расход в питателе']
            *properties['молярная масса жидкости']['низа']
            /properties['молярная масса жидкости']['питания'])
        
        diameter['массовый поток пара верха'] = (
            balance['массовый расход в дефлегматоре']*(Ropt+1)
            *properties['молярная масса газа']['верха']
            /properties['молярная масса газа']['дистиллята'])
        
        diameter['массовый поток пара низа'] = (
            balance['массовый расход в дефлегматоре']*(Ropt+1)
            *properties['молярная масса газа']['низа']
            /properties['молярная масса газа']['дистиллята'])
        
        diameter['предельная скорость пара верха'] = np.sqrt(
            1.2*np.exp(-4*(diameter['массовая нагрузка жидкости верха']
                           /diameter['массовый поток пара верха'])**0.25
                       *(properties['плотность пара']['верха']
                         /properties['плотность жидкости']['верха'])**0.125)
            *(9.8*filling['свободный объем']**3
              *properties['плотность жидкости']['верха'])
            /(filling['удельная поверхность'] * properties['плотность пара']['верха']
              *properties['вязкость жидкости']['верха']**0.16))
        
        diameter['предельная скорость пара низа'] = np.sqrt(
            1.2*np.exp(-4*(diameter['массовая нагрузка жидкости низа']
                           /diameter['массовый поток пара низа'])**0.25
                    *(properties['плотность пара']['низа']
                      /properties['плотность жидкости']['низа'])**0.125)
            *(9.8*filling['свободный объем']**3
              *properties['плотность жидкости']['низа'])
            /(filling['удельная поверхность']*properties['плотность пара']['низа']
            *properties['вязкость жидкости']['низа']**0.16))
        
        diameter['рабочая скорость пара верха'] = diameter['предельная скорость пара верха']*0.7
        
        diameter['рабочая скорость пара низа'] = diameter['предельная скорость пара низа']*0.7
        
        diameter['диаметр верха'] = np.sqrt(
            4*diameter['массовый поток пара верха']
            /(np.pi * diameter['рабочая скорость пара верха']
              *properties['плотность пара']['верха']))
        
        diameter['диаметр низа'] = np.sqrt(
            4*diameter['массовый поток пара низа']
            /(np.pi*diameter['рабочая скорость пара низа']*properties['плотность пара']['низа']))
        
        diameter['стандартный размер обечайки'] = np.array(
            [standart_list[standart_list > diameter['диаметр низа']].min(),
             standart_list[standart_list > diameter['диаметр верха']].min()]).max()
        
        diameter['действительная рабочая скорость верха'] = (
            diameter['рабочая скорость пара верха']
            *(diameter['диаметр верха']
              /diameter['стандартный размер обечайки'])**2)
        
        diameter['действительная рабочая скорость низа'] = (
            diameter['рабочая скорость пара низа']
            *(diameter['диаметр низа']
              /diameter['стандартный размер обечайки'])**2)
        
        diameter['% от предельной скорости верха'] = (
            diameter['действительная рабочая скорость верха']
            /diameter['предельная скорость пара верха']*100)
        
        diameter['% от предельной скорости низа'] = (
            diameter['действительная рабочая скорость низа']
            /diameter['предельная скорость пара низа']*100)
        
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
        variables['вязкость жидкости верха при 20°С'] = (
            u_a*properties['содержание легколетучего в жидкости']['верха']
            +u_b*(1-properties['содержание легколетучего в жидкости']['верха']))
        
        variables['вязкость жидкости низа при 20°С'] = (
            u_a*properties['содержание легколетучего в жидкости']['низа']
            +u_b*(1-properties['содержание легколетучего в жидкости']['низа']))
        
        p_a = Calculations.get_value(component= Substance['A'], attribute='density_organic_liquid', temperature=20)
        p_b = Calculations.get_value(component= Substance['B'], attribute='density_organic_liquid', temperature=20)        
        variables['плотность жидкости верха при 20°С'] = (
            p_a*properties['содержание легколетучего в жидкости']['верха']
            +p_b*(1-properties['содержание легколетучего в жидкости']['верха']))
        
        variables['плотность жидкости низа при 20°С'] = (
            p_a*properties['содержание легколетучего в жидкости']['низа']
            +p_b*(1-properties['содержание легколетучего в жидкости']['низа']))
        
        def get_m(x): return np.poly1d(xy_diagram)(x)/x
        x_bottom = np.linspace(balance['xw'],balance['xf'],100)
        x_top = np.linspace(balance['xf'],balance['xp'],100)
        
        hight = pd.Series(dtype=float)
        hight['отношение нагрузок пар/жидкость верха'] = (Ropt+1)/Ropt

        hight['отношение нагрузок пар/жидкость низа'] = (
            (Ropt+1)
            /(Ropt +(balance['массовый расход в питателе']
                     *properties['молярная масса жидкости']['дистиллята']
                     /(balance['массовый расход в дефлегматоре']
                       *properties['молярная масса жидкости']['питания']))))

        hight['коэффициент диффузии жидкости верха при 20°С'] = (
            np.double([10**(-6)])
            /(np.sqrt(variables['вязкость жидкости верха при 20°С'])
            *(properties['молярный объем жидкости']['дистиллята']**(1/3)
              +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
        
        hight['коэффициент диффузии жидкости низа при 20°С'] = (
            np.double([10**(-6)])
            /(np.sqrt(variables['вязкость жидкости низа при 20°С'])
            *(properties['молярный объем жидкости']['дистиллята']**(1/3)
              +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
        
        hight['температурный коэффициент верха'] = (
            0.2*np.sqrt(variables['вязкость жидкости верха при 20°С'])
            /variables['плотность жидкости верха при 20°С']**(1/3))
        
        hight['температурный коэффициент низа'] = (
            0.2*np.sqrt(variables['вязкость жидкости низа при 20°С'])
            /variables['плотность жидкости низа при 20°С']**(1/3))
        
        hight['коэффициент диффузии жидкости низа'] = (
            hight['коэффициент диффузии жидкости низа при 20°С']
            *(1 + hight['температурный коэффициент низа']
              *(properties['температура']['низа'] - 20)))
    
        hight['коэффициент диффузии жидкости верха'] = (
            hight['коэффициент диффузии жидкости верха при 20°С']
            *(1 + hight['температурный коэффициент верха']
              *(properties['температура']['верха'] - 20)))
        
            
        hight['коэффициент диффузии пара верха'] = (
            np.double(4.22*10**(-2))
            *(np.double(273)
              +properties['температура']['верха'])**(3/2)
            /(PRESSURE*(properties['молярный объем жидкости']['дистиллята']**(1/3)
                        +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба']))
        
        hight['коэффициент диффузии пара низа'] = (
            (np.double(4.22*10**(-2))
             *(np.double(273)+properties['температура']['низа'])**(3/2)
            /(PRESSURE*(properties['молярный объем жидкости']['дистиллята']**(1/3)
                        +properties['молярный объем жидкости']['куба']**(1/3))**2)
            *np.sqrt(1/properties['молярная масса жидкости']['дистиллята']
                     +1/properties['молярная масса жидкости']['куба'])))
        
        hight['средний коэффициент распределения верха'] = get_m(x_top).mean()
        hight['средний коэффициент распределения низа'] = get_m(x_bottom).mean()
        
        hight['критерий Прандтля жидости верха'] = (
            properties['вязкость жидкости']['верха']*10**-3
            /(properties['плотность жидкости']['верха']
              *hight['коэффициент диффузии жидкости верха']))
        
        hight['критерий Прандтля жидкости низа'] = (
            properties['вязкость жидкости']['низа']*10**-3
            /(properties['плотность жидкости']['низа']
              *hight['коэффициент диффузии жидкости низа']))
        
        hight['высота единицы переноса жидкости верха'] = (
            0.258*phi_top*c_top
            *hight['критерий Прандтля жидости верха']**(1/2)
            *3**0.15)
        
        hight['высота единицы переноса жидкости низа'] = (
            0.258*phi_bottom*c_bottom
            *hight['критерий Прандтля жидкости низа']**(1/2)
            *3**0.15)
        
        if diameter['стандартный размер обечайки'] > 0.8:
            d = 1.24
        else:
            d = 1
            
        hight['критерий Прандтля пара верха'] = (
            properties['вязкость пара']['верха']*10**-3
            /(properties['плотность пара']['верха']
              *hight['коэффициент диффузии пара верха']))
        
        hight['критерий Прандтля пара низа'] = (
            properties['вязкость пара']['низа']*10**-3
            /(properties['плотность пара']['низа']
              *hight['коэффициент диффузии пара низа']))
        
        hight['массовая плотность орошения верха'] = (
            diameter['массовая нагрузка жидкости верха']
            /(0.785 * diameter['стандартный размер обечайки']**2))
            
        hight['массовая плотность орошения низа'] = (
            diameter['массовая нагрузка жидкости низа']
            /(0.785 * diameter['стандартный размер обечайки']**2))
        
        hight['высота единицы переноса пара верха'] = (
            (0.0175*psi_top*hight['критерий Прандтля пара верха']
             *diameter['стандартный размер обечайки']**d * 3**0.33)
            /((hight['массовая плотность орошения верха']
               *properties['вязкость жидкости']['верха']**0.16
               *(1000/properties['плотность жидкости']['верха'])**1.25 
               *(((72.8*10**-3)
                  /(properties['поверхностное натяжение жидкости']['верха']/1000))**0.8)))**0.6)
                                                    
        hight['высота единицы переноса пара низа'] = (
            (0.0175*psi_bottom*hight['критерий Прандтля пара низа']
             *diameter['стандартный размер обечайки']**d * 3**0.33)
            /((hight['массовая плотность орошения низа'] * properties['вязкость жидкости']['низа']**0.16
               *(1000/properties['плотность жидкости']['низа'])**1.25
               *(((72.8*10**-3)
                  /(properties['поверхностное натяжение жидкости']['низа']/1000))**0.8)))**0.6)
        
        hight['общая высота единицы переноса верха'] = (
            hight['высота единицы переноса пара верха']
            +hight['средний коэффициент распределения верха']
            *hight['отношение нагрузок пар/жидкость верха']
            *hight['высота единицы переноса жидкости верха'])
        
        hight['общая высота единицы переноса низа'] = (
            hight['высота единицы переноса пара низа']
            +hight['средний коэффициент распределения низа']
            *hight['отношение нагрузок пар/жидкость низа']
            *hight['высота единицы переноса жидкости низа'])
        
        hight['высота насадки верха'] = hight['общая высота единицы переноса верха'] * top
        hight['высота насадки низа'] = hight['общая высота единицы переноса низа'] * bottom
        
        hight['общая высота насадки в колонне'] = (
            hight['высота насадки верха']
            +hight['высота насадки низа'])
        
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
        
        thermal_balance['теплота забираемая водой в дефлегматоре'] = (
            balance['массовый расход в дефлегматоре']
            *(1+Ropt)*properties['теплота парообразования жидкости']['дистиллята'])

        thermal_balance['теплота передаваемая паром от испарителя'] = (
            balance['массовый расход в питателе']
            *properties['теплота парообразования жидкости']['питания'])


        thermal_balance['теплота исходной смеси'] = (
            balance['массовый расход в питателе']
            *properties['температура']['питания']
            *properties['удельная теплоемкость жидкости']['питания']/1000)

        thermal_balance['теплота кубовой жидкости'] = (
            balance['массовый расход в кубовом остатке']
            *properties['температура']['куба']
            *properties['удельная теплоемкость жидкости']['куба']/1000)

        thermal_balance['теплота дистиллята'] = (
            balance['массовый расход в дефлегматоре']
            *properties['температура']['дистиллята']
            *properties['удельная теплоемкость жидкости']['дистиллята']/1000)

        thermal_balance['теплота получаемая кипящей жидкостью'] = (
            thermal_balance['теплота забираемая водой в дефлегматоре']
            +thermal_balance['теплота дистиллята']
            +thermal_balance['теплота кубовой жидкости']
            -thermal_balance['теплота исходной смеси'])
        
        return thermal_balance

class Figures():
    
    def plot_xy_diagram(diagram, A_name, plot_type='matplotlib'):
        """Возвращает фигуру из двух графиков ху_т и х_у диаграмм

        Args:
            diagram (pd.Dataframe): таблица с данными равновесия
            A_name (any): название легколетучего компонента для подписи осей
            plot_type (str, optional): может быть и 'plotly'. Defaults to 'matplotlib'.

        Returns:
            _type_: фигура из двух графиков
        """
    
        if plot_type=='plotly':
            fig = make_subplots(rows=1, cols=2)

            fig.add_trace(go.Scatter(x=diagram['x'], y=diagram['t'],
                                    line=dict(
                                        color='rgb(0, 77, 153)',
                                        width=3),
                                    mode='lines+markers',
                                    name='жидкая фаза',
                                    ), row=1, col=1)
            fig.add_trace(go.Scatter(x=diagram['y'], y=diagram['t'],
                                    line=dict(
                                        color='rgb(41, 163, 41)',
                                        width=3),
                                    mode='lines+markers',
                                    name='паровая фаза'
                                    ), row=1, col=1)

            p_x = dfc.get_fit(diagram['x'], diagram['t'])
            p_y = dfc.get_fit(diagram['y'], diagram['t'])
            fig.add_trace(go.Scatter(x=diagram['x'], y=p_x,
                                    line=dict(
                                        color='red',
                                        width=0.7),
                                    mode='lines',
                                    name='аппроксимация жидкой фазы',
                                    ), row=1, col=1)
            fig.add_trace(go.Scatter(x=diagram['y'], y=p_y,
                                    line=dict(
                                        color='blue',
                                        width=0.7),
                                    mode='lines',
                                    name='аппроксимация паровой фазы',
                                    ), row=1, col=1)

            #x-y диаграмма
            fig.add_trace(go.Scatter(x=diagram['x'], y=diagram['y'],
                                    line=dict(
                                        color='rgb(0, 77, 153)',
                                        width=2),
                                    mode='lines+markers',
                                    name='x-y диаграмма'
                                    ), row=1, col=2)

            x_y = dfc.get_fit(diagram['x'], diagram['y'])
            fig.add_trace(go.Scatter(x=diagram['x'], y=x_y,
                                    line=dict(
                                        color='red',
                                        width=0.7),
                                    mode='lines',
                                    name='аппроксимация x-y диаграммы',
                                    ), row=1, col=2)

            _ = [0, 1]
            fig.add_trace(go.Scatter(x=_, y=_,
                                    line=dict(
                                        color='grey',
                                        width=1),
                                    mode='lines+markers',
                                    name='линия нулевого разделения'
                                    ), row=1, col=2)

            #настраиваем график внутри
            fig.update_xaxes(title_text= f'мольная доля {A_name}', 
                            row=1, col=1,
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
            fig.update_xaxes(title_text= f'мольная доля {A_name} в жидкости', 
                            row=1, col=2,
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
            fig.update_yaxes(title_text="температура, °C", 
                            row=1, col=1,
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
            fig.update_yaxes(title_text=f'мольная доля {A_name} в паре', 
                            row=1, col=2,
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)

            #настраиваем график снаружи и на границах
            indent = diagram['t'].max()/75
            fig.update_xaxes(range=[-0.05, 1.05],
                            row=1, col=1,
                            showline=True, linewidth=2, linecolor='black',
                            mirror=True,
                            ticks='inside')
            fig.update_xaxes(range=[-0.05, 1.05],
                            row=1, col=2,
                            showline=True, linewidth=2, linecolor='black',
                            mirror=True,
                            ticks='inside')
            fig.update_yaxes(range=[diagram['t'].min()-indent, diagram['t'].max()+indent], 
                            row=1, col=1,
                            showline=True, linewidth=2, linecolor='black',
                            mirror=True,
                            ticks='inside')
            fig.update_yaxes(range=[-0.05, 1.05], 
                            row=1, col=2,
                            showline=True, linewidth=2, linecolor='black',
                            mirror=True,
                            ticks='inside')
            fig.update_layout(
                autosize=False,
                margin=dict(l=20, r=5, t=20, b=2),
                showlegend=False,
                plot_bgcolor='white')
            
            return fig
                    
        if plot_type=='matplotlib':
            plt.style.use(['science', 'no-latex', 'notebook', 'grid'])

            fig = plt.figure(figsize=(20,10))
            axes = fig.add_subplot(1,2,1)
            axes2 = fig.add_subplot(1,2,2)

            axes.plot(diagram['x'], diagram['t'], label='состав жидкости')
            axes.plot(diagram['y'], diagram['t'], label='состав пара')
            axes.set_ylabel('Температура, °С', fontsize=15)
            axes.set_xlabel(f'Мольная доля {A_name}', fontsize=15)

            p_x = dfc.get_fit(diagram['x'], diagram['t'])
            p_y = dfc.get_fit(diagram['y'], diagram['t'])
            axes.plot(diagram['x'], p_x, 'o--', color='red', lw=0.5, ms=2)
            axes.plot(diagram['y'], p_y, 'o--', color='blue', lw=0.5, ms=2)
            axes.legend(loc='upper right')

            axes2.plot(diagram['x'], diagram['y'])
            x_y = dfc.get_fit(diagram['x'], diagram['y'])
            axes2.plot(diagram['x'], x_y, 'o--', color='red', lw=1, ms=2)
            _ = [0, 1]
            axes2.plot(_, _, color='black', lw=0.5)
            axes2.set_ylabel(f'Мольная доля {A_name} в паре', fontsize=15)
            axes2.set_xlabel(f'Мольная доля {A_name} в жидкости', fontsize=15)
        
        return
    
    def plot_range_phlegm_number(fig, i, diagram, stair_line_x, stair_line_y, x_y, _x, x_, W_line, P_line, N, yf, R, plot_type):
        """функция вызывается из 'range_phlegm_number' и в зависимости от выбранного 'plot_type' строит фигуру либо 
        из библиотеки matplotlib либо из plotly
        
        """ 
        
        _ = [0, 1]
        if plot_type == 'matplotlib':
            
            axes = fig.add_subplot(10,5,i)
            axes.plot(stair_line_x,stair_line_y, 'o--', lw=1, ms=1)
            axes.plot(_, _, color='black', lw=0.5)
            axes.plot(diagram['x'], x_y, color='red', lw=1, ms=2)
            axes.plot(_x, W_line, color='green', lw=1, ms=2)
            axes.plot(x_, P_line, color='green', lw=1, ms=2)
            axes.set_title(f"N = {N[yf]}, R = {round(R[i-1],2)}, Yf = {round(yf,2)}")
        
        if plot_type == 'plotly':
            
            row = int(np.ceil(i/5))
            f = lambda y: y%5 if y%5 != 0 else 5
            col = f(i)
                    
            fig.add_annotation(xref="x domain",yref="y domain",x=0, y=1.1, showarrow=False,
                    text=f"N = {N[yf]}, R = {round(R[i-1],2)}, Yf = {round(yf,2)}",
                    row=row, col=col)
            
            fig.add_trace(go.Scatter(x=diagram['x'], y=diagram['y'],
                                    line=dict(
                                        color='rgb(0, 77, 153)',
                                        width=2),
                                    mode='lines',
                                    name='x-y диаграмма'
                                    ), row=row, col=col)
            
            fig.add_trace(go.Scatter(x=diagram['x'], y=x_y,
                                    line=dict(
                                        color='red',
                                        width=0.5),
                                    mode='lines',
                                    name='аппроксимация x-y диаграммы'
                                    ), row=row, col=col)
            
            fig.add_trace(go.Scatter(x=list(*np.array(stair_line_x).T), y=list(*np.array(stair_line_y).T),
                                    line=dict(
                                        color='red',
                                        width=1),
                                    mode='lines',
                                    name='ступени изменения концентраций'
                                    ), row=row, col=col)
            
            fig.add_trace(go.Scatter(x=_x, y=W_line,
                                    line=dict(
                                        color='black',
                                        width=1),
                                    mode='lines',
                                    name='исчерпывающая рабочая линия'
                                    ), row=row, col=col)
            
            fig.add_trace(go.Scatter(x=x_, y=P_line,
                                    line=dict(
                                        color='black',
                                        width=1),
                                    mode='lines',
                                    name='укрепляющая рабочая линия'
                                    ), row=row, col=col)
            
            fig.add_trace(go.Scatter(x=_, y=_,
                                    line=dict(
                                        color='grey',
                                        width=1),
                                    mode='lines',
                                    name='линия нулевого разделения'
                                    ), row=row, col=col)
            
            #настраиваем график снаружи и на границах
            fig.update_xaxes(range=[-0.05, 1.05],
                                row=row, col=col,
                                showline=True, linewidth=2, linecolor='black',
                                mirror=True,
                                ticks='inside')
            fig.update_yaxes(range=[-0.05, 1.05], 
                                row=row, col=col,
                                showline=True, linewidth=2, linecolor='black',
                                mirror=True,
                                ticks='inside')
            
            #настраиваем график внутри
            fig.update_xaxes(row=row, col=col,
                                gridcolor='rgb(105,105,105)',
                                griddash='1px',
                                zeroline=False)
            
            fig.update_yaxes(row=row, col=col,
                                gridcolor='rgb(105,105,105)',
                                griddash='1px',
                                zeroline=False)
        
        return fig
    def plot_optimal_phlegm_number(R, Ngraf, Nfit, optimal, Ropt, plot_type):
    
        if plot_type == 'matplotlib':
            plt.style.use(['science', 'no-latex', 'notebook', 'grid'])
            fig = plt.figure(figsize=(7,7))
            axes = fig.add_subplot()
            axes.plot(R,Ngraf)
            axes.plot(R,Nfit, '--')
            axes.set_xlabel(r'R')
            axes.set_ylabel(r'N(R+1)')
            axes.set_title(f"Оптимальное флегмовое число = {np.round(R[optimal],2)}") 
            
        if plot_type == 'plotly':
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=R, y=Nfit,
                                    line=dict(
                                        color='red',
                                        width=2),
                                    mode='lines',
                                    name='аппроксимация'))
            
            fig.add_trace(go.Scatter(x=R, y=Ngraf,
                                    line=dict(
                                        color='black',
                                        width=2),
                                    mode='lines',
                                    name='значения рабочего флегмового числа'))
            
            fig.add_annotation(x=float(R[optimal]), y=float(Nfit.min()),
                text=f"R оптимальное={Ropt}", 
                showarrow=True,
                arrowhead=1)
            
            #настраиваем график
            fig.update_xaxes(
                title_text= 'R',
                gridcolor='rgb(105,105,105)',
                griddash='1px',
                showline=True, linewidth=2, linecolor='black',
                mirror=True,
                ticks='inside',
                zeroline=False)
            
            fig.update_yaxes(
                title_text="N(R+1)",
                gridcolor='rgb(105,105,105)',
                griddash='1px',
                showline=True, linewidth=2, linecolor='black',
                mirror=True,
                ticks='inside',
                zeroline=False)
            
            fig.update_layout(
                    autosize=False,
                    width=500,
                    height=500,
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
            
            return fig
        
        return
    
    def plot_plate_column_height(stair_line_x, stair_line_y, diagram, diagram_xy, kinetic_frame, kinetic_xy,
                            _x, W_line, x_, P_line, _):
        fig = go.Figure()            
        fig.add_trace(go.Scatter(x=list(map(float, stair_line_x)), y=list(map(float, stair_line_y)),
                                line=dict(
                                    color='red',
                                    width=1),
                                mode='lines',
                                name='ступени изменения концентраций'))            
        fig.add_trace(go.Scatter(x=diagram['x'], y=diagram_xy,
                                line=dict(
                                    color='black',
                                    width=0.5),
                                mode='lines',
                                name='аппроксимация x-y диаграммы'))            
        fig.add_trace(go.Scatter(x=kinetic_frame['значение кинетической кривой'].index, y=kinetic_xy,
                                line=dict(
                                    color='blue',
                                    width=2),
                                mode='lines',
                                name='кинетическая кривая'))
        fig.add_trace(go.Scatter(x=_x, y=W_line,
                                line=dict(
                                    color='black',
                                    width=1),
                                mode='lines',
                                name='исчерпывающая рабочая линия'))
        fig.add_trace(go.Scatter(x=x_, y=P_line,
                                line=dict(
                                    color='black',
                                    width=1),
                                mode='lines',
                                name='укрепляющая рабочая линия'))
        fig.add_trace(go.Scatter(x=_, y=_,
                                line=dict(
                                    color='grey',
                                    width=1),
                                mode='lines',
                                name='линия нулевого разделения'))
        
        fig.update_xaxes(range=[-0.01, 1.01],
                            showline=True, linewidth=2, linecolor='black',
                            mirror=True,
                            ticks='inside',
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
        
        fig.update_yaxes(range=[-0.05, 1.05],
                            showline=True, linewidth=2, linecolor='black',
                            mirror=True,
                            ticks='inside',
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
        
        return fig
