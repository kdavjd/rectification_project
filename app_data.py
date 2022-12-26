
import pandas as pd
from functions.data_functions import DataFunctions as dfс

am = pd.read_excel('data/atomic_mass.xlsx')
aqua_liquid_saturation = pd.read_excel('data/ph_H2O_saturation_liquid.xlsx')
aqua_vapor_saturation = pd.read_excel('data/ph_H2O_saturation_gas.xlsx')
aqua_vapor_saturation_by_pressure = pd.read_excel('data/ph_H2O_saturation_gas_by_pressure.xlsx')
mass_concentration_solution = pd.read_excel('data/mass_concentration_solution.xlsx')
boiling_point_solution = pd.read_excel('data/boiling_point_solution.xlsx')
density_solution = pd.read_excel('data/density_solution.xlsx')
vicosity_solution = pd.read_excel('data/vicosity_solution.xlsx')
specific_heat_capacity_solution = pd.read_excel('data/heat_capacity_solution.xlsx')
thermal_conductivity_solutions = pd.read_excel('data/thermal_conductivity_solutions.xlsx')
ph_gases = pd.read_excel('data/ph_gases.xlsx')
ph_organic = pd.read_excel('data/ph_organic.xlsx')
density_organic_liquid = pd.read_excel('data/density_organic_liquid.xlsx')
vicosity_organic_liquid = pd.read_excel('data/vicosity_organic_liquid.xlsx')
interfactial_tension_organic_liquid = pd.read_excel('data/interfacial_tension_organic_liquid.xlsx')
thermal_expansion_organic_liquid = pd.read_excel('data/thermal_expansion_organic_liquid.xlsx')
heat_capacity_organic_liquid = pd.read_excel('data/heat_capacity_organic_liquid.xlsx')
thermal_conductivity_organic_liquid = pd.read_excel('data/thermal_conductivity_organic_liquid.xlsx')
heat_vaporization_organic_liquid = pd.read_excel('data/heat_vaporization_organic_liquid.xlsx')
vapor_pressure_organic_liquid = pd.read_excel('data/vapor_pressure_organic_liquid.xlsx')
vicosity_organic_vapor = pd.read_excel('data/vicosity_organic_vapor.xlsx')

exclude_list =['salts','name','formula','salt']
#Поверхностное натяжение органических жидкостей [мДж/м^2]
interfactial_tension_organic_liquid = dfс.delete_hyphens(interfactial_tension_organic_liquid,exclude_list)

#Теплопроводность органических жидкостей [Вт/(м*K)]
thermal_conductivity_organic_liquid = dfс.delete_hyphens(thermal_conductivity_organic_liquid,exclude_list)

#Свойства водяного пара в состянии насыщения в зависимости от давления
aqua_vapor_saturation_by_pressure = dfс.delete_hyphens(aqua_vapor_saturation_by_pressure,exclude_list)

#Коэффициенты объемного теплового расширения органических жидкостей b*10^3, K^-1
thermal_expansion_organic_liquid = dfс.delete_hyphens(thermal_expansion_organic_liquid,exclude_list)

#Теплота парообразования органических жидкостей [кДж/кг]
heat_vaporization_organic_liquid = dfс.delete_hyphens(heat_vaporization_organic_liquid,exclude_list)

#Удельная  теплоемкость водных р-ров cp = cpAQ - w * (d1 - d2*w -d3e-3 *t^2)
specific_heat_capacity_solution = dfс.delete_hyphens(specific_heat_capacity_solution,exclude_list)

#Теплопроводность водных р-ров неорганических соединений lya = lyaAQ * (1 - f*w)
thermal_conductivity_solutions = dfс.delete_hyphens(thermal_conductivity_solutions,exclude_list)

#Давление насыщенного пара [мм.рт.ст.] над органической жидкостью
vapor_pressure_organic_liquid = dfс.delete_hyphens(vapor_pressure_organic_liquid,exclude_list)

#Удельная теплоемкость [Дж/(кг*K)] органических жидкостей
heat_capacity_organic_liquid = dfс.delete_hyphens(heat_capacity_organic_liquid,exclude_list)

#Концентрации насыщенных водных растворов неорганических веществ кг/кг при °С
mass_concentration_solution = dfс.delete_hyphens(mass_concentration_solution,exclude_list)

#Динамическая вязкость органических жидкостей [мПа*с]
vicosity_organic_liquid = dfс.delete_hyphens(vicosity_organic_liquid,exclude_list)

#Плотность органических жидкостей [кг/м^3]
density_organic_liquid = dfс.delete_hyphens(density_organic_liquid,exclude_list)

#Температуры кипения водных растворов неорганических веществ при н.у.
boiling_point_solution = dfс.delete_hyphens(boiling_point_solution,exclude_list)

#Физические свойства воды на линии насыщения
aqua_liquid_saturation = dfс.delete_hyphens(aqua_liquid_saturation,exclude_list)

#Вязкость паров органических веществ [мкПа*с]
vicosity_organic_vapor = dfс.delete_hyphens(vicosity_organic_vapor,exclude_list)

#Свойства водяного пара в состянии насыщения в зависимости от температуры
aqua_vapor_saturation = dfс.delete_hyphens(aqua_vapor_saturation,exclude_list)

#Вязкость водных растворов неорганических веществ u = uaq*exp^[w(b1 + b2e-2 * t - b3e-7 * t^2)]
vicosity_solution = dfс.delete_hyphens(vicosity_solution,exclude_list)

#Плотность водных растворов p = paq*exp^[w*(a1 + a2e-4 * t - a3e-6 * t^2)], w[кг/кг], t[°C], p[кг/м^3], u[Па*c]
density_solution = dfс.delete_hyphens(density_solution,exclude_list)

#Основные характеристики органических веществ
ph_organic = dfс.delete_hyphens(ph_organic,exclude_list)

#Свойства газов при н.у.
ph_gases = dfс.delete_hyphens(ph_gases,exclude_list)
#am = delete_hyphens(am)

class Сomponent():
                
    def __init__(self, name):        
        self.interfactial_tension_organic_liquid = interfactial_tension_organic_liquid[interfactial_tension_organic_liquid['name'] == name].drop('name', axis=1)
        self.thermal_conductivity_organic_liquid = thermal_conductivity_organic_liquid[thermal_conductivity_organic_liquid['name'] == name].drop('name', axis=1)
        self.thermal_expansion_organic_liquid = thermal_expansion_organic_liquid[thermal_expansion_organic_liquid['name'] == name].drop('name', axis=1)
        self.heat_vaporization_organic_liquid = heat_vaporization_organic_liquid[heat_vaporization_organic_liquid['name'] == name].drop('name', axis=1)        
        self.vapor_pressure_organic_liquid = vapor_pressure_organic_liquid[vapor_pressure_organic_liquid['name'] == name].drop('name', axis=1)
        self.heat_capacity_organic_liquid = heat_capacity_organic_liquid[heat_capacity_organic_liquid['name'] == name].drop('name', axis=1)
        self.vicosity_organic_liquid = vicosity_organic_liquid[vicosity_organic_liquid['name'] == name].drop('name', axis=1)
        self.vicosity_organic_liquid = vicosity_organic_liquid[vicosity_organic_liquid['name'] == name].drop('name', axis=1)
        self.density_organic_liquid = density_organic_liquid[density_organic_liquid['name'] == name].drop('name', axis=1)
        self.vicosity_organic_vapor = vicosity_organic_vapor[vicosity_organic_vapor['name'] == name].drop('name', axis=1)
        self.ph_organic = ph_organic[ph_organic['name'] == name]
        


