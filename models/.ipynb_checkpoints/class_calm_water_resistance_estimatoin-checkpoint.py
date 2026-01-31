# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:58:27 2024


"""


'''
静水阻力估算
不考虑风浪流
'''
import numpy as np
import attr
    
@attr.s(frozen=True, auto_attribs=True)
# COSCO的船
class ShipParam(object):
    # 几何参数
    Lwl: float = 279.9*0.9
    Lpp: float = 279.9*0.95
    B: float = 40.3 # 水线宽度B_wl?
    Awl: float = 10151.973 # Lwl*40.3 # 截面面积
    Vol: float = 87872  # displacement volume

    # hull coefficients 这些值全部取的经验参数
    Cp: float = 0.69 #知道C_B和C_wp后可算
    Cm: float = 0.98 # 知道A_m后可算
    C_B: float = 0.72 #可算0.8219 
    C_wp: float = 0.82 #C_wl, 可算
    L_CB: float = -0.75 # 纵向浮心位置，需要

    #吃水
    T: float =  14.021# 吃水
    Tf: float = 14.021 # forward draught，
    
    Ss: float = 13682 # 不清楚，需要
    ks: float = 150 * 10 ** - 6 # 需要

    g: float = 9.81
    A_bt: float = 30.8 #30.8 # transverse sectional area，需要
    hb: float = 4 # 4 不太清楚，需要
    C_stern: float = 10 # 根据具体船型取值
    k2: float = 1.5 # 有经验值，也有复杂计算公式，需要
    transom_a: float = 42.6  # 不清楚，需要
    i_E: float = 5.0 # 可通过复杂公式计算

    Sapp: float = 50 # full scale wetted surface，需要
    I: float = 14762925 # 不清楚，需要
    kyy: float = 60.2966 # 不清楚，需要
    A_transverse: float = 41*5 # 不清楚，需要
    # A_M:float= 200 #the midship section area under water m2
    SFOC:float=200 #g/kWh ship's fuel efficiency

    
def speedGPS2Water(v, heading_ship, cu, cv):
    '''
    Function to convert speed over ground into speed through water
    '''

    V_water = v - cu * np.sin(np.deg2rad(heading_ship)) - cv * np.cos(np.deg2rad(heading_ship))

    return V_water

def speed2shallow(v, waterdepth, ship = ShipParam()):
    '''
    Function to calculate the reduction of ship speed due to shallow water
    v : speed through water [m/s]
    '''    
    imarea_index = ship.A_M * ship.T / 6.8 / pow(abs(waterdepth), 2)
    if imarea_index >= 0.5:
        delta_v = v * (0.1242 * (imarea_index -0.05) + 1 - np.sqrt(np.tanh(ship.g * abs(waterdepth) / pow(v, 2.0))))
        V_shallow = v + delta_v
    else:
        V_shallow = v

    return V_shallow

@attr.s(init=False)
class HoltropParam(object):
    def __init__(self, ship: ShipParam = ShipParam()):
        """
        Returns parameters required for the Holtrop calm water resistance calculation

        :param ship: dictionary of ship parameters
        :return: dictionary of holtrop parameters
        """
        self.c13 = 1 + 0.003 * ship.C_stern

        if ship.T/ship.Lwl > 0.05:
            self.c12 = (ship.T / ship.Lwl) ** 0.2228446
        elif ship.T/ship.Lwl < 0.02:
            self.c12 = 0.479948
        else:
            self.c12 = 48.20 * (ship.T / ship.Lwl - 0.02) ** 2.078 + 0.479948
        self.L_R = ship.Lwl * (1 - ship.Cp + 0.06 * ship.Cp * ship.L_CB / (4 * ship.Cp - 1))
        self.k1 = (self.c13 * (0.93 + self.c12 * ((ship.B / self.L_R) ** 0.92497) *
                ((0.95 - ship.Cp) ** -.521448) * ((1 - ship.Cp + 0.0225 * ship.L_CB) ** 0.6906)))
        self.ie = (1 + 89 * np.exp(- (ship.Lwl / ship.B) ** 0.80856 * (1 - ship.C_wp) ** 0.30484
                                   * (1 - ship.Cp - 0.0225 * ship.L_CB) ** 0.6367 * (
                                           self.L_R / ship.B) ** 0.34574 *
                                   (100 * ship.Vol / (ship.Lwl ** 3)) ** 0.16302))
        if ship.Cp < 0.80:
            self.c16 = 8.07981 * ship.Cp - 13.8673 * (ship.Cp ** 2) + 6.984388 * ship.Cp ** 3
        elif ship.Cp > 0.80:
            self.c16 = 1.73014 - 0.7067 * ship.Cp

        if (ship.Lwl ** 3) / ship.Vol < 512:
            self.c15 = -1.69385
        elif (ship.Lwl ** 3) / ship.Vol > 1727:
            self.c15 = 0
        else:
            self.c15 = -1.69385 + (ship.Lwl / (ship.Vol ** (1/3)) - 8)/2.36

        self.m1 = 0.0140407 * ship.Lwl / ship.T - 1.75254 * ship.Vol ** (1 / 3) / ship.Lwl \
                  - 4.79323 * ship.B / ship.Lwl - self.c16

        if ship.B / ship.Lwl < 0.11:
            self.c7 = 0.229577 * (ship.B / ship.Lwl) ** 0.33333
        elif ship.B / ship.Lwl > 0.25:
            self.c7 = 0.5 - 0.0625 * ship.Lwl / ship.B
        else:
            self.c7 = ship.B / ship.Lwl

        self.c1 = 2223105 * self.c7 ** 3.78613 * (ship.T / ship.B) ** 1.07961 * (90 - self.ie) ** \
                  (-1.37565)
        self.c3 = 0.56 * ship.A_bt ** 1.5 / (ship.B * ship.T * (0.31 * np.sqrt(ship.A_bt) +
                                                                ship.Tf - ship.hb))
        self.c5 = 1 - 0.8 * ship.transom_a / (ship.B * ship.T * ship.Cm)
        self.c2 = np.exp(- 1.89 * np.sqrt(self.c3))

        if ship.Tf / ship.Lwl < 0.04:
            self.c4 = ship.Tf / ship.Lwl
        elif ship.Tf / ship.Lwl > 0.04:
            self.c4 = 0.04

        if ship.Lwl / ship.B < 12:
            self.lamda = 1.446 * ship.Cp - 0.03 * ship.Lwl / ship.B
        elif ship.Lwl / ship.B > 12:
            self.lamda = 1.446 * ship.Cp - 0.36
        self.d = - 0.9
        self.PB = 0.56 * np.sqrt(ship.A_bt) / (ship.Tf - 1.5 * ship.hb)
        self.C_A = (0.006 * (ship.Lwl + 100) ** -0.16 - 0.00205 + 0.003 * np.sqrt(ship.Lwl / 7.5)
                    * ship.C_B ** 4 * self.c2 * (0.04 - self.c4))

def reynolds_number(v, ship = ShipParam()):
    '''
    Function to calculate reynolds number
    '''
    re_v = v * ship.Lpp * 1000000. / 1.187
    return re_v

def frictional_coefficient(v, ship = ShipParam()):
    re_v = reynolds_number(v, ship)
    c_f = 0.075 / pow((np.log10(re_v) - 2), 2.0)

    return c_f

def frictional_resistance(v, ship = ShipParam(), rho_sw = 1025.):
    c_f = frictional_coefficient(v, ship)
    r_f = 0.5 * rho_sw * ship.Ss * v * v * c_f

    return r_f

# def calculate_frictional_resistance(dict_ship_parameters,fluid_density=float(999.138/1000),kinematic_viscosity_factor=1.1899999999999998e-06,CF_reduction=0.0):
#     #自己之前的方法
#     Reynolds_number=calculate_Reynolds_number(dict_ship_parameters,kinematic_viscosity_factor)
#     C_F=calculate_C_F(Reynolds_number)
#     #减阻效应
#     C_F=C_F*(1-CF_reduction)
#     S=calculate_watter_area_hull(dict_ship_parameters)
#     v=dict_ship_parameters['V ship speed(knots)']*0.514
#     R_F=0.5*fluid_density*S*math.pow(v,2)*C_F
#     return R_F

def appendage_resistance(v, ship = ShipParam(), rho_sw = 1025.):

    c_f = frictional_coefficient(v, ship)
    r_app = 0.5 * rho_sw * v * v * ship.Sapp * ship.k2 * c_f

    return r_app

def froude_number(v, ship = ShipParam()):
    '''
    Function to calculate froude number
    '''
    fn_v = v / np.sqrt(ship.g * ship.Lpp)
    return fn_v

def calm_water_wave_resistance(v, ship = ShipParam(), rho_sw = 1025.):
    """
    Function to calculate resistance for appendages
    """

    h = HoltropParam(ship)
    fn = froude_number(v)
    m2 = h.c15 * ship.Cp * ship.Cp * np.exp(-0.1 * pow(fn, -2.))
    try:
        r_w = h.c1 * h.c2 * h.c5 * ship.Vol * rho_sw * ship.g * np.exp(h.m1 * pow(fn, h.d) + m2 * np.cos(h.lamda * pow(fn, -2.)))
    except Exception as e:
        print("An error occurred:", e)
        print(f"fn: {fn}, h.c1: {h.c1}, h.c2: {h.c2}, h.c5: {h.c5}, ship.Vol: {ship.Vol}, rho_sw: {rho_sw}, ship.g: {ship.g}")
        print(f"h.m1: {h.m1}, h.d: {h.d}, m2: {m2}, h.lamda: {h.lamda}")
    return r_w

def bulb_resistance(v, ship = ShipParam(), rho_sw = 1025.):
    """
    Function to calculate bulb resistance
    """

    h = HoltropParam(ship)

    fni = v / np.sqrt(ship.g * (ship.Tf - ship.hb - 0.25 * np.sqrt(ship.A_bt)) + 0.15 * v * v)
    r_b = 0.11 * np.exp(-3 / h.PB / h.PB) * fni * fni * fni * pow(ship.A_bt, 1.5) * rho_sw * ship.g / (1 + fni * fni)

    return r_b

def correlation_resistance(v, ship = ShipParam(), rho_sw = 1025.):
    """
    Function to calculate model-ship correlation resistance
    """

    h = HoltropParam(ship)
    r_a = 0.5 * rho_sw * v * v * ship.Ss * h.C_A

    return r_a

def calm_water_resistance(v, ship = ShipParam(), rho_sw = 1025.):
    """
    Function to calculate calm water resistance, i.e. absence of wind, waves and currents
    """

    h = HoltropParam(ship)
    r_f = frictional_resistance(v, ship, rho_sw)
    r_app = appendage_resistance(v, ship, rho_sw)
    r_w = calm_water_wave_resistance(v, ship, rho_sw)
    r_b = bulb_resistance(v, ship, rho_sw)
    r_a = correlation_resistance(v, ship, rho_sw)
    r_cw = r_f * h.k1 + r_app + r_w + r_b + r_a

    return r_cw

def calm_resistance(v, heading_ship, h_waterdepth, 
                    currentU, currentV,ship = ShipParam()):
    '''
    计算
    h_waterdepth 水深 m
    '''
    V_water = speedGPS2Water(v, heading_ship, currentU, currentV)
    # V_shallow = speed2shallow(V_water, h_waterdepth) #浅水效应
    V_shallow=V_water
    r_calm = calm_water_resistance(V_shallow,ship)
    
    return r_calm

def speed_to_power(v, heading_ship,h_waterdepth,
                   currentU, currentV,
                   ship = ShipParam()):
    '''
    计算
    
    h_waterdepth, 
    windU, windV, Hs, Tp, Hdg, 
    '''
    V_water = speedGPS2Water(v, heading_ship, currentU, currentV)
    # V_shallow = speed2shallow(V_water, h_waterdepth)
    V_shallow=V_water
    
    r_calm = calm_resistance(V_shallow,heading_ship,h_waterdepth,
                       currentU, currentV)

    r_total = (r_calm) / 1000
    # r_total=r_calm/1000
    etaR = 1.0 # 准确计算较复杂，但整体波动小，经验值是1-1.07或者0.98
    etaO = 0.60 # 可以算，但需要知道V_A和thruster force T的具体意义，经验值0.55-0.7
    etaS = 0.99 # 无计算公式，取经验值，0.99，0.98，0.95-0.96三种情况
    etaH = 1.1 # 要计算t, w，1-t/1-w,需要知道力和速度的关系。

    P_eff  = r_total * V_water 
    P_shaft = P_eff / (etaR * etaO * etaS * etaH)
    Fuel_kg_per_hour = P_shaft * ship.SFOC * 1e-3
    Fuel_kg_per_nm   = P_shaft * ship.SFOC * 1e-3 / v / 0.514444

    return P_eff, P_shaft, Fuel_kg_per_hour, Fuel_kg_per_nm

def Cal_R_calm(row):
    heading_ship = row['heading']
    cu = row['stream_val']*np.sin(np.deg2rad(row['stream_direction']))
    cv = row['stream_val']*np.cos(np.deg2rad(row['stream_direction']))
    sog = row['SOG']*0.5144

    V_water = speedGPS2Water(sog, heading_ship, cu, cv)
    V_shallow=V_water
    r_calm = calm_water_resistance(V_shallow)

    return r_calm/1000


if __name__=='__main__':

    v_speed=20*0.514 #船舶对地航速 m/s
    heading_ship=60 #船首向
    currentU, currentV=3,2 #水流两个方向的速度 m/s

    P_eff, P_shaft, Fuel_kg_per_hour, Fuel_kg_per_nm=speed_to_power(
        v_speed,heading_ship,0,
        currentU, currentV,)
    print(Fuel_kg_per_nm)
    
    