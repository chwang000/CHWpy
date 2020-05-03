
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

sys.path.append(r'E:\CH\programming\CHWpy\chw_sci_function')
from xrd_functions import *

test = lorentz_profile_range(xo=5,fwhm=1,x_min=1,x_max=10,x_step=0.1)
test2 = pv_profile_range(xo=5,NA=0.1,NB=0.2,fwhm=1,x_min=1,x_max=10,x_step=0.1)
test3 = pv_peak_range(xo=5,x_min=1,x_max=10,x_step=0.1,ha=0.02,hb=0.02,hc=0.02,lora=0.02,lorb=0.02,lorc=0.02)
test1 = tchz_profile_range(5,U=0.00039,V=-0.00221,W=-0.00146,X=0.0,Y=0.00957,Z=0.0,x_min=1,x_max=10,x_step=0.1)
plt.plot(test[:,0],test[:,1])
plt.plot(test1[:,0],test1[:,1])
plt.plot(test2[:,0],test2[:,1])
plt.plot(test3[:,0],test3[:,1])
plt.show()

'''
def hstack_1d_ary(ary1=[],ary2=[]):
    """ put two 1D array to one 2D array"""
    print(type(ary1),type(ary2))
    ary1 = np.array(ary1)
    ary2 = np.array(ary2)
    ary1 = ary1[:,np.newaxis]
    ary2 = ary2[:,np.newaxis]
    return np.concatenate((ary1,ary2),axis=1)


def GProfileRange(xo,fwhm,x_min,x_max,x_step):
    """XRD Peak functions in whole range, """
    out = []
    x = np.arange(x_min,x_max,x_step,dtype=float)
    j = 0
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    y = [g1 / (fwhm * np.sqrt(np.pi)) * np.exp(-g2 * (xtmp - xo)**2 / (fwhm**2)) for xtmp in x]
    #y = y /np.max(y)
    profile = hstack_1d_ary(x,y)
    return profile

def lorentz_profile_range(xo,fwhm,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Lorentz type"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
    j = 0
    l1 = 2 / math.pi
    l2 = 4
    y = [(l1 / fwhm) / (1 + (l2 * (xtmp - xo)**2 / (fwhm**2))) for xtmp in x]
    y = np.array(y)
    #y = y / np.max(y)
    profile = hstack_1d_ary(x, y)
    return profile



#Pseudo-Voigt
def pv_profile_range(xo,fwhm,NA,NB,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Pseudo-Voigt"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
    j = 0
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    l1 = 2 / math.pi
    l2 = 4
    y = [(NA + NB * xtmp) * g1 / (fwhm * np.sqrt(np.pi)) * np.exp(-g2 * (xtmp - xo)**2 / (fwhm**2))
        + (1-(NA + NB * xtmp)) * (l1 / fwhm) / (1 + (l2 * (xtmp - xo)**2 / (fwhm**2))) for xtmp in x]
    #y = y / np.max(y)
    profile = hstack_1d_ary(x, y)
    return profile

#Pseudo-Voigt in refinement format
def pv_peak_range(xo,ha,hb,hc,lora,lorb,lorc,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Pseudo-Voigt in refinement format"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
    j = 0
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    l1 = 2 / math.pi
    l2 = 4
    y = [(lora + lorb * np.tan(xtmp * np.pi /360) + lorc / np.cos(xtmp * np.pi /360))
       * g1 / ((ha + hb * np.tan(xtmp * np.pi /360) + hc / np.cos(xtmp * np.pi /360)) * np.sqrt(np.pi))
       * np.exp(-g2 * (xtmp - xo)**2 / ((ha + hb * np.tan(xtmp * np.pi /360) + hc / np.cos(xtmp * np.pi /360))**2))
       + (1-(lora + lorb * np.tan(xtmp * np.pi /360) + lorc / np.cos(xtmp * np.pi /360)))
       * (l1 / (ha + hb * np.tan(xtmp * np.pi /360) + hc / np.cos(xtmp * np.pi /360)))
       / (1 + (l2 * (xtmp - xo)**2 / ((ha + hb * np.tan(xtmp * np.pi /360) + hc / np.cos(xtmp * np.pi /360))**2))) for xtmp in x]
    #y = y / np.max(y)
    profile = hstack_1d_ary(x, y)
    return profile


#TCHZ profile
#ref: TCHZ_Peak_Type(pku, 0.00039,pkv, -0.00221,pkw, -0.00146,!pkz, 0.0000,pky, 0.00957,!pkx, 0.0000)
def tchz_profile_range(xo,U=0.00039,V=-0.00221,W=-0.00146,X=0.0,Y=0.00957,Z=0.0,x_min=5,x_max=120,x_step=0.01):
    x = np.arange(x_min,x_max,x_step,dtype=float)
    out = []
    j = 0
    A = 2.69269
    B = 2.42843
    C = 4.47163
    D = 0.07842
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    l1 = 2 / math.pi
    l2 = 4
    for xtmp in x:
        x_rad = xtmp * math.pi /360
        tmp_0 = math.fabs(U * (math.tan(xtmp * math.pi /360))**2 + V * math.tan(xtmp * math.pi /360) + W + Z * (math.cos(xtmp * math.pi /360))**2)
        GM_G = math.sqrt(tmp_0)
        GM_L = X * math.tan(xtmp * math.pi /360) + Y / math.cos(xtmp * math.pi /360)
        tmp_1 = GM_G**5 + A * GM_G**4 * GM_L + B * GM_G**3 * GM_L**2 + C * GM_G**2 * GM_L**3 \
        + D * GM_G * GM_L**4 + GM_L**5
        fwhm = math.pow(tmp_1,0.2)
        q = GM_L / fwhm
        mixing = 1.36603 * q - 0.47719 * q**2 + 0.1116 * q**3
        y = mixing * g1 / (fwhm * math.sqrt(math.pi)) * math.exp(-g2 * (xtmp - xo)**2 / (fwhm**2)) \
        + (1-mixing) * (l1 / fwhm) / (1 + (l2 * (xtmp - xo)**2 / (fwhm**2)))
        if j == 0:
            out = [xtmp,y]
        else:
            out = np.row_stack((out, [xtmp,y]))
        j += 1
    #out[:,1] = out[:,1] / np.max(out[:,1])
    return out
'''