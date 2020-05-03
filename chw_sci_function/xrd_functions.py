# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:52:49 2019
Functions used in XRD analysis
XRD 常用的参数与方程函数
@author: CHWang
"""

import math
import numpy as np
import random
import pandas as pd
import gemmi
#import matplotlib.pyplot as plt
#import decimal
    #decimal.getcontext().prec = num_decimal
    #x = decimal.Decimal(x)


def hstack_1d_ary(ary1=[],ary2=[]):
    """ put two 1D array to one 2D array"""
    ary1 = np.array(ary1)
    ary2 = np.array(ary2)
    ary1 = ary1[:,np.newaxis]
    ary2 = ary2[:,np.newaxis]
    return np.concatenate((ary1,ary2),axis=1)


# Wavelength list
def wavelength_line(X_line):
    """ Wavelength list of different element:
    CuKa1,CuKa2,CuKb1,MoKa1,MoKa2,MoKb1,FeKa1,FeKa2,FeKb1,CoKa1,
    CrKa1,CrKa2,WKa1,AgKa1"""
    if X_line == "CuKa1":
        return 1.540562
    elif X_line == "CuKa2":
        return 1.54439
    elif X_line == "CuKb1":
        return 1.39222
    elif X_line == "MoKa1":
        return 0.70930
    elif X_line == "MoKa2":
        return 0.71359
    elif X_line == "MoKb1":
        return 0.63229
    elif X_line == "FeKa1":
        return 1.93604
    elif X_line == "FeKa2":
        return 1.93998
    elif X_line == "FeKb1":
        return 1.75661
    elif X_line == "CoKa1":
        return 1.78897
    elif X_line == "CrKa1":
        return 2.28970
    elif X_line == "CrKa2":
        return 2.293606
    elif X_line == "WKa1":
        return 0.2090100
    elif X_line == "AgKa1":
        return 0.5594075


# keV to Angstrom
def kev_to_wavelength(kev):
    """ Convert keV to Angstrom """
    return 12.3984 / kev


# Calculate cell volume.
def cell_vol(lpa,lpb,lpc,lpal,lpbe,lpga):
    """Calculate cell volume from cell parameters: a,b,c,alpha,beta,gamma."""
    tmp_V = lpa*lpb*lpc*math.sqrt(1-math.cos(lpal*math.pi/180.0)*math.cos(lpal*math.pi/180.0)\
                                  -math.cos(lpbe*math.pi/180.0)*math.cos(lpbe*math.pi/180.0) \
                                  -math.cos(lpga*math.pi/180.0)*math.cos(lpga*math.pi/180.0)\
                                  +2*math.cos(lpal*math.pi/180.0)*math.cos(lpbe*math.pi/180.0)*math.cos(lpga*math.pi/180.0))
    return tmp_V


# Calculate the d-spacing of hkl
def d_spacing(lpa,lpb,lpc,lpal,lpbe,lpga,h,k,l):
    """Calculate the d-spacing of hkl with cell parameters: a,b,c,alpha,beta,gamma."""
    tmp_V = cell_vol(lpa,lpb,lpc,lpal,lpbe,lpga)
    tmp_S11 = (lpb) * (lpb) * (lpc) * (lpc) * (math.sin(lpal * math.pi / 180)) * (math.sin(lpal * math.pi / 180))
    tmp_S22 = (lpa) * (lpa) * (lpc) * (lpc) * (math.sin(lpbe * math.pi / 180)) * (math.sin(lpbe * math.pi / 180))
    tmp_S33 = (lpa) * (lpa) * (lpb) * (lpb) * (math.sin(lpga * math.pi / 180)) * (math.sin(lpga * math.pi / 180))
    tmp_S12 = lpa * lpb * (lpc) * (lpc) * (math.cos(lpal * math.pi / 180) * math.cos(lpbe * math.pi / 180) - math.cos(lpga * math.pi / 180))
    tmp_S23 = lpb * lpc * (lpa) * (lpa) * (math.cos(lpbe * math.pi / 180) * math.cos(lpga * math.pi / 180) - math.cos(lpal * math.pi / 180))
    tmp_S13 = lpa * lpc * (lpb) * (lpb) * (math.cos(lpal * math.pi / 180) * math.cos(lpga * math.pi / 180) - math.cos(lpbe * math.pi / 180))
    sq_inv_d = (1 / tmp_V / tmp_V) * (tmp_S11 * h * h + tmp_S22 * k * k + tmp_S33 * l * l
    + 2 * tmp_S12 * h * k + 2 * tmp_S23 * k * l + 2 * tmp_S13 * h * l)
    #d_spacing = math.sqrt(1/sq_inv_d)
    return math.sqrt(1/sq_inv_d)
    #1/d^2 = (1/V^2) (S11*h^2 + S22*k^2 + S33*l^2 + 2*S12*h*k + 2*S23*k*l+ 2*S13*h*l)


def sg_symbol_to_sg_no(symbol='Fm-3m'):
    """Find the spacegroup number of a spacegroup symbol.
    if error, return 1 (P1)."""
    symbol = str(symbol)
    symbol_origin = symbol
    sg_no = 1
    try:
        sg_no = gemmi.SpaceGroup(symbol).number
    except:
        pass
    if sg_no == 0 and symbol[-2:] == ':2':
        symbol = symbol.replace(':2','')
        sg_no = gemmi.SpaceGroup(symbol).number
    elif sg_no == 0 and symbol[-2:] != ':2':
        print(symbol_origin, 'not recognised, set to "P1".')
        sg_no = 1
    return sg_no


# General Peak functions
def gaussian_profile(xo,fwhm,x_cut_off,x_interval):
    """ Gaussian Profile"""
    out = []
    j = 0
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    num_interval = math.ceil(x_cut_off / x_interval)
    for i in range(-num_interval, num_interval):
        x = xo + i * x_interval
        y = g1 / (fwhm * math.sqrt(math.pi)) * math.exp(-g2 * (x - xo)**2 / (fwhm**2))
        if j == 0:
            out = [x,y]
        else:
            out = np.row_stack((out, [x,y]))
        j += 1
    return out


def lorentz_profile(xo,fwhm,x_cut_off,x_interval):
    """ Lorentz Profile"""
    out = []
    j = 0
    l1 = 2 / math.pi
    l2 = 4
    num_interval = math.ceil(x_cut_off / x_interval)
    for i in range(-num_interval, num_interval):
        x = xo + i * x_interval
        y = (l1 / fwhm) / (1 + (l2 * (x - xo)**2 / (fwhm**2)))
        if j == 0:
            out = [x,y]
        else:
            out = np.row_stack((out, [x,y]))
        j += 1
    return out


#Pseudo-Voigt
def pv_profile(xo,fwhm,NA,NB,x_cut_off,x_interval):
    """Pseudo-Voigt profile"""
    out = []
    j = 0
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    l1 = 2 / math.pi
    l2 = 4
    num_interval = math.ceil(x_cut_off / x_interval)
    for i in range(-num_interval, num_interval):
        x = xo + i * x_interval
        mixing = NA + NB * x
        y = mixing * g1 / (fwhm * math.sqrt(math.pi)) * math.exp(-g2 * (x - xo)**2 / (fwhm**2)) \
        + (1-mixing) * (l1 / fwhm) / (1 + (l2 * (x - xo)**2 / (fwhm**2)))
        if j == 0:
            out = [x,y]
        else:
            out = np.row_stack((out, [x,y]))
        j += 1
    return out


#Pseudo-Voigt in refinement format
def pv_peak(xo,ha,hb,hc,lora,lorb,lorc,x_cut_off,x_interval):
    """Pseudo-Voigt in refinement format"""
    out = []
    j = 0
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    l1 = 2 / math.pi
    l2 = 4
    num_interval = math.ceil(x_cut_off / x_interval)
    for i in range(-num_interval, num_interval):
        x = xo + i * x_interval
        x_rad = x * math.pi /360
        fwhm = ha + hb * math.tan(x_rad) + hc / math.cos(x_rad)
        mixing = lora + lorb * math.tan(x_rad) + lorc / math.cos(x_rad)
        y = mixing * g1 / (fwhm * math.sqrt(math.pi)) * math.exp(-g2 * (x - xo)**2 / (fwhm**2)) \
        + (1-mixing) * (l1 / fwhm) / (1 + (l2 * (x - xo)**2 / (fwhm**2)))
        if j == 0:
            out = [x,y]
        else:
            out = np.row_stack((out, [x,y]))
        j += 1
    return out


# TCHZ profile
# ref: TCHZ_Peak_Type(pku, 0.00039,pkv, -0.00221,pkw, -0.00146,!pkz, 0.0000,pky, 0.00957,!pkx, 0.0000)
def tchz_profile(xo,U=0.00039,V=-0.00221,W=-0.00146,X=0.0,Y=0.00957,Z=0.0,x_cut_off=1.0,x_interval=0.01):
    """TCHZ profile in general Refinement software
    ref: TCHZ_Peak_Type(pku, 0.00039,pkv, -0.00221,pkw, -0.00146,!pkz, 0.0000,pky, 0.00957,!pkx, 0.0000)"""
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
    num_interval = math.ceil(x_cut_off / x_interval)
    for i in range(-num_interval, num_interval):
        x = xo + i * x_interval
        x_rad = x * math.pi /360
        tmp_0 = math.fabs(U * (math.tan(x_rad))**2 + V * math.tan(x_rad) + W + Z * (math.cos(x_rad))**2)
        GM_G = math.sqrt(tmp_0)
        GM_L = X * math.tan(x_rad) + Y / math.cos(x_rad)
        tmp_1 = GM_G**5 + A * GM_G**4 * GM_L + B * GM_G**3 * GM_L**2 + C * GM_G**2 * GM_L**3 \
        + D * GM_G * GM_L**4 + GM_L**5
        fwhm = math.pow(tmp_1,0.2)
        q = GM_L / fwhm
        mixing = 1.36603 * q - 0.47719 * q**2 + 0.1116 * q**3
        y = mixing * g1 / (fwhm * math.sqrt(math.pi)) * math.exp(-g2 * (x - xo)**2 / (fwhm**2)) \
        + (1-mixing) * (l1 / fwhm) / (1 + (l2 * (x - xo)**2 / (fwhm**2)))
        if j == 0:
            out = [x,y]
        else:
            out = np.row_stack((out, [x,y]))
        j += 1
    return out


# XRD Peak functions in whole range
def gaussian_profile_range(xo,fwhm,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Gaussian type"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    y = [g1 / (fwhm * np.sqrt(np.pi)) * np.exp(-g2 * (xtmp - xo)**2 / (fwhm**2)) for xtmp in x]
    #y = y /np.max(y)
    profile = hstack_1d_ary(x,y)
    return profile


def lorentz_profile_range(xo,fwhm,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Lorentz type"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
    l1 = 2 / math.pi
    l2 = 4
    y = [(l1 / fwhm) / (1 + (l2 * (xtmp - xo)**2 / (fwhm**2))) for xtmp in x]
    #y = y / np.max(y)
    profile = hstack_1d_ary(x, y)
    return profile


#Pseudo-Voigt
def pv_profile_range(xo,fwhm,NA,NB,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Pseudo-Voigt"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
    g1 = 2 * math.sqrt(math.log(2)/math.pi)
    g2 = 4 * math.log(2)
    l1 = 2 / math.pi
    l2 = 4
    y = [(NA + NB * xtmp) * g1 / (fwhm * np.sqrt(np.pi)) * np.exp(-g2 * (xtmp - xo)**2 / (fwhm**2)) \
        + (1-(NA + NB * xtmp)) * (l1 / fwhm) / (1 + (l2 * (xtmp - xo)**2 / (fwhm**2))) for xtmp in x]
    #y = y / np.max(y)
    profile = hstack_1d_ary(x, y)
    return profile


#Pseudo-Voigt in refinement format
def pv_peak_range(xo,ha,hb,hc,lora,lorb,lorc,x_min,x_max,x_step):
    """XRD Peak functions in whole range, Pseudo-Voigt in refinement format"""
    x = np.arange(x_min,x_max,x_step,dtype=float)
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
    """XRD Peak functions in whole range, TCHZ profile
    ref: TCHZ_Peak_Type(pku, 0.00039,pkv, -0.00221,pkw, -0.00146,!pkz, 0.0000,pky, 0.00957,!pkx, 0.0000)"""
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

def hkl_read(work_directory, sym_class, num_runs, num_peaks, zero_error, twoTh_error, twoTh_min):
    #zero = random.uniform(-0.1,0.1)
    error = twoTh_error               #Data errors used for the reflection merging
    index_out_id = 1
    #read data
    i = 1
    cell_prm = []
    cell_prm_read = []
    cell_prm_use = []
    index = []
    index_read = []
    index_read_tmp = []
    while (i <= num_runs):
        index_out_id = i
        zero = random.gauss(0, zero_error) #2th zero error to be applied
        if i%1000 == 0:
            print("Symmetry Class:", sym_class, ". NO. ", i, "Read.")
            #print("Zeroshift: ", format(zero, '0.4f'), "applied.")
        i += 1
        dataname = sym_class+'_hkl_%05d.txt' %index_out_id
        prmname = sym_class+'_prm_%05d.txt' %index_out_id
        fulldataname = work_directory + "\\"+ dataname
        fullprmname = work_directory + "\\"+ prmname

        #Read the cell parameters.
        with open(fullprmname,"r") as prmF:
            prmline = prmF.readline().strip()
            prmline = prmline.split()
            wavelength_read = np.array([float(prmline[0])])
            cell_prm_read = np.array([[float(prmline[0]),float(prmline[1]), float(prmline[2]), float(prmline[3]), float(prmline[4]),float(prmline[5]),float(prmline[6]),float(prmline[7]),int(prmline[8])]])
            cell_prm_use = np.array([[zero,float(prmline[1]), float(prmline[2]), float(prmline[3]), float(prmline[4]),float(prmline[5]),float(prmline[6]),int(prmline[8])]])
            #print(cell_prm_read,np.shape(cell_prm_read))
            cell_sg = int(cell_prm_read[:,7])
            #print(wavelength_read)
            #print(cell_prm_read)
            #print(cell_sg)
            #prmF.close
            if i <= 2:
                cell_prm = cell_prm_use
                #print(cell_prm)
            else:
                cell_prm = np.concatenate((cell_prm, cell_prm_use))
        #print(cell_prm,cell_prm.shape)
        #Read hkl index from data files
        pks_read = pd.read_csv(fulldataname,sep="\s+")
        #Sort data ascending
        pks_read = pks_read.sort_values(by='2th(CuKa)') #ascending=False）
        #print(pks_read)
        pks_2th = pks_read['2th(CuKa)'].values + zero # Apply zero shift
        index_read = np.array(pks_2th)
        #print("index_read: ", index_read[(i-1)])

        #Read num_peaks DIFFERENT peaks
        n = 0
        m = 0
        index_read_tmp = []
        while (n < num_peaks):
            #Exclude the reflections < twoTh_min
            if n == 0:
                for tmp0 in index_read:
                    try:
                        tmp1 = index_read[m]
                    except IndexError:
                        tmp1 = 0.0
                        break
                    if tmp1 < twoTh_min:
                        m += 1
                    else:
                        break
                index_read_tmp.append(index_read[m])
                #print("n: ",n,"index_read_tmp: ", index_read_tmp)
                m += 1
                n += 1
            else:
                for tmp0 in index_read:
                    try:
                        tmp1 = index_read[m]
                    except IndexError:
                        tmp1 = 0.0
                    #print("m: ",m,"tmp1: ",tmp1)
                    tmp2 = index_read_tmp[(n-1)]
                    #print("n: ",n,"tmp2: ",tmp2)
                    tmp3 = tmp1 -tmp2
                    #print("tmp3: ",tmp3)
                    if tmp3 <= error:
                        m += 1
                    else:
                        break
                index_read_tmp.append(tmp1)
                #print("n: ",n,"m: ",m,"index_read_tmp: ", index_read_tmp)
                n += 1
                m += 1
        #print("n: ",n,"m: ",m)
        if i <= 2:
            index = np.array([index_read_tmp[0:num_peaks]])
        else:
            index = np.concatenate((index, np.array([index_read_tmp[0:num_peaks]])))
    print(i-1, " ", sym_class, " data read in!")
    index_output_filename = work_directory + "\\"+ sym_class + "_index.txt"
    cellPrm_output_filename = work_directory + "\\"+ sym_class + "_cell_prm.txt"
    #print("Details in files: ", cellPrm_output_filename,index_output_filename)
    np.savetxt(index_output_filename, index,fmt="%10.5f")
    np.savetxt(cellPrm_output_filename, cell_prm,fmt="%10.5f")
    return index, cell_prm



if __name__ == "__main__":

    # test the codes
    import matplotlib.pyplot as plt
    V1 = cell_vol(5,5,5,90,90,90)
    d1 = d_spacing(5,5,5,90,90,90,1,1,1)
    print("V1: ",V1," d1: ",d1)
    #X = GaussianProfile(5.0,0.1,0.5,0.001)
    #Y = LorentzProfile(5.0,0.1,0.5,0.001)
    X = tchz_profile_range(15.0,0.00039,-0.00221,-0.00146,0,0.00957,0,0.5,80,2.5)
    Y = tchz_profile_range(25.0,0.00039,-0.00221,-0.00146,0,0.00957,0,0.5,80,2.5)
    Z = tchz_profile_range(45.0,0.00039,-0.00221,-0.00146,0,0.00957,0,0.5,80,2.5)
    P = pv_peak(5,0.02,0.02,0.02,0.02,0.02,0.02,0.5,0.01)
    T = tchz_profile(5.0,0.00039,-0.00221,-0.00146,0,0.00957,0,0.5,0.001)
    print(Z)
    print(X.shape)
    plt.plot(X[:,0],X[:,1],'ro',Y[:,0],Y[:,1],'bs',Z[:,0],Z[:,1],'g^',T[:,0],T[:,1],'o-')
    plt.plot(P[:,0],P[:,1],'o')
    plt.title("Profiles", fontsize=20)
    plt.xlabel('2th. (deg.)',fontsize=16)
    plt.ylabel('Int. (Arb.Unit)',fontsize=16)
    plt.axis([0.5, 80, -0.1, 1.1])
    plt.show()

    #test data read
    work_directory =   r"""E:\CH\2018_NWPU\data\AI_XRD\data"""
    #Refl, cellprm = hkl_read(work_directory, "cubic", 100, 20, 0.05, 0.005, 5)
    #print(Refl)
    #print(cellprm)



