from periodictable import *
import numpy as np
import re

# print("Cd density ",f'{periodictable.Cd.density:.2f}',periodictable.Cd.density_units)
# Cd density 8.65 g/cm^3
# for el in elements:
#    print(el.symbol,el.name,el.mass)


def formula_analyze(form=""):
    formula_anl = formula(form)
    fmula = formula_anl.atoms
    kys = list(fmula.keys())
    print(formula_anl)
    for key in kys:
        print(f'{str(key):3}', f'{fmula[key]:5.2f}', f'{formula_anl.mass_fraction[key]:4.3f}')
    return formula_anl


def formula_analyze_norm(form=""):
    formula_anl = formula(form)
    fmula = formula_anl.atoms
    kys = list(fmula.keys())
    # print(formula_anl)
    sum_atm = 0
    for key in kys:
        sum_atm += fmula[key]
    fmula_norm = ''
    for key in kys:
        atm_tmp = fmula[key] / sum_atm
        fmula_norm += str(key) + f'{atm_tmp:.3f}'
    formula_anl_norm = formula(fmula_norm)
    return formula_anl_norm


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


X = elements.symbol('Fe')
print(X.name, X.number, X.symbol)
print("Cd density ", f'{elements.Cd.density:.3f}', elements.Cd.density_units)
fmula = 'Si(CCl4)4'
cmp = formula_analyze(fmula)
cmp1 = formula_analyze_norm(fmula)
# max1 = max(cmp.atoms[])
# print(cmp.atoms)
print('{:.3f}'.format(cmp.mass))
print(cmp1)
print('{:.3f}'.format(cmp1.mass))
Z_number = cmp.mass / cmp1.mass
print('{:.2f}'.format(Z_number))
print(cmp1.mass_fraction)

# [A-Z][a-z]*\\d*|\\([^)]+\\)\\d
# fract = Regex("(0|[1-9][0-9]*|)([.][0-9]*)")

# whole = Regex("[1-9][0-9]*")
# Formula analysis
fmla = '0.86(Mg0.4Zn0.6)2SiO4-0.14CaTiO3'  # 'LixZn2-xVxSi1-xO4 (x=0.8)'
# '(Mg0.93Zn0.07)2SnO4' #'10.5CaO-22.2B2O3-67.3SiO2' #'MgO-B2O3-SiO2+10 wt% TiO2' #'AlPO4+5 wt% MgF2'
fmla = fmla.replace('wt%', '%wt')
# fmla = fmla.replace('-',' ')
fmla_split = fmla.split(sep='-')
print('fmla_split:', fmla_split)
# pattern = re.compile('')
formula_rewrited = ''
for fmla_tmp in fmla_split:
    atom_n = re.findall('[A-Z][a-z]?|[0-9]+[.]?[0-9]*', fmla_tmp)
    if is_number(atom_n[0]):
        atom_n_all = ''
        for n in range(1, len(atom_n)):
            atom_n_all = atom_n_all + atom_n[n]
        atom_n_all = '(' + atom_n_all + ')' + atom_n[0]
    else:
        atom_n_all = fmla
    # print('atom_n',atom_n,atom_n_all)
    formula_rewrited = formula_rewrited + ' ' + atom_n_all
print(formula_rewrited)
# condition = re.compile('')
# fmla_split = fmla.split(sep='+')
# print(fmla_split)
# Begin with number
# fmla2 = str(fmla_split[1]) + r'//' + str(fmla_split[0])
# print(fmla2)
fmla3 = formula(formula_rewrited)
fmla3 = formula_analyze_norm(formula_rewrited)
print(fmla3, fmla3.atoms)

'''
Y = elements[92]
print(Y, Y.name, Y.mass, Y.number, Y.isotopes)
Y = elements.list('number','symbol', 'mass', 'density', format="%d %-2s: %6.2f u %5.2f g/cm^3")
print(Y)

pattern = re.compile(r"""
              (?P<element>[A-Z][a-z]?)
             |(?P<number>\d+)
             |(?P<bracket>[](){}[])
             |(?P<other>.)
             """, re.VERBOSE)

'''
