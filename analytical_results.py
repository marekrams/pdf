import numpy as np
from scipy.special import jv

def timelike(func):
    def cleanup(g, t, x):
        mb = g / np.sqrt(np.pi)
        t2mx2 = t**2 - x**2
        t2mx2[t2mx2 <= 1e-11] = 1e-11
        arg = mb * np.sqrt(t2mx2)
        tmp = func(mb, t, x, t2mx2, arg)
        tmp[t2mx2 <= 1e-10] = 0
        return tmp
    return cleanup

@timelike
def fT00(mb, t, x, t2mx2, arg):
    return (mb ** 2) * (np.pi / 2) * (jv(0,  arg) ** 2 + (t**2 + x**2) * jv(1, arg)**2 / t2mx2)

@timelike
def fT11(mb, t, x, t2mx2, arg):
    return (mb ** 2) * (np.pi / 2) * (-jv(0,  arg) ** 2 + (t**2 + x**2) * jv(1, arg)**2 / t2mx2)

@timelike
def fT01(mb, t, x, t2mx2, arg):
    return (mb ** 2) * np.pi * (t * x) * jv(1, arg)**2 / t2mx2

@timelike
def fj0(mb, t, x, t2mx2, arg):
    return -mb * (x / np.sqrt(t2mx2)) * jv(1, arg)

@timelike
def fj1(mb, t, x, t2mx2, arg):
    return - mb * (t / np.sqrt(t2mx2)) * jv(1, arg)

@timelike
def fnu(mb, t, x, t2mx2, arg):
    return mb * np.exp(0.577215665) * (1 - np.cos(2 * np.pi * (1 - jv(0, arg)))) / (2 * np.pi)

@timelike
def fLn(mb, t, x, t2mx2, arg):
    return jv(0, arg)
