import os
import numpy as np

def calc_mi(c_11, c_10, c_01, c_00):
    total = c_11 + c_10 + c_01 + c_00
    p11, p10 = c_11/total, c_10/total
    p01, p00 = c_01/total, c_00/total
    
    px1, px0 = p11 + p01, p10 + p00
    py1, py0 = p11 + p10, p01 + p00
    
    mi = 0
    if p11 > 0: mi += p11 * np.log2(p11 / (px1 * py1))
    if p10 > 0: mi += p10 * np.log2(p10 / (px0 * py1))
    if p01 > 0: mi += p01 * np.log2(p01 / (px1 * py0))
    if p00 > 0: mi += p00 * np.log2(p00 / (px0 * py0))
    return mi

print(f"MI for 'prize': {calc_mi(145, 35, 1540, 16740)}")
print(f"MI for 'hello': {calc_mi(190, 55, 12200, 6740)}")