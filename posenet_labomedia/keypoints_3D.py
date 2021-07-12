
"""
"""

import numpy as np

# moyenne de tous les points de la frame
average = 6.5

# valeurs le la frame précédente
p_prev = [22, 6, 7, 10, 8, 6, 6.1, 5.9, 7, 10, 9, 11, 21, 20]

# valeurs le la frame actuelle
points = [21, 5, 8, 9,  8, 7, 6,   5,   7, 8,  8,  9, 20, 21]
pts = np.asarray(points)

# Suppression du plus petit et du plus grand
pts = np.sort(pts)
print(pts)
pts = pts[1:-1]
print(pts)

# [5, 6, 7, 7, 8, 8, 8, 8, 9, 9, 20, 21]
average = np.average(pts)
print(average)  # 9.6

# Exclusion des trop éloignés
g = pts[ (pts >= average*0.8) & (pts <= average*1.2) ]
print(g)
# [8 8 8 8 9 9]
