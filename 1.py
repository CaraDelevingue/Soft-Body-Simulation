import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#参考构型
X = np.array([
    [0.0,0.0,0.0],
    [0.0,0.0,1.0],
    [0.0,1.0,0.0],
    [1.0,0.0,0.0]
])

#参考边矩阵
Dm = np.column_stack((
    X[1]-X[0],
    X[2]-X[0],
    X[3]-X[0]
))

#Dm^-1
Dm_inv = np.linalg.inv(Dm)

#参考体积
V0 = np.abs(np.linalg.det(Dm)) / 6.0

#当前构型
x = X.copy()
v = np.zeros_like(x)

#标准lame参数
#Young modulus
E = 1e4

#Poisson ratio
nu = 0.3

mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu)*(1 - 2 * nu))

#形变梯度F
def compute_F(x, Dm_inv):
    Ds = np.column_stack((
        x[1]-x[0],
        x[2]-x[0],
        x[3]-x[0]
    ))
    return Ds @ Dm_inv

#实现Neo-Hookean应力
def compute_P(F, mu, lam):
    J = np.linalg.det(F)
    #防止J<=0,发生爆炸
    J = max(J, 1e-6)
    FinvT = np.linalg.inv(F).T
    return mu * (F - FinvT) + lam * np.log(J) * FinvT

#计算应力到节点力
def compute_forces(x, Dm_inv, V0, mu, lam):
    F = compute_F(x, Dm_inv)
    P = compute_P(F, mu, lam)

    H = -V0 * P @ Dm_inv.T

    f = np.zeros((4,3))
    f[1] = H[:,0]
    f[2] = H[:,1]
    f[3] = H[:, 2]
    f[0] = -f[1] - f[2] - f[3]

    return f

#重力
#密度=1
mass = V0 / 4.0
gravity = np.array([0, -9.8, 0])

'''
#半隐式欧拉
dt = 0.001

for step in range(1000):
    f_int = compute_forces(x, Dm_inv, V0, mu, lam)

    f_total = f_int + mass * gravity

    a = f_total / mass

    v += dt*a
    x += dt*v
'''

def draw_tetra(ax, x):
    # 四面体的 4 个三角面
    faces = [
        [x[0], x[1], x[2]],
        [x[0], x[1], x[3]],
        [x[0], x[2], x[3]],
        [x[1], x[2], x[3]],
    ]

    poly = Poly3DCollection(faces, alpha=0.5)
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dt = 0.005

for step in range(500):
    ax.clear()

    # --- 物理更新 ---
    f_int = compute_forces(x, Dm_inv, V0, mu, lam)
    f_total = f_int + mass * gravity
    a = f_total / mass

    v[:] += dt * a
    x[:] += dt * v

    x[0] = X[0]
    v[0] = 0

    # --- 可视化 ---
    draw_tetra(ax, x)

    ax.set_xlim(-1, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 2)

    plt.pause(0.01)
