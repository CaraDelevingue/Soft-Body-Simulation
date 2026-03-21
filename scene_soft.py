import warp as wp
import warp.sim.render
import numpy as np
import matplotlib.pyplot as plt

import my_sim.my_integrator
import my_sim.my_modelbuilder

wp.init()

class Example:

    def __init__(self):
        fps = 60
        self.frame_dt = 1.0/fps

        #每帧拆分为32子步
        self.sim_substeps = 32

        #每个子步时间
        self.sim_dt = self.frame_dt / self.sim_substeps

        #当前时间
        self.sim_time = 0.0

        #----------
        #三角形网格
        #----------

        #5*5网格
        self.grid_size = 5

        #粒子数量
        self.num_particles = self.grid_size * self.grid_size

        #粒子质量
        self.mass = np.ones(self.num_particles)

        #初始速度
        self.velocity = np.zeros((self.num_particles,3))

        #粒子间隔
        spacing = 0.5

        #FEM材料参数
        self.young = 3000.0
        self.poisson = 0.3

        #由杨氏模量和泊松比换算Lame参数
        self.mu = self.young / (2.0 * (1.0 + self.poisson))
        self.lam = self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

        #初始化粒子位置
        self.position = np.zeros((self.num_particles,3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                self.position[idx] = np.array([
                    j * spacing, 
                    5.0 + i* spacing, 
                    0.0
                    ])

        #重力
        self.gravity =  np.array([0.0, -9.8, 0.0])

        #----------
        #结构力连接
        #----------

        self.elements = []

        #每个弹簧记录（i， j， rest_length）
        for i in range(self.grid_size-1):
            for j in range(self.grid_size-1):
                idx0 = i * self.grid_size + j
                idx1 = i * self.grid_size + (j + 1)
                idx2 = (i + 1) * self.grid_size + j
                idx3 = (i + 1) * self.grid_size + (j + 1)

                #连接三角形
                self.elements.append((idx0, idx1, idx3))
                self.elements.append((idx0, idx3, idx2))
        
        #初始形状的逆矩阵 Dm_inv
        self.rest_matrices = []
        self.areas = []

        for (i0, i1, i2) in self.elements:
            x0 = self.position[i0][:2]
            x1 = self.position[i1][:2]
            x2 = self.position[i2][:2]

            Dm = np.column_stack((x1 - x0, x2 - x0))
            Dm_inv = np.linalg.inv(Dm)

            area = 0.5 * abs(np.linalg.det(Dm))

            self.rest_matrices.append(Dm_inv)
            self.areas.append(area)

        #顶部固定粒子
        self.fixed_indices = []
        for j in range(self.grid_size):
            idx = (self.grid_size - 1) * self.grid_size + j
            self.fixed_indices.append(idx)

        # 记录固定点初始位置
        self.fixed_positions = self.position[self.fixed_indices].copy()


    def simulate(self):
        for _ in range(self.sim_substeps):

            #-----------
            #计算FEM内力
            #-----------

            #每个粒子内力初始化为0
            internal_forces = np.zeros_like(self.position)

            for e,(i0, i1, i2) in enumerate(self.elements):
                x0 = self.position[i0][:2]
                x1 = self.position[i1][:2]
                x2 = self.position[i2][:2]

                #当前构型矩阵
                De = np.column_stack((x1 - x0, x2 - x0))

                #参考构型逆矩阵
                Dm_inv = self.rest_matrices[e]

                #deformation gradient
                F = De @ Dm_inv

                #Green strain
                I = np.eye(2)
                E = 0.5 * (F.T @ F - I)#形变

                #StVK 一阶Piola应力
                P = F @ (2.0 * self.mu * E + self.lam * np.trace(E) * I)

                #三角形面积
                area = self.areas[e]

                #单元力矩阵 H = -A * P * Dm_inv^T
                H = -area * P @ Dm_inv.T

                #分配到三个顶点
                f1 = H[:, 0]
                f2 = H[:, 1]
                f0 = -(f1 + f2)

                internal_forces[i0, 0:2] += f0
                internal_forces[i1, 0:2] += f1
                internal_forces[i2, 0:2] += f2

            #对每个粒子计算
            for i in range(self.num_particles):
                #计算重力 + 物体内力
                force = self.mass[i] * self.gravity 
                force += internal_forces[i]

                #地面接触检测
                if self.position[i, 1] < 0.0:
                    penetration = -self.position[i, 1]

                    #地面刚度
                    k = 10000.0
                    #阻尼系数
                    c = 50.0

                    #地面弹簧力
                    normal_force = k* penetration

                    #法向速度
                    v_normal = self.velocity[i, 1]

                    #阻尼力
                    damping_force = -c * v_normal

                    #计算合力
                    force[1] += normal_force + damping_force    

                #牛顿第二定律
                acceleration = force / self.mass[i]

                #半隐式欧拉积分
                self.velocity[i] += acceleration * self.sim_dt
                self.position[i] += self.velocity[i] * self.sim_dt

                # 恢复固定点的位置和速度
                for k, idx in enumerate(self.fixed_indices):
                    self.position[idx] = self.fixed_positions[k]
                    self.velocity[idx] = np.array([0.0, 0.0, 0.0])

            #时间前进一个子步时间
            self.sim_time += self.sim_dt
    
    def step(self):
        self.simulate()

    def render(self):
        plt.clf()

        # 画三角形边
        for (i0, i1, i2) in self.elements:
            p0 = self.position[i0]
            p1 = self.position[i1]
            p2 = self.position[i2]

            xs = [p0[0], p1[0], p2[0], p0[0]]
            ys = [p0[1], p1[1], p2[1], p0[1]]

            plt.plot(xs, ys)

        # 画粒子点
        plt.scatter(self.position[:, 0], self.position[:, 1], s=20)

        # 画地面
        plt.axhline(y=0.0)

        plt.xlim(-1, 4)
        plt.ylim(-1, 8)
        plt.gca().set_aspect('equal')
        plt.title(f"time = {self.sim_time:.3f}")
        plt.pause(0.01)


if __name__ == "__main__":
    example = Example()

    for i in range(200):
        example.step()
        if i % 2 == 0:
            example.render()

    plt.ioff()
    plt.show()