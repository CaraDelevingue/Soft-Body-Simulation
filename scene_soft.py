import warp as wp
import warp.sim.render
import numpy as np

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
        # 多粒子系统
        #----------

        #粒子数量
        self.num_particles = 3

        #粒子质量
        self.mass = np.ones(self.num_particles)

        #初始速度
        self.velocity = np.zeros((self.num_particles,3))

        #初始位置
        self.position = np.zeros((self.num_particles,3))
        for i in range(self.num_particles):
            self.position[i] = np.array([0.0, 5.0 + i, 0.0])

        #重力
        self.gravity =  np.array([0.0, -9.8, 0.0])

        #----------
        #弹簧连接
        #----------
        self.springs = []

        #每个弹簧记录（i， j， rest_length）
        for i in range(self.num_particles -1):
            p_i = self.position[i]
            p_j = self.position[i+1]

            rest_length = np.linalg.norm(p_j - p_i)

            self.springs.append((i, i + 1, rest_length))
        
        #结构力刚度
        self.spring_k = 5000.0


    def simulate(self):
        for _ in range(self.sim_substeps):

            #----------
            #弹簧力(基于当前position)
            #----------
            spring_forces = np.zeros((self.num_particles, 3))

            for (i, j, rest_length) in self.springs:
                x_i = self.position[i]
                x_j = self.position[j]

                delta = x_j - x_i
                length = np.linalg.norm(delta)

                if length == 0:
                    continue

                direction = delta / length

                #Hooke 定律
                force_magnitude = self.spring_k * (length - rest_length)

                force = force_magnitude * direction

                spring_forces[i] += force
                spring_forces[j] -= force

            #对每个粒子计算
            for i in range(self.num_particles):
                #计算重力
                force = self.mass[i] * self.gravity

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

            #时间前进一个子步时间
            self.sim_time += self.sim_dt
    
    def step(self):
        self.simulate()


if __name__ == "__main__":
    example = Example()

    for i in range(200):
        example.step()
        print("frame", i,"position = ")
        print(example.position)