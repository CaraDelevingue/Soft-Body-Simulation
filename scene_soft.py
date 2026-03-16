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
        #二维网格
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
        #弹簧连接
        #----------
        self.springs = []

        #每个弹簧记录（i， j， rest_length）
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j

                #右边连接
                if j < self.grid_size -1:
                    right_idx = i * self.grid_size + (j + 1)
                    rest = np.linalg.norm(self.position[idx] - self.position[right_idx])
                    self.springs.append((idx, right_idx, rest))

                #上边连接
                if i < self.grid_size -1:
                    up_idx = (i + 1) * self.grid_size + j
                    rest = np.linalg.norm(self.position[idx] - self.position[up_idx])
                    self.springs.append((idx, up_idx, rest))
                
                # 右上对角
                if i < self.grid_size - 1 and j < self.grid_size - 1:
                    diag1 = (i + 1) * self.grid_size + (j + 1)
                    rest = np.linalg.norm(self.position[idx] - self.position[diag1])
                    self.springs.append((idx, diag1, rest))

                # 左上对角
                if i < self.grid_size - 1 and j > 0:
                    diag2 = (i + 1) * self.grid_size + (j - 1)
                    rest = np.linalg.norm(self.position[idx] - self.position[diag2])
                    self.springs.append((idx, diag2, rest))
        
        #结构力刚度
        self.spring_k = 3000.0

        #结构力阻尼系数
        self.spring_damping = 2.0


    def simulate(self):
        for _ in range(self.sim_substeps):

            #----------
            #弹簧力(基于当前position)
            #----------
            spring_forces = np.zeros((self.num_particles, 3))

            for (i, j, rest_length) in self.springs:
                x_i = self.position[i]
                x_j = self.position[j]

                v_i = self.velocity[i]
                v_j = self.velocity[j]

                delta = x_j - x_i
                length = np.linalg.norm(delta)

                if length == 0:
                    continue

                direction = delta / length

                #----------
                #弹簧弹性力
                #----------
                #Hooke 定律
                force_magnitude = self.spring_k * (length - rest_length)
                elasttic_force = force_magnitude * direction

                #----------
                #弹簧阻尼力
                #----------
                relative_velocity = v_j - v_i
                damping_magnitude = - self.spring_damping * np.dot(relative_velocity,direction)
                damping_force = damping_magnitude *direction

                #总弹簧力
                force = elasttic_force + damping_force

                spring_forces[i] += force
                spring_forces[j] -= force

            #对每个粒子计算
            for i in range(self.num_particles):
                #计算重力 + 物体内力
                force = self.mass[i] * self.gravity + spring_forces[i]

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