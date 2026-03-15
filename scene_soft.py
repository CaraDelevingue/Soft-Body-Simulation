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
        self.sim_substeps = 1

        #每个子步时间
        self.sim_dt = self.frame_dt / self.sim_substeps

        #当前时间
        self.sim_time = 0.0

        #粒子状态
        #粒子质量
        self.mass = 1.0

        #初始位置
        self.position = np.array([0.0, 5.0, 0.0])

        #初始速度
        self.velocity = np.array([0.0, 0.0, 0.0])

        #重力
        self.gravity =  np.array([0.0, -9.8, 0.0])

    def simulate(self):
        for _ in range(self.sim_substeps):

            #计算重力
            force = self.mass * self.gravity

            #牛顿第二定律
            acceleration = force / self.mass

            #半隐式欧拉积分
            self.velocity += acceleration * self.sim_dt
            self.position += self.velocity * self.sim_dt

            #时间前进一个子步时间
            self.sim_time += self.sim_dt
    
    def step(self):
        self.simulate()


if __name__ == "__main__":
    example = Example()

    for i in range(10):
        example.step()
        print("frame", i,"position = ", example.position,"time = ", example.sim_time)
