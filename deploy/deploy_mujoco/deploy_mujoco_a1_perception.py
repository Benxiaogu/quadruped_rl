import time

import mujoco.viewer
import mujoco
from mujoco import mjtTrn, mjtObj
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import time

LEGGED_GYM_ROOT_DIR = '/home/dawn/real/unitree_rl_gym'

import numpy as np
import mujoco

class StairHeightQuery:
    def __init__(self, model, hfield_name="stair_hfield"):
        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        self.size = model.hfield_size[self.hfield_id]
        self.nrow = model.hfield_nrow[self.hfield_id]
        self.ncol = model.hfield_ncol[self.hfield_id]
        
        # 计算实际尺寸
        self.x_total = self.size[0] * 2  # X方向总长
        self.y_total = self.size[1] * 2  # Y方向总长
        
    def world_to_hfield(self, x, y):
        """世界坐标转高度场坐标"""
        # 转换为左下角原点坐标
        hx = (x + self.size[0]) / self.x_total
        hy = (y + self.size[1]) / self.y_total
        
        # 转换为网格索引
        col = int(hx * (self.ncol - 1))
        row = int(hy * (self.nrow - 1))
        
        return col, row
    
    def get_height(self, x, y):
        """获取指定位置高度（带边界检查）"""
        col = int((x + self.size[0]) / self.x_total * (self.ncol - 1))
        col = np.clip(col, 0, self.ncol-2)
        
        # 所有行高度相同（适用于直楼梯）
        return model.hfield_data[col] * self.size[2]  # 乘以最大高度

    def get_height_batch(self, points):
        """批量查询高度"""
        # points: (N, 2)数组，包含[x,y]坐标
        heights = []
        for x, y in points:
            col = int((x + self.size[0]) / self.x_total * (self.ncol - 1))
            col = np.clip(col, 0, self.ncol-2)
            heights.append(model.hfield_data[col] * self.size[2])
        return np.array(heights)

def create_stair_hfield(ncol=256, nrow=64, 
                       step_height=0.1, 
                       step_length=0.3,
                       total_steps=10):
    """生成楼梯高度场数据"""
    h_data = np.zeros(nrow * ncol, dtype=np.float32)
    
    # 每个台阶占据的列数
    cols_per_step = int(step_length * ncol / 3.2)  # 3.2m为总宽度
    
    for step in range(total_steps):
        start_col = step * cols_per_step
        end_col = (step + 1) * cols_per_step
        height = (step + 1) * step_height
        
        # 设置当前台阶区域高度
        h_data[start_col:end_col] = height
        
    return h_data.reshape(nrow, ncol)

# 初始化模型
model = mujoco.MjModel.from_xml_path("stair_terrain.xml")
hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "stair_hfield")

# 生成并填充高度数据
stair_data = create_stair_hfield()
model.hfield_data[:] = stair_data.flatten()  # 必须展平为一维数组

# 初始化数据
data = mujoco.MjData(model)

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def setup_height_detection(m, d):
    # 初始化射线检测器
    exclude_geoms = ["robot_body"]  # 需要排除检测的几何体名称
    exclude_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) 
                  for name in exclude_geoms]
    return exclude_ids

def raycast(m, d, xy_point, max_height=1.0):
    """ 返回指定(x,y)坐标点的地面高度
    参数:
        xy_point: [x, y] 坐标
        max_height: 最大检测高度（建议设为机器人跳跃高度）
    """
    # 设置射线参数
    pnt = np.array([xy_point[0], xy_point[1], max_height])  # 起始点（高空）
    vec = np.array([0, 0, -max_height*2])                   # 方向向量（垂直向下）
    
    # 排除机器人自身的碰撞检测
    exclude_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) 
                  for name in ["robot_geom1", "robot_geom2"]]
    
    # 执行射线检测
    result = mujoco.Ray(m, d, pnt, vec, 
                       group=0,  # 碰撞检测组
                       flg_static=1, 
                       excludegeom=exclude_ids)
    
    return result.hit[2] if result.hit is not None else 0.0  # 未命中时返回0

# # 使用示例
# point_xy = [1.5, 0.3]  # 要检测的x,y坐标
# height = raycast(d, point_xy)
# print(f"坐标({point_xy[0]}, {point_xy[1]})的地面高度: {height:.3f} 米")

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_obs_frame = config["num_obs_frame"]

        hip_reduction = config["hip_reduction"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # observations
    obs_buff = torch.zeros(1, num_obs*num_obs_frame, dtype=torch.float)
    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 1mx1.6m rectangle (without center line)
    measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)

    # 初始化高度查询器
    height_query = StairHeightQuery(m)

    # # 遍历所有执行器
    # print("执行器顺序及其控制的关节：")
    # for actuator_id in range(m.nu):
    #     # 获取执行器名称
    #     actuator_name = mujoco.mj_id2name(m, mjtObj.mjOBJ_ACTUATOR, actuator_id)
        
    #     # 获取执行器类型和关联的关节ID
    #     trn_type = m.actuator_trntype[actuator_id]
    #     trn_id = m.actuator_trnid[actuator_id, 0]  # 假设第一个目标为关节
        
    #     # 检查是否为关节型执行器
    #     if trn_type == mjtTrn.mjTRN_JOINT:
    #         joint_name = mujoco.mj_id2name(m, mjtObj.mjOBJ_JOINT, trn_id)
    #     else:
    #         joint_name = "N/A (非关节执行器)"
        
    #     # 输出信息
    #     print(f"执行器索引 {actuator_id} - 名称: {actuator_name}")
    #     print(f"    控制关节: {joint_name} (关节索引: {trn_id})")
    #     print("-----------------------------------------")

    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        exclude_ids = setup_height_detection(m,d)

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            robot_x = data.qpos[0]

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = cmd * cmd_scale
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action

                obs_buff[:,: num_obs * (num_obs_frame - 1)] = obs_buff[:,num_obs : num_obs * num_obs_frame].clone()
                obs_buff[:, -num_obs:] = torch.from_numpy(obs)

                obs_tensor = obs_buff.clone()

                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                action[[0, 3, 6, 9]] *= hip_reduction
                target_action = action * action_scale + default_angles

                target_dof_pos = target_action[[3,4,5,0,1,2,9,10,11,6,7,8]]


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
