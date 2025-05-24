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

            # 获取机器人基准高度
            robot_base_z = d.qpos[2]
            # 获取前方1米处地面高度
            lookahead_point = [robot_base_z[0]+1.0, robot_base_z[1]]
            terrain_h = raycast(m, d, lookahead_point, exclude_ids)
            
            # 计算高度差
            height_diff = robot_base_z[2] - terrain_h

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
