本项目目前仅测试了A1机器狗mujoco仿真模型迁移[deploy/deploy_mujoco/deploy_mujoco_a1.py](https://github.com/Benxiaogu/quadruped_rl/blob/main/deploy/deploy_mujoco/deploy_mujoco_a1.py)和[deploy/deploy_mujoco/deploy_mujoco_a1_hc.py](https://github.com/Benxiaogu/quadruped_rl/blob/main/deploy/deploy_mujoco/deploy_mujoco_a1_hc.py)，其他程序均未测试，本项目是持续更新...

A1 机器狗爬楼梯：

```bash
cd deploy/deploy_mujoco
python deploy_mujoco_a1.py a1.yaml
```

https://www.bilibili.com/video/BV1y3EvzqEtR/?vd_source=9e51c239510424a92903fe86ed4d9e71

A1 机器狗匍匐：

```bash
cd deploy/deploy_mujoco
python deploy_mujoco_a1_hc.py a1_hc.yaml
```

https://www.bilibili.com/video/BV1oEj8zzEqE/?vd_source=9e51c239510424a92903fe86ed4d9e71