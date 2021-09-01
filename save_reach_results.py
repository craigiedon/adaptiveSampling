from typing import Tuple

import probRobScene
import json
import pyrep.objects
from probRobScene.core.experiment_design import try_unif_design
from probRobScene.core.object_types import show_3d
from probRobScene.core.plotUtil3d import draw_cube
from probRobScene.core.regions import AABB
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs
from probRobScene.wrappers.coppelia.setupFuncs import top_of
from pyrep import PyRep
from pyrep.objects import Camera
from pyrep.robots.arms.panda import Panda
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
import matplotlib.pyplot as plt

from stl import *


def get_above_object_path(agent: Panda, target_obj: pyrep.objects.Object, z_offset: float = 0.0,
                          ig_cols: bool = False) -> ArmConfigurationPath:
    pos = top_of(target_obj)
    pos[2] += z_offset

    path = agent.get_path(position=pos, euler=[-np.pi, 0.0, np.pi / 2.0], ignore_collisions=ig_cols)  # , euler=orient)
    # path.visualize()
    return path


pr = PyRep()
pr.launch(headless=False, responsive_ui=True)

scene_view = Camera('DefaultCamera')
scene_view.set_position([-0.43, 3.4, 2.25])
scene_view.set_orientation(np.array([114, 0.0, 0.0]) * np.pi / 180.0)

scenario = probRobScene.scenario_from_file("scenarios/cubeOnTable.prs")
ex_world, used_its = scenario.generate()
c_objs = cop_from_prs(pr, ex_world)

cube = c_objs["CUBOID"][0]
initial_cube_pos = np.array(cube.get_position())
# print(cube)
panda_1, gripper_1 = Panda(0), PandaGripper(0)

initial_arm_config = panda_1.get_configuration_tree()
initial_arm_joint_pos = panda_1.get_joint_positions()
initial_gripper_config = gripper_1.get_configuration_tree()
initial_gripper_joint_pos = gripper_1.get_joint_positions()


def reset_arm():
    pr.set_configuration_tree(initial_arm_config)
    pr.set_configuration_tree(initial_gripper_config)
    panda_1.set_joint_positions(initial_arm_joint_pos, disable_dynamics=True)
    panda_1.set_joint_target_velocities([0] * 7)
    gripper_1.set_joint_positions(initial_gripper_joint_pos, disable_dynamics=True)
    gripper_1.set_joint_target_velocities([0] * 2)


pr.start()
pr.step()

ts = pr.get_simulation_timestep()
print("timestep:", ts)


def sim_fun(cube_pos: np.ndarray) -> List[Tuple]:
    reset_arm()
    cube.set_position(cube_pos)

    max_timesteps = 100
    state_information = []

    try:
        arm_path = get_above_object_path(panda_1, cube, 0.03)
        move_done = False
    except Exception as e:
        print(e)
        move_done = True

    for t in range(max_timesteps):
        # print(t)
        if not move_done:
            move_done = arm_path.step()

        pr.step()

        target_pos = np.array(top_of(cube)) + np.array([0.0, 0.0, 0.03])
        arm_pos = panda_1.get_tip().get_position()
        state_information.append((target_pos.tolist(), arm_pos.tolist()))

    return state_information


# dist_func = lambda state: np.linalg.norm(state[0] - state[1]) / 10.0 Div by 10 to normalize signals for agm robustness
epsilon = 0.05
dist_func = lambda state: np.linalg.norm(np.array(state[0]) - np.array(state[1])) - epsilon
spec = F(LEQ0(dist_func), 0, 99)

scenes = try_unif_design(scenario, 499)
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# for i, scene in enumerate(scenes):
#     for obj in scene.objects:
#         if not hasattr(obj, 'model_name') or obj.model_name != "Table":
#             show_3d(obj, ax)
#
# w_min_corner, w_max_corner = AABB(scenes[0].workspace)
# w_dims = w_max_corner - w_min_corner
#
# draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)
#
# total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)
#
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.tight_layout()
#
# show_3d(scenes[0].objects[0], ax)
#
# plt.show()

cube_positions = [np.array(s.objects[2].position.coordinates) for s in scenes]

pr.start()
simulation_traces = []
for i, c_pos in enumerate(cube_positions):
    print("Sim iteration: ", i)
    simulation_traces.append({"input_state": c_pos.tolist(), "trace": sim_fun(c_pos)})

pr.stop()
pr.shutdown()

# print(simulation_traces)
# with open('simulation_traces/cube_reach_1000.json', 'w', encoding='utf-8') as f:
#     json.dump(simulation_traces, f, ensure_ascii=False, indent=4)


print("Num simulation traces: ", len(simulation_traces))