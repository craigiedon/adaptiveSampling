import GPy
import probRobScene
import pyrep.objects
from matplotlib import pyplot as plt
from probRobScene.wrappers.coppelia.setupFuncs import top_of
from pyrep import PyRep
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Camera
import numpy as np
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from scipy.optimize import minimize

from mepeTest import mepe_sampling, plot_gp_1d
from stl import *
import sys
from probRobScene.wrappers.coppelia import robotControl as rc
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper


def get_above_object_path(agent: Panda, target_obj: pyrep.objects.Object, z_offset: float = 0.0,
                          ig_cols: bool = False) -> ArmConfigurationPath:
    pos = top_of(target_obj)
    pos[2] += z_offset

    path = agent.get_path(position=pos, euler=[-np.pi, 0.0, np.pi / 2.0], ignore_collisions=ig_cols)  # , euler=orient)
    # path.visualize()
    return path


scenario = probRobScene.scenario_from_file("scenarios/cubeOnTable.prs")
pr = PyRep()
pr.launch(headless=True, responsive_ui=False)

scene_view = Camera('DefaultCamera')
scene_view.set_position([-0.43, 3.4, 2.25])
scene_view.set_orientation(np.array([114, 0.0, 0.0]) * np.pi / 180.0)

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


def sim_fun(cube_x_guess: float, obj_spec: List[STLExp]) -> float:
    reset_arm()
    new_cube_pos = np.array([cube_x_guess, -0.2, initial_cube_pos[2]])
    cube.set_position(new_cube_pos)

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
        state_information.append((target_pos, arm_pos))

    score = agm_rob(obj_spec[0], state_information, 0)
    print("Cube offset: ", cube_x_guess, "score:", score)
    return score


# dist_func = lambda state: np.linalg.norm(state[0] - state[1]) / 10.0 Div by 10 to normalize signals for agm robustness
epsilon = 0.05
dist_func = lambda state: np.linalg.norm(state[0] - state[1]) - epsilon
spec = F(LEQ0(dist_func), 0, 99)
# # minimization loop
pr.start()

domain = (-0.85, 0.85)
budget = 20
initial_points = np.linspace(domain[0], domain[1], num=3).reshape(-1, 1)
candidate_points = np.linspace(domain[0], domain[1], num=100).reshape(-1, 1)
gp = mepe_sampling(budget=budget, initial_points=initial_points, candidate_points=candidate_points, sim_func=lambda x: sim_fun(x, [spec]), plot_intermediates=False)

fig, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2)
plot_gp_1d(ax_0, gp, candidate_points)

rand_X = np.random.uniform(domain[0], domain[1], size=(budget, 1))
plt.show()

pr.stop()
pr.shutdown()
