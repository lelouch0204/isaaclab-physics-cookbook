# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Bouncing Balls — Restitution Comparison

Six spheres dropped from the same height, each with a different restitution
coefficient (0.0 to 1.0). Watch how bounce height increases with restitution.

.. code-block:: bash

    isaaclab -p examples/bouncing_balls.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bouncing balls with different restitution values.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

# --- Configuration ---
RESTITUTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COLORS = [
    (0.85, 0.10, 0.10),  # red        — no bounce
    (0.90, 0.45, 0.05),  # orange     — very low
    (0.90, 0.80, 0.10),  # yellow     — low
    (0.20, 0.75, 0.20),  # green      — medium
    (0.15, 0.50, 0.85),  # blue       — high
    (0.60, 0.20, 0.85),  # purple     — perfect bounce
]
DROP_HEIGHT = 2.5
SPHERE_RADIUS = 0.1
SPACING = 0.6  # distance between ball centres along X


def design_scene() -> dict[str, RigidObject]:
    """Create ground, lights, and six bouncing spheres."""

    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Make the ground perfectly elastic so the ball restitution dominates
    ground_mat = sim_utils.RigidBodyMaterialCfg(
        restitution=1.0,
        restitution_combine_mode="max",
    )
    ground_mat.func("/World/defaultGroundPlane/GroundMaterial", ground_mat)

    # Dome light
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Spheres
    spheres: dict[str, RigidObject] = {}
    for i, (rest, color) in enumerate(zip(RESTITUTIONS, COLORS)):
        x = (i - (len(RESTITUTIONS) - 1) / 2) * SPACING
        xform_path = f"/World/BallPos{i}"
        sim_utils.create_prim(xform_path, "Xform", translation=[x, 0.0, 0.0])

        ball_cfg = RigidObjectCfg(
            prim_path=f"{xform_path}/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=SPHERE_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(restitution=rest),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, metallic=0.2, roughness=0.4
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DROP_HEIGHT)),
        )
        spheres[f"ball_{i}"] = RigidObject(cfg=ball_cfg)

    return spheres


def run_simulator(sim: SimulationContext, spheres: dict[str, RigidObject]):
    """Step the simulation, resetting every 500 steps."""
    sim_dt = sim.get_physics_dt()
    step = 0
    ball_list = [spheres[f"ball_{i}"] for i in range(len(RESTITUTIONS))]

    while simulation_app.is_running():
        # Periodic reset
        if step % 500 == 0:
            step = 0
            for ball in ball_list:
                root_state = ball.data.default_root_state.clone()
                ball.write_root_pose_to_sim(root_state[:, :7])
                ball.write_root_velocity_to_sim(root_state[:, 7:])
                ball.reset()
            print("----------------------------------------------")
            print("[INFO] Resetting balls")
            labels = " | ".join(f"r={r}" for r in RESTITUTIONS)
            print(f"  Restitution: {labels}")
            print(f"  Drop height: {DROP_HEIGHT} m")

        # Physics step
        for ball in ball_list:
            ball.write_data_to_sim()
        sim.step()
        step += 1
        for ball in ball_list:
            ball.update(sim_dt)

        # Print heights every 50 steps
        if step % 50 == 0:
            heights = [ball.data.root_pos_w[:, 2].item() for ball in ball_list]
            parts = [f"r={r:.1f}: {h:.3f}m" for r, h in zip(RESTITUTIONS, heights)]
            print(f"[step {step:4d}]  " + "  ".join(parts))


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -3.5, 2.5], target=[0.0, 0.0, 0.8])

    spheres = design_scene()
    sim.reset()
    print("[INFO] Setup complete — dropping balls...")
    run_simulator(sim, spheres)


if __name__ == "__main__":
    main()
    simulation_app.close()
