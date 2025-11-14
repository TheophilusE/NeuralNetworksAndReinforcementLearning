import pybullet as p
import pybullet_data
import numpy as np
import time


class PyBulletPendulum:
    """Create a simple single- or double- pendulum in PyBullet.

    The implementation uses `createMultiBody` to assemble links with revolute
    joints and exposes a compatible interface with the simple simulator.
    """

    def __init__(self, mode="single", dt=0.02, gui=False):
        self.mode = mode
        self.dt = dt
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        # simple plane
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self._build_pendulum()

    def _build_pendulum(self):
        # base is fixed at world origin
        if self.mode == "single":
            mass = 1.0
            length = 1.0
            # Use a box with halfExtents instead of a capsule to avoid
            # pybullet versions that don't accept a `height`/`length` kwarg
            half_ext = [0.05, 0.05, length / 2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
            linkMasses = [mass]
            linkCollisionShapeIndices = [col]
            linkVisualShapeIndices = [vis]
            linkPositions = [[0, 0, -length / 2]]
            linkOrientations = [[0, 0, 0, 1]]
            linkInertialFramePositions = [[0, 0, 0]]
            linkInertialFrameOrientations = [[0, 0, 0, 1]]
            # first link should be parented to the base (index 0)
            linkParentIndices = [0]
            linkJointTypes = [p.JOINT_REVOLUTE]
            linkJointAxis = [[1, 0, 0]]

            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
                                          basePosition=[0, 0, 1.5],
                                          linkMasses=linkMasses,
                                          linkCollisionShapeIndices=linkCollisionShapeIndices,
                                          linkVisualShapeIndices=linkVisualShapeIndices,
                                          linkPositions=linkPositions,
                                          linkOrientations=linkOrientations,
                                          linkInertialFramePositions=linkInertialFramePositions,
                                          linkInertialFrameOrientations=linkInertialFrameOrientations,
                                          linkParentIndices=linkParentIndices,
                                          linkJointTypes=linkJointTypes,
                                          linkJointAxis=linkJointAxis,
                                          physicsClientId=self.client)
            # record joint count and disable default motor control if joints created
            self.n_joints = p.getNumJoints(self.body, physicsClientId=self.client)
            if self.n_joints > 0:
                p.setJointMotorControlArray(self.body, list(range(self.n_joints)), p.VELOCITY_CONTROL, forces=[0] * self.n_joints, physicsClientId=self.client)
            else:
                print(f"Warning: created body {self.body} has no joints (n_joints=0)")
        else:
            # double pendulum: two links
            mass = 1.0
            length = 1.0
            # Replace capsule shapes with narrow boxes for compatibility
            half_ext1 = [0.04, 0.04, length / 2]
            half_ext2 = [0.03, 0.03, length / 2]
            col1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext1, physicsClientId=self.client)
            vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext1, rgbaColor=[0, 0, 1, 1], physicsClientId=self.client)
            col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext2, physicsClientId=self.client)
            vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext2, rgbaColor=[0, 1, 0, 1], physicsClientId=self.client)

            linkMasses = [mass, mass]
            linkCollisionShapeIndices = [col1, col2]
            linkVisualShapeIndices = [vis1, vis2]
            linkPositions = [[0, 0, -length / 2], [0, 0, -length]]
            linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
            linkInertialFramePositions = [[0, 0, 0], [0, 0, 0]]
            linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
            # parent indices for two-link chain: first link -> base (0), second -> first (1)
            linkParentIndices = [0, 1]
            linkJointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
            linkJointAxis = [[1, 0, 0], [1, 0, 0]]

            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
                                          basePosition=[0, 0, 1.5],
                                          linkMasses=linkMasses,
                                          linkCollisionShapeIndices=linkCollisionShapeIndices,
                                          linkVisualShapeIndices=linkVisualShapeIndices,
                                          linkPositions=linkPositions,
                                          linkOrientations=linkOrientations,
                                          linkInertialFramePositions=linkInertialFramePositions,
                                          linkInertialFrameOrientations=linkInertialFrameOrientations,
                                          linkParentIndices=linkParentIndices,
                                          linkJointTypes=linkJointTypes,
                                          linkJointAxis=linkJointAxis,
                                          physicsClientId=self.client)
            # record joint count and disable default motor control if joints created
            self.n_joints = p.getNumJoints(self.body, physicsClientId=self.client)
            if self.n_joints > 0:
                p.setJointMotorControlArray(self.body, list(range(self.n_joints)), p.VELOCITY_CONTROL, forces=[0] * self.n_joints, physicsClientId=self.client)
            else:
                print(f"Warning: created body {self.body} has no joints (n_joints=0)")

    def get_state(self):
        if self.mode == "single":
            if getattr(self, 'n_joints', 0) > 0:
                js = p.getJointState(self.body, 0, physicsClientId=self.client)
                angle = js[0]
                vel = js[1]
                pos = p.getLinkState(self.body, 0, physicsClientId=self.client)[0]
                # return `pos1` to be consistent with double-pendulum keys
                return {"theta": angle, "theta_dot": vel, "pos1": pos}
            else:
                # Fallback: no joints were created; return base position/zeroed angles
                base_pos, _ = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
                return {"theta": 0.0, "theta_dot": 0.0, "pos1": base_pos}
        else:
            if getattr(self, 'n_joints', 0) >= 2:
                js0 = p.getJointState(self.body, 0, physicsClientId=self.client)
                js1 = p.getJointState(self.body, 1, physicsClientId=self.client)
                pos0 = p.getLinkState(self.body, 0, physicsClientId=self.client)[0]
                pos1 = p.getLinkState(self.body, 1, physicsClientId=self.client)[0]
                return {"th1": js0[0], "th2": js1[0], "w1": js0[1], "w2": js1[1], "pos1": pos0, "pos2": pos1}
            elif getattr(self, 'n_joints', 0) == 1:
                js0 = p.getJointState(self.body, 0, physicsClientId=self.client)
                pos0 = p.getLinkState(self.body, 0, physicsClientId=self.client)[0]
                base_pos, _ = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
                return {"th1": js0[0], "th2": 0.0, "w1": js0[1], "w2": 0.0, "pos1": pos0, "pos2": base_pos}
            else:
                base_pos, _ = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
                return {"th1": 0.0, "th2": 0.0, "w1": 0.0, "w2": 0.0, "pos1": base_pos, "pos2": base_pos}

    def step(self, torque=0.0):
        # apply torque to first joint only
        if getattr(self, 'n_joints', 0) > 0:
            # apply torque to first joint only
            p.setJointMotorControl2(self.body, 0, p.TORQUE_CONTROL, force=torque, physicsClientId=self.client)
        else:
            # no joint present: apply an external torque to the base as a fallback
            # (linkIndex=-1 applies to base). This avoids exceptions but has different dynamics.
            try:
                p.applyExternalTorque(self.body, -1, [torque, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.client)
            except Exception:
                # If applyExternalTorque isn't available or fails, silently continue
                pass
        p.stepSimulation(physicsClientId=self.client)
        if self.gui:
            time.sleep(self.dt)

    def close(self):
        p.disconnect(physicsClientId=self.client)
