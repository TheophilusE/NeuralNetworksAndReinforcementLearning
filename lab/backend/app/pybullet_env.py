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
            col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.05, height=length, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.05, length=length, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
            linkMasses = [mass]
            linkCollisionShapeIndices = [col]
            linkVisualShapeIndices = [vis]
            linkPositions = [[0, 0, -length / 2]]
            linkOrientations = [[0, 0, 0, 1]]
            linkInertialFramePositions = [[0, 0, 0]]
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
                                          linkParentIndices=linkParentIndices,
                                          linkJointTypes=linkJointTypes,
                                          linkJointAxis=linkJointAxis,
                                          physicsClientId=self.client)
            # disable default motor control
            p.setJointMotorControlArray(self.body, [0], p.VELOCITY_CONTROL, forces=[0], physicsClientId=self.client)
        else:
            # double pendulum: two links
            mass = 1.0
            length = 1.0
            col1 = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.04, height=length, physicsClientId=self.client)
            vis1 = p.createVisualShape(p.GEOM_CAPSULE, radius=0.04, length=length, rgbaColor=[0, 0, 1, 1], physicsClientId=self.client)
            col2 = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.03, height=length, physicsClientId=self.client)
            vis2 = p.createVisualShape(p.GEOM_CAPSULE, radius=0.03, length=length, rgbaColor=[0, 1, 0, 1], physicsClientId=self.client)

            linkMasses = [mass, mass]
            linkCollisionShapeIndices = [col1, col2]
            linkVisualShapeIndices = [vis1, vis2]
            linkPositions = [[0, 0, -length / 2], [0, 0, -length]]
            linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
            linkInertialFramePositions = [[0, 0, 0], [0, 0, 0]]
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
                                          linkParentIndices=linkParentIndices,
                                          linkJointTypes=linkJointTypes,
                                          linkJointAxis=linkJointAxis,
                                          physicsClientId=self.client)
            p.setJointMotorControlArray(self.body, [0, 1], p.VELOCITY_CONTROL, forces=[0, 0], physicsClientId=self.client)

    def get_state(self):
        if self.mode == "single":
            js = p.getJointState(self.body, 0, physicsClientId=self.client)
            angle = js[0]
            vel = js[1]
            pos = p.getLinkState(self.body, 0, physicsClientId=self.client)[0]
            return {"theta": angle, "theta_dot": vel, "pos": pos}
        else:
            js0 = p.getJointState(self.body, 0, physicsClientId=self.client)
            js1 = p.getJointState(self.body, 1, physicsClientId=self.client)
            pos0 = p.getLinkState(self.body, 0, physicsClientId=self.client)[0]
            pos1 = p.getLinkState(self.body, 1, physicsClientId=self.client)[0]
            return {"th1": js0[0], "th2": js1[0], "w1": js0[1], "w2": js1[1], "pos1": pos0, "pos2": pos1}

    def step(self, torque=0.0):
        # apply torque to first joint only
        if self.mode == "single":
            p.setJointMotorControl2(self.body, 0, p.TORQUE_CONTROL, force=torque, physicsClientId=self.client)
        else:
            # torque on first joint
            p.setJointMotorControl2(self.body, 0, p.TORQUE_CONTROL, force=torque, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        if self.gui:
            time.sleep(self.dt)

    def close(self):
        p.disconnect(physicsClientId=self.client)
