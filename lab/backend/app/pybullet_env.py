import pybullet as p
import pybullet_data
import numpy as np
import base64
import time


class PyBulletCartPole:
    """Create a cart-and-pole (single or double) system in PyBullet.

    The cart can move along X on a simple track and the pendulum rod(s) are
    revolute joints attached to the cart. The `step(force)` method accepts a
    scalar horizontal force applied to the cart.
    """

    def __init__(self, mode="single", dt=0.02, gui=False, track_length=2.0):
        self.mode = mode
        self.dt = dt
        self.gui = gui
        self.track_length = float(track_length)
        try:
            self.client = p.connect(p.GUI if gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            p.setTimeStep(self.dt, physicsClientId=self.client)
        except Exception:
            raise

        # simple plane
        try:
            p.loadURDF("plane.urdf", physicsClientId=self.client)
        except Exception:
            pass

        self._build_cartpole()

    def _build_cartpole(self):
        # Cart is created as the base link attached to a fixed base via a prismatic joint
        # We'll create a multi-body with first link as cart (prismatic) then pendulum links
        cart_mass = 1.0
        cart_half = [0.2, 0.15, 0.08]
        cart_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=cart_half, physicsClientId=self.client)
        cart_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=cart_half, rgbaColor=[0.4, 0.4, 0.4, 1.0], physicsClientId=self.client)

        linkMasses = [cart_mass]
        linkCollisionShapeIndices = [cart_col]
        linkVisualShapeIndices = [cart_vis]
        # cart positioned at base offset; joint axis along X for prismatic motion
        linkPositions = [[0, 0, 0.1]]
        linkOrientations = [[0, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1]]
        linkParentIndices = [0]
        linkJointTypes = [p.JOINT_PRISMATIC]
        linkJointAxis = [[1, 0, 0]]

        # We'll add pendulum rod(s) as additional links parented to the cart link index 1
        masses = []
        cols = []
        vis = []
        positions = []
        orientations = []
        parent_idxs = []
        joint_types = []
        joint_axes = []

        # pendulum properties
        rod_mass = 0.1
        rod_length = 1.0
        rod_half = [0.03, 0.03, rod_length / 2]

        if self.mode == 'single':
            # one rod attached to cart
            col1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=rod_half, physicsClientId=self.client)
            vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=rod_half, rgbaColor=[0.8, 0.2, 0.2, 1.0], physicsClientId=self.client)
            masses.append(rod_mass)
            cols.append(col1)
            vis.append(vis1)
            # attach at cart top center; joint is revolute about Y axis
            positions.append([0, 0, rod_length / 2 + 0.18])
            orientations.append([0, 0, 0, 1])
            parent_idxs.append(1)
            joint_types.append(p.JOINT_REVOLUTE)
            joint_axes.append([0, 1, 0])
        else:
            # double: two rods chained
            col1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=rod_half, physicsClientId=self.client)
            vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=rod_half, rgbaColor=[0.8, 0.2, 0.2, 1.0], physicsClientId=self.client)
            col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=rod_half, physicsClientId=self.client)
            vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=rod_half, rgbaColor=[0.2, 0.8, 0.2, 1.0], physicsClientId=self.client)
            # first rod parented to cart (link index 1)
            masses.append(rod_mass)
            cols.append(col1)
            vis.append(vis1)
            positions.append([0, 0, rod_length / 2 + 0.18])
            orientations.append([0, 0, 0, 1])
            parent_idxs.append(1)
            joint_types.append(p.JOINT_REVOLUTE)
            joint_axes.append([0, 1, 0])
            # second rod parented to first rod (index will be 2)
            masses.append(rod_mass)
            cols.append(col2)
            vis.append(vis2)
            positions.append([0, 0, rod_length])
            orientations.append([0, 0, 0, 1])
            parent_idxs.append(2)
            joint_types.append(p.JOINT_REVOLUTE)
            joint_axes.append([0, 1, 0])

        # assemble multi-body: base mass 0, with links: cart (prismatic) then pendulum link(s)
        try:
            self.body = p.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=-1,
                                          basePosition=[0, 0, 0.0],
                                          linkMasses=linkMasses + masses,
                                          linkCollisionShapeIndices=linkCollisionShapeIndices + cols,
                                          linkVisualShapeIndices=linkVisualShapeIndices + vis,
                                          linkPositions=linkPositions + positions,
                                          linkOrientations=linkOrientations + orientations,
                                          linkInertialFramePositions=linkInertialFramePositions + [[0,0,0]] * len(masses),
                                          linkInertialFrameOrientations=linkInertialFrameOrientations + [[0,0,0,1]] * len(masses),
                                          linkParentIndices=linkParentIndices + parent_idxs,
                                          linkJointTypes=linkJointTypes + joint_types,
                                          linkJointAxis=linkJointAxis + joint_axes,
                                          physicsClientId=self.client)
        except Exception:
            # Fallback: create a simple visual-only body and rely on step simulation elsewhere
            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, physicsClientId=self.client)

        # disable default motors so we can apply forces
        try:
            n = p.getNumJoints(self.body, physicsClientId=self.client)
            if n > 0:
                p.setJointMotorControlArray(self.body, list(range(n)), p.VELOCITY_CONTROL, forces=[0] * n, physicsClientId=self.client)
        except Exception:
            pass

        # add a ball at the end of the last rod for visual flourish
        try:
            sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.06, physicsClientId=self.client)
            sph_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.9, 0.8, 0.2, 1.0], physicsClientId=self.client)
            self.ball = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere, baseVisualShapeIndex=sph_vis, basePosition=[0,0,0], physicsClientId=self.client)
        except Exception:
            self.ball = None

        self.n_joints = p.getNumJoints(self.body, physicsClientId=self.client) if hasattr(self, 'body') else 0

    def get_state(self):
        # return cart position, cart velocity (approx), and link angles/vels
        try:
            if self.mode == 'single':
                # joint 0: cart prismatic, joint 1: pendulum revolute
                if self.n_joints >= 2:
                    js0 = p.getJointState(self.body, 0, physicsClientId=self.client)
                    js1 = p.getJointState(self.body, 1, physicsClientId=self.client)
                    return {"x": js0[0], "x_dot": js0[1], "theta": js1[0], "theta_dot": js1[1]}
                else:
                    return {"x": 0.0, "x_dot": 0.0, "theta": 0.0, "theta_dot": 0.0}
            else:
                # double: cart + two joints
                if self.n_joints >= 3:
                    js0 = p.getJointState(self.body, 0, physicsClientId=self.client)
                    js1 = p.getJointState(self.body, 1, physicsClientId=self.client)
                    js2 = p.getJointState(self.body, 2, physicsClientId=self.client)
                    return {"x": js0[0], "x_dot": js0[1], "th1": js1[0], "th2": js2[0], "w1": js1[1], "w2": js2[1]}
                else:
                    return {"x": 0.0, "x_dot": 0.0, "th1": 0.0, "th2": 0.0, "w1": 0.0, "w2": 0.0}
        except Exception:
            return {}

    def step(self, force=0.0):
        # Apply horizontal force to the cart (link 0) by applying external force to link index 0
        try:
            if getattr(self, 'n_joints', 0) > 0:
                # apply force to the cart link's center of mass in world frame
                p.applyExternalForce(self.body, 0, [float(force), 0, 0], [0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.client)
            else:
                # fallback: apply to base
                p.applyExternalForce(self.body, -1, [float(force), 0, 0], [0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.client)
        except Exception:
            pass
        # step simulator
        try:
            p.stepSimulation(physicsClientId=self.client)
        except Exception:
            pass
        if self.gui:
            import time
            time.sleep(self.dt)

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass

    def get_scene_tree(self):
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        except Exception:
            base_pos, base_orn = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

        vis_map = {}
        try:
            vis = p.getVisualShapeData(self.body, physicsClientId=self.client)
            for v in vis:
                link_index = v[1]
                geom_type = v[2] if len(v) > 2 else None
                dims = v[3] if len(v) > 3 else None
                filename = v[4] if len(v) > 4 else None
                local_pos = v[5] if len(v) > 5 else None
                local_orn = v[6] if len(v) > 6 else None
                rgba = v[7] if len(v) > 7 else None
                vis_map[link_index] = {"geom_type": int(geom_type) if geom_type is not None else None, "dimensions": dims, "filename": filename, "local_position": local_pos, "local_orientation": local_orn, "rgba": rgba}
        except Exception:
            vis_map = {}

        links = []
        try:
            nlinks = p.getNumJoints(self.body, physicsClientId=self.client)
        except Exception:
            nlinks = getattr(self, 'n_joints', 0)

        links.append({"link_index": -1, "world_position": list(base_pos), "world_orientation": list(base_orn), "visual": vis_map.get(-1)})
        for i in range(nlinks):
            try:
                ls = p.getLinkState(self.body, i, physicsClientId=self.client)
                pos = ls[0] if len(ls) > 0 else None
                orn = ls[1] if len(ls) > 1 else None
            except Exception:
                pos, orn = None, None
            links.append({"link_index": int(i), "world_position": list(pos) if isinstance(pos, (list, tuple)) else pos, "world_orientation": list(orn) if isinstance(orn, (list, tuple)) else orn, "visual": vis_map.get(i)})

        scene = {"bodies": [{"body_id": int(self.body), "base_position": list(base_pos), "base_orientation": list(base_orn), "links": links}]}
        return scene


    # (alias moved to module scope below)

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

    def get_scene_tree(self):
        """Return a serializable representation of the PyBullet scene for frontend replication.

        The returned structure contains bodies (here the single pendulum body), the
        base transform and a list of links with world transforms and basic visual shape
        metadata when available.
        """
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        except Exception:
            base_pos, base_orn = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

        # Collect visual shape metadata (may include base with linkIndex == -1)
        vis_map = {}
        def _coerce_value(x):
            # Convert numpy arrays/lists/tuples to lists, bytes to utf-8 or base64
            if isinstance(x, (list, tuple, np.ndarray)):
                return [ _coerce_value(xx) for xx in x ]
            if isinstance(x, (bytes, bytearray)):
                try:
                    return x.decode('utf-8')
                except Exception:
                    return base64.b64encode(x).decode('ascii')
            # memoryview -> bytes
            if isinstance(x, memoryview):
                try:
                    b = x.tobytes()
                    return _coerce_value(b)
                except Exception:
                    return None
            return x

        try:
            vis = p.getVisualShapeData(self.body, physicsClientId=self.client)
            for v in vis:
                # tuple layout varies slightly across pybullet versions; be defensive
                # typical: (objectUniqueId, linkIndex, visualGeometryType, dimensions, filename, localPos, localOrn, rgba)
                link_index = v[1]
                geom_type = v[2] if len(v) > 2 else None
                dims = v[3] if len(v) > 3 else None
                filename = v[4] if len(v) > 4 else None
                local_pos = v[5] if len(v) > 5 else None
                local_orn = v[6] if len(v) > 6 else None
                rgba = v[7] if len(v) > 7 else None
                vis_map[link_index] = {
                    "geom_type": int(geom_type) if geom_type is not None else None,
                    "dimensions": _coerce_value(dims),
                    "filename": _coerce_value(filename),
                    "local_position": _coerce_value(local_pos),
                    "local_orientation": _coerce_value(local_orn),
                    "rgba": _coerce_value(rgba),
                }
        except Exception:
            vis_map = {}

        # Build link list including base (-1) and joint links
        links = []
        try:
            nlinks = p.getNumJoints(self.body, physicsClientId=self.client)
        except Exception:
            nlinks = getattr(self, 'n_joints', 0)

        # base entry
        links.append({
            "link_index": -1,
            "world_position": list(base_pos),
            "world_orientation": list(base_orn),
            "visual": vis_map.get(-1),
        })

        for i in range(nlinks):
            try:
                ls = p.getLinkState(self.body, i, physicsClientId=self.client)
                # choose indices that are most commonly available (pos, orn)
                pos = ls[0] if len(ls) > 0 else None
                orn = ls[1] if len(ls) > 1 else None
            except Exception:
                pos, orn = None, None
            links.append({
                "link_index": int(i),
                "world_position": list(pos) if isinstance(pos, (list, tuple, np.ndarray)) else pos,
                "world_orientation": list(orn) if isinstance(orn, (list, tuple, np.ndarray)) else orn,
                "visual": vis_map.get(i),
            })

        scene = {
            "bodies": [
                {
                    "body_id": int(self.body),
                    "base_position": list(base_pos),
                    "base_orientation": list(base_orn),
                    "links": links,
                }
            ]
        }
        return scene


    
