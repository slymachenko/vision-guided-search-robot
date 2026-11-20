'''
In this file the swift simulation will be created and started,
the ambient will be a closed enviornment like a room and inside there should be a set of walls
and small obstacles.
'''

import spatialgeometry as sg
import spatialmath as sm
import swift

def swift_env_creator():
    env = swift.Swift()
    env.launch(realtime=True, browser="browser")

    obstacles = [
        # walls (cuboids)
        {"type": "box", "kind": "wall", "size": (2.01, 0.01, 0.2), "center": (1.0, 0.0, 0.1)},
        {"type": "box", "kind": "wall", "size": (2.01, 0.01, 0.2), "center": (1.0, 2.0, 0.1)},
        {"type": "box", "kind": "wall", "size": (0.01, 2.01, 0.2), "center": (0.0, 1.0, 0.1)},
        {"type": "box", "kind": "wall", "size": (0.01, 2.01, 0.2), "center": (2.0, 1.0, 0.1)},

        # box obstacles
        {"type": "box", "kind": "obstacle", "size": (0.8, 0.3, 0.2),  "center": (1.3, 1.0, 0.1)},
        {"type": "box", "kind": "obstacle", "size": (0.2, 0.6, 0.2),  "center": (0.3, 0.5, 0.1)},
        {"type": "box", "kind": "obstacle", "size": (0.1, 0.3, 0.2),  "center": (0.7, 0.4, 0.1)},
        {"type": "box", "kind": "obstacle", "size": (0.6, 0.05, 0.2), "center": (0.3, 1.7, 0.1)},

        # cylinders
        {"type": "cylinder", "kind": "obstacle", "radius": 0.1, "height": 0.2, "center": (1.2, 0.4, 0.1)},
        {"type": "cylinder", "kind": "obstacle", "radius": 0.1, "height": 0.2, "center": (0.3, 1.3, 0.1)},
        {"type": "cylinder", "kind": "obstacle", "radius": 0.2, "height": 0.2, "center": (1.3, 1.6, 0.1)},
    ]

    for ob in obstacles:
        if ob["type"] == "box":
            shape = sg.Cuboid(scale=list(ob["size"]), color=(0.5,0.5,0.5,1.0))
        elif ob["type"] == "cylinder":
            shape = sg.Cylinder(radius=ob["radius"], length=ob["height"], color=(0.5,0.5,0.5,1.0))
        shape.T = sm.SE3(*ob["center"])
        env.add(shape)

    objective = {"type": "sphere", "kind": "objective", "radius": 0.02, "center": (1.8, 1.8, 0.02)}
    shape = sg.Sphere(radius=objective["radius"], color=(1,0,0,1.0))
    shape.T = sm.SE3(*objective["center"])
    env.add(shape)


    env.step(0.0)  # refresh the viewer so the object shows up

    return env,obstacles,objective
if __name__ == "__main__":
    env, obstacles, objective = swift_env_creator()
    for _ in range(1000):
        env.step(0.05)