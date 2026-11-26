# HW 4.7: suspended cloth dropped onto a horse
#   uv run python main.py 4_6 scenes/hw4/custom_scene.py
#   uv run python main.py 4_6 scenes/hw4/custom_scene.py -output outputs/hw_4_7.mp4 -duration 5 -fps 30
#
# Model credit: "Horse free download" by pgonarg (https://sketchfab.com/pgonarg),
# https://sketchfab.com/3d-models/horse-free-download-9b77bb7542354b10bb7b4c6a25866724
# licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

import numpy as np
import trimesh
from pathlib import Path
from scenes.hw4.suspended_cloth_collide import make_scene as suspended_scene


def load_model():
    mesh = trimesh.load(Path(__file__).with_name('horse.gltf'),
                        force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    triangles = np.asarray(mesh.faces, dtype=np.uint32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()

    # Height 2.2 (wider back for the cloth), centered, on the floor
    bmin, bmax = vertices.min(axis=0), vertices.max(axis=0)
    vertices -= (bmin + bmax) / 2
    vertices *= 2.2 / (bmax[1] - bmin[1])
    vertices[:, 1] -= vertices[:, 1].min()

    return {
        'vertices': np.ascontiguousarray(vertices),
        'triangles': np.ascontiguousarray(triangles),
        'vertex_normals': np.ascontiguousarray(normals),
    }


def make_scene():
    scene = suspended_scene()
    scene['collision_mesh'] = load_model()

    scene['spheres'] = np.zeros((0, 4), dtype=np.float32)
    scene['planes'] = np.array([[0, 1, 0, 0]], dtype=np.float32)

    cloth_count = scene['cloth_grid_size'][0] * scene['cloth_grid_size'][1]
    scene['velocities'][:cloth_count, 1] = -2.5

    scene['substeps'] = 6
    scene['iterations'] = 30
    scene['static_friction'] = 0.9
    scene['kinetic_friction'] = 0.6

    scene['cam_pos'] = (4.5, 3.5, -4.5)
    scene['cam_target'] = (0, 1.3, 0)

    return scene
