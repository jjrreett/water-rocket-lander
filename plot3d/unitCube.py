import numpy as np


def unit_cube_vertices():
    """Returns the vertices of a unit cube centered at the origin."""
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )


def cube_faces_from_vertices(vertices):
    """Creates 6 faces of the cube from 8 vertices."""
    return [
        vertices[[0, 1, 2, 3]],  # Bottom
        vertices[[4, 5, 6, 7]],  # Top
        vertices[[0, 1, 5, 4]],  # Front
        vertices[[2, 3, 7, 6]],  # Back
        vertices[[0, 3, 7, 4]],  # Left
        vertices[[1, 2, 6, 5]],  # Right
    ]
