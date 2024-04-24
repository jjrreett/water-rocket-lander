import numpy as np
import pytest


def rotate_around_vector(phi, w):
    # Ensure w is a unit vector
    w = w / np.linalg.norm(w)

    # Quaternion components
    q0 = np.cos(phi / 2)
    qx, qy, qz = np.sin(phi / 2) * w

    # Rotation matrix
    R = np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * q0,
                2 * qx * qz + 2 * qy * q0,
            ],
            [
                2 * qx * qy + 2 * qz * q0,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * q0,
            ],
            [
                2 * qx * qz - 2 * qy * q0,
                2 * qy * qz + 2 * qx * q0,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )

    return R


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.q = np.array([w, x, y, z])  # Storing components in a NumPy array

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.w} + {self.x}i + {self.y}j + {self.z}k)"

    def __add__(self, other):
        # Element-wise addition of two quaternions
        result = self.q + other.q
        return Quaternion(result[0], result[1], result[2], result[3])

    def __sub__(self, other):
        # Element-wise subtraction of two quaternions
        result = self.q - other.q
        return Quaternion(result[0], result[1], result[2], result[3])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # Quaternion multiplication (Hamilton product)
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return Quaternion(w, x, y, z)
        else:
            # Scalar multiplication
            return Quaternion(
                self.w * other, self.x * other, self.y * other, self.z * other
            )

    def __neg__(self):
        # Negation of a quaternion
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        """Returns the inverse of the quaternion"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @staticmethod
    def from_angle_axis(angle, axis):
        """Create a quaternion from an angle and axis"""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        scalar = np.cos(angle / 2.0)
        vector = np.sin(angle / 2.0) * axis
        return Quaternion(scalar, *vector)

    def rotate_vector(self, v):
        """Rotates a vector using this quaternion"""
        q_vec = Quaternion(0, *v)
        q_inv = self.inverse()
        q_rotated = self * q_vec * q_inv
        return np.array([q_rotated.x, q_rotated.y, q_rotated.z])

    @staticmethod
    def from_scalar_vector(scalar, vector):
        """Create a quaternion from a scalar and a 3D vector"""
        return Quaternion(scalar, vector[0], vector[1], vector[2])

    def to_rotation_matrix(self):
        """Convert a quaternion into a rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )


# Pytest to verify the rotation
def test_rotate_x_vector_around_z_to_y():
    x_vector = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    angle_90_deg = np.pi / 2
    quaternion = Quaternion.from_angle_axis(angle_90_deg, z_axis)
    rotated_vector = quaternion.rotate_vector(x_vector)
    np.testing.assert_almost_equal(rotated_vector, np.array([0, 1, 0]))
