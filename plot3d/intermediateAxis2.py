import matplotlib.pyplot as plt
import numpy as np
import quaternion
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich import print
from rich.traceback import install
from scipy.integrate import solve_ivp

from unitCube import cube_faces_from_vertices, unit_cube_vertices

install()


def animate_cube(size, quaternions, interval):
    # Initialize figure and axis for 3D rendering
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rotating Cube with Solid Colored Faces")

    # Create cube vertices and faces
    vertices = unit_cube_vertices() - vec3(0.5, 0.5, 0.5)
    vertices = vertices * size
    faces = cube_faces_from_vertices(vertices)
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow"]
    cube_poly = Poly3DCollection(faces, facecolors=colors, edgecolor="k", alpha=0.9)
    ax.add_collection3d(cube_poly)

    def update(num, poly, vertices, orientations):
        """Update function for animation."""
        # Apply rotation to vertices
        R = quaternion.as_rotation_matrix(orientations[num])
        rotated_vertices = vertices @ R
        rotated_faces = cube_faces_from_vertices(rotated_vertices)

        poly.set_verts(rotated_faces)
        return (poly,)

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=range(len(quaternions)),
        fargs=(cube_poly, vertices, quaternions),
        interval=interval,
    )

    plt.show()


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def rectangular_prism_inertia(m, vec):
    w, h, d = vec

    # Calculate the inertia tensor for a rectangular prism
    Ix = (1 / 12) * m * (h**2 + d**2)
    Iy = (1 / 12) * m * (w**2 + d**2)
    Iz = (1 / 12) * m * (w**2 + h**2)
    I = np.diag([Ix, Iy, Iz])  # Inertia tensor matrix
    return I

def simulation(size):
    # Initial conditions
    mass = 0.1
    I = rectangular_prism_inertia(mass, size)
    initial_q = np.quaternion(1, 0, 0, 0)
    initial_omega_body = np.array([-0.03, 10, 0.01])

    # Initial state vector
    initial_state = np.concatenate([quaternion.as_float_array(initial_q), initial_omega_body])

    # Time span and time steps
    t_span = (0, 10)
    t_eval = np.arange(0, 10, 0.01)

    # No external torque
    torque_body = vec3(0, 0, 0)

    def rigid_body_ode(t, y):
        # y[:4] are the quaternion components
        # y[4:] are the angular velocity components
        q = np.quaternion(*y[:4])
        omega_body = y[4:]
        
        # Compute the omega quaternion (with zero real part)
        omega_quat = quaternion.from_float_array([0, *omega_body])
        
        # Quaternion derivative
        q_dot = 0.5 * q * omega_quat
        
        # Angular velocity derivative
        omega_body_dot = np.linalg.inv(I) @ (torque_body - np.cross(omega_body, I @ omega_body))
        
        return np.concatenate([quaternion.as_float_array(q_dot), omega_body_dot])
    
    # Solve the differential equations
    result = solve_ivp(rigid_body_ode, t_span, initial_state, t_eval=t_eval, method='RK45')

    # Extract the quaternions from the solution
    quaternions = result.y[:4].T  # Transpose to get the list of quaternions
    qs =  [quaternion.from_float_array(q) for q in quaternions]
    return qs



def main():
    # size = vec3(0.5, 1, 2)
    # I = rectangular_prism_inertia(0.1, size)

    # # Initial quaternion (no rotation)
    # q = np.quaternion(1, 0, 0, 0)

    # # Simulation parameters
    # dt = 0.01  # Time step
    # time = np.arange(0, 10, dt)  # Total time of 10 seconds
    # quaternions = [q]  # List to save quaternions

    # torque_body = vec3(0, 0, 0)  # No external torque
    # omega_body = np.array(
    #     [-0.03, 10, 0.01]
    # )  # Small perturbations around intermediate axis

    # for t in time:
    #     omega_body_dot = np.linalg.inv(I) @ (
    #         torque_body - np.cross(omega_body, I @ omega_body)
    #     )
    #     omega_body += omega_body_dot * dt
    #     omega_body_quad = quaternion.from_vector_part(omega_body)
    #     q_dot = 0.5 * q * omega_body_quad
    #     q = q + q_dot * dt
    #     q /= np.linalg.norm(quaternion.as_float_array(q))
    #     quaternions.append(q)

    
    size = vec3(0.5, 1, 2)
    quaternions = simulation(size)
    animate_cube(size, quaternions, 50)


if __name__ == "__main__":
    main()
