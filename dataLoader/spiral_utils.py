from torch import Tensor, tensor, maximum, cos, sin, linspace, stack, subtract, divide, cross
from torch.linalg import norm
from numpy import pi


def create_spiral_points(num_loops=3, num_points=100):
    # Set the number of loops and the number of points per loop
    num_loops = 3
    num_points = 100

    z = linspace(0., 1., num_points)

    theta_max = num_loops * 2 * pi
    theta = linspace(0, theta_max, num_points)
    b = 0.2 ** (z)
    a = 2.0

    r = a + b * theta

    x = r * cos(theta)
    y = r * sin(theta)
    z = 1. - z

    scaling_value = maximum(x.max(), y.max())
    x = x / scaling_value
    y = y / scaling_value

    x += 1.0
    y += 1.0
    x = x / 2.
    y = y / 2.

    return stack((x, y, z), dim=-1)


def scale_spiral_to_roi(scene_aabb, spiral_points):
    return spiral_points * (scene_aabb[1] - scene_aabb[0]) + scene_aabb[0]


def make_look_at(position, target, up):
    forward = subtract(target, position)
    forward = divide(forward, norm(forward))
    right = cross(forward, up)

    # if forward and up vectors are parallel, right vector is zero;
    #   fix by perturbing up vector a bit
    if norm(right) < 0.001:
        epsilon = tensor([0.001, 0, 0])
        right = cross(forward, up + epsilon)

    right = divide(right, norm(right))

    up = cross(right, forward)
    up = divide(up, norm(up))

    # return np.array([[right[0], up[0], -forward[0], position[0]],
    # 			[right[1], up[1], -forward[1], position[1]],
    # 			[right[2], up[2], -forward[2], position[2]],
    # 			[0, 0, 0, 1]])
    return tensor([
        [right[0], up[0], forward[0], position[0]],
        [right[1], up[1], forward[1], position[1]],
        [right[2], up[2], forward[2], position[2]],
        [0, 0, 0, 1]])


def create_spiral(scene_aabb: Tensor, up: Tensor, invert_z: bool = False) -> Tensor:
    center_point = (scene_aabb[0] + scene_aabb[1]) / 2.

    spiral_points = create_spiral_points()
    if invert_z:
        spiral_points[..., -1] = 1. - spiral_points[..., -1]
    camera_positions = scale_spiral_to_roi(scene_aabb, spiral_points)

    c2ws = []
    for camera_position in camera_positions:
        c2w = make_look_at(camera_position, center_point, up)
        c2ws.append(c2w)
    return stack(c2ws)
