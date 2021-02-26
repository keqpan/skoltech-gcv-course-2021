import k3d
import numpy as np


def illustrate_camera(
    camera_pose,
    l=1.0,
    w=1.0,
    use_head=False,
    hs=1.0,
):
    camera_center = np.array([
        camera_pose.frame_origin,
        camera_pose.frame_origin,
        camera_pose.frame_origin])

    camera_frame = np.array([
        camera_pose.frame_axes
    ]) * l

    vectors = k3d.vectors(
        camera_center,
        camera_frame,
        use_head=use_head,
        head_size=hs,
        line_width=w,
        colors=[0xff0000, 0xff0000, 0x00ff00, 0x00ff00, 0x0000ff, 0x0000ff],)

    return vectors


def illustrate_mesh(mesh, plot=None):
    if plot is None:
        plot = k3d.plot()
        
    plt_surface = k3d.mesh(mesh.vertices, mesh.faces,
                           color_map = k3d.colormaps.basic_color_maps.Blues,
                           attribute=mesh.vertices[:, 2])

    plot += plt_surface
    return plot


def illustrate_points(points, plot=None, size=0.1, colors=None):
    if plot is None:
        plot = k3d.plot(name='points')
        
    if colors is not None:
        plt_points = k3d.points(positions=points, point_size=size, colors=colors)
    else:
        plt_points = k3d.points(positions=points, point_size=size)
    plot += plt_points
    plt_points.shader='3d'
    return plot


def fibonacci_sphere_sampling(number_of_views=1, seed=None, radius=1.0, positive_z=False):
    # Returns [x,y,z] tuples of a fibonacci sphere sampling
    # http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
    # commonly needed to evenly cover an sphere enclosing the object
    # for rendering from that points
    # (hypothetically this should be giving us most important projections of a 3D shape)
    if positive_z:
        number_of_views *= 2
    rnd = 1.
    if seed is not None:
        np.random.seed(seed)
        rnd = np.random.random() * number_of_views

    points = []
    offset = 2. / number_of_views
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(number_of_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % number_of_views) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        if positive_z:
            s = np.arcsin(z / radius) * 180.0 / np.pi
            if z > 0.0 and s > 30:
                points.append([radius * x, radius * y, radius * z])
        else:
            points.append([radius * x, radius * y, radius * z])

    return points

