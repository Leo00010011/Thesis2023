import open3d as o3d
import os
import re


def load_dataset(config):
    path = config['path_dataset']
    path_intrinsic = config['path_intrinsic']
    path_trajectory = os.path.join(path, config['template_global_traj'])
    path_mesh = os.path.join(path, config['template_global_mesh'])

    def get_file_list(path, extension=None):

        def sorted_alphanum(file_list_ordered):
            def convert(text): return int(text) if text.isdigit() else text

            def alphanum_key(key): return [
                convert(c) for c in re.split('([0-9]+)', key)
            ]
            return sorted(file_list_ordered, key=alphanum_key)

        if extension is None:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
        else:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and
                os.path.splitext(f)[1] == extension
            ]
        file_list = sorted_alphanum(file_list)
        return file_list

    depth_image_path = get_file_list(os.path.join(path, "depth/"),
                                     extension=".png")
    color_image_path = get_file_list(os.path.join(path, "color_masked/"),
                                     extension=".jpg")
    assert (len(depth_image_path) == len(color_image_path))

    intrinsic = o3d.io.read_pinhole_camera_intrinsic(
        os.path.join(path_intrinsic))

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
        os.path.join(path_trajectory))

    rgbd_images = []
    selected_params = []
    for i in range(0, len(depth_image_path), 10):
        param = camera_trajectory.parameters[i]
        param.intrinsic = intrinsic
        selected_params.append(param)
        depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
        color = o3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=config['depth_scale'], depth_trunc=config['max_depth'], convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory.parameters = selected_params
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(path_mesh))

    return mesh, rgbd_images, camera_trajectory


def run(config):
    mesh, rgbd_images, camera_trajectory = load_dataset(config)
    mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(mesh,
                                                                              rgbd_images,
                                                                              camera_trajectory,
                                                                              o3d.pipelines.color_map.NonRigidOptimizerOption(maximum_iteration=100, maximum_allowable_depth=config['max_depth']))

    o3d.io.write_triangle_mesh(os.path.join(
        config['path_dataset'], 'scene', 'color_map.ply'), mesh)
