import numpy as np
from functools import lru_cache
from cv2 import getOptimalNewCameraMatrix
from bs4 import BeautifulSoup
import os
from .utils import rescale_poses, recenter_poses


def cast_double_or_default(tag, default):
    if tag is None:
        return default
    else:
        return float(tag.string)


supported_extensions = [
    '.im24', '.m4b', '.ico', '.avc', '.flc', '.ism', '.tf8', '.mods', '.dnxhr', '.yuv', '.gsm', '.pfm', '.ps', '.h263',
    '.orf', '.mhd', '.m2v', '.mos', '.nhdr', '.png', '.fits', '.apng', '.stk', '.vb', '.wdp', '.3fr', '.tiff', '.bif',
    '.roq', '.pef', '.dat', '.webp', '.vc2', '.ftc', '.xpm', '.yop', '.rw2', '.bmv', '.3g2', '.flm', '.icb', '.bufr',
    '.swf', '.gdcm', '.nrw', '.xwd', '.drf', '.vda', '.iim', '.xl', '.lfr', '.pcx', '.pxr', '.fpx', '.jpe', '.avs2',
    '.jif', '.rgba', '.sga', '.cs1', '.im8', '.bsdf', '.qcif', '.jpg', '.mic', '.mri', '.fff', '.gxf', '.h264', '.mpg',
    '.kux', '.dcm', '.fts', '.srw', '.lvf', '.wmf', '.rwz', '.cavs', '.a64', '.ser', '.pxn', '.avs', '.cine', '.lsm',
    '.jp2', '.cur', '.wbmp', '.ftu', '.hevc', '.jpc', '.dvd', '.mts', '.pix', '.v', '.xbm', '.amr', '.h5', '.gif',
    '.rgb', '.mpo', '.zif', '.cdg', '.sgi', '.mef', '.jxr', '.vc1', '.sun', '.spe', '.pbm', '.y4m', '.jpf', '.bmq',
    '.h26l', '.ia', '.mgh', '.A64', '.hdr', '.SPIDER', '.sti', '.vtk', '.rec', '.im', '.adp', '.btf', '.avi', '.grib',
    '.pxm', '.sr2', '.fz', '.dcx', '.jng', '.m2t', '.mpd', '.pnm', '.yuv10', '.exr', '.raw', '.mj2', '.hdf5',
    '.XVTHUMB', '.cut', '.bw', '.bay', '.cpk', '.MCIDAS', '.lbm', '.raf', '.wbm', '.m4v', '.ogg', '.psd', '.nut',
    '.PCX', '.tga', '.rcv', '.ifv', '.ptx', '.lfp', '.mnc', '.mvi', '.sdr2', '.bmp', '.kc2', '.nia', '.g3', '.psp',
    '.rs', '.265', '.targa', '.dsc', '.v210', '.mpc', '.ismv', '.msp', '.mk3d', '.mka', '.koa', '.mxg', '.m2ts',
    '.xface', '.ct', '.dcr', '.jls', '.ogv', '.mpeg', '.npz', '.gbr', '.avs3', '.wap', '.nrrd', '.pcd', '.f4v', '.fit',
    '.img', '.mov', '.svag', '.crw', '.pdf', '.sunras', '.pic', '.3gp', '.pam', '.svs', '.mjpg', '.m4a', '.mxf', '.fli',
    '.qpi', '.qtk', '.ipl', '.j2k', '.isma', '.jpeg', '.k25', '.dv', '.ipu', '.mnc2', '.rdc', '.ptiff', '.j2c', '.blp',
    '.dav', '.hdp', '.pct', '.viv', '.hdf', '.dip', '.jpx', '.wtv', '.srf', '.imx', '.pcoraw', '.rm', '.dnxhd', '.dng',
    '.264', '.icns', '.ivr', '.nef', '.dc2', '.mjpeg', '.gel', '.mp4', '.ppm', '.ras', '.sr', '.mha', '.flv', '.h265',
    '.mrw', '.cif', '.gipl', '.jfif', '.asf', '.webm', '.ljpg', '.mkv', '.cr2', '.palm', '.y', '.ty', '.ptif', '.eps',
    '.iiq', '.mks', '.dif', '.h261', '.idf', '.arw', '.pict', '.tif', '.emf', '.mdc', '.cgi', '.kdc', '.pgm', '.moflex',
    '.im1', '.ts', '.pgmyuv', '.dpx', '.qptiff', '.vob', '.amv', '.dicom', '.cdxl', '.drc', '.xmv', '.dib', '.iff',
    '.ivf', '.m1v', '.vst', '.wmv', '.cap', '.ecw', '.avr', '.IMT', '.chk', '.dds', '.nii', '.rwl', '.ndpi', '.obu',
    '.erf']


@lru_cache(1)
def load_cameras_xml(camera_filepath, base_dir, img_resize_factor=1., img_dirname='undistorted_images'):
    filenames = []
    cam2world = []
    Ks = []
    metashape_filenames = []
    metashape_masks_filenames = []

    # Open the file and read the contents
    with open(camera_filepath, 'r', encoding='utf-8') as file:
        cams_xml = file.read()

    cams_file_content = BeautifulSoup(cams_xml, "xml")
    chunk_results = cams_file_content.find_all("chunk")
    assert len(chunk_results) == 1, "Only one chunk should be present inside the file"
    chunk = chunk_results[0]

    sensors = chunk.find("sensors")
    if sensors is None:
        print(f"No sensors list found inside chunk, ignoring file {camera_filepath}")
        return {}

    cameras = chunk.find("cameras")
    if cameras is None:
        print(f"No cameras list found inside chunk, ignoring file {camera_filepath}")
        return {}

    cameras = cameras.find_all("camera")
    if cameras is None:
        print(f"No camera found inside cameras list, ignoring file {camera_filepath}")
        return {}

    # print(f"{camera_filepath} contains {len(cameras)} cameras")
    if len(cameras) == 0:
        print(f"Sequence is empty, ignoring file {camera_filepath}")
        return {}

    for pos_idx, camera in enumerate(cameras):
        if camera.get('enabled') == 'false':
            continue

        image_filename = camera.get('label')
        if image_filename is None:
            camera_id = camera.get('id', pos_idx)
            print(f"Camera #{camera_id} inside {camera_filepath} do not have any file label, ignored")
            continue

        sensor_id = camera.get('sensor_id')
        transform_str = camera.find('transform').string
        transform = np.array([float(x) for x in transform_str.split(' ')], dtype=np.float32).reshape(4, -1)

        sensor = sensors.find('sensor', {'id': sensor_id})
        if sensor is None:
            camera_id = camera.get('id', pos_idx)
            print(
                f"Sensor #{sensor_id} not found inside sensors list, ignoring image #{camera_id} inside {camera_filepath}")
            continue

        sensor_resolution = sensor.find('resolution')
        if sensor_resolution is None:
            camera_id = camera.get('id', pos_idx)
            print(
                f"Sensor #{sensor_id} does not have a resolution parameter, ignoring connected image #{camera_id} inside {camera_filepath}")
            continue

        W = int(sensor_resolution.get('width'))
        H = int(sensor_resolution.get('height'))

        sensor_calibration = sensor.find('calibration')
        if sensor_calibration is None:
            camera_id = camera.get('id', pos_idx)
            print(
                f"Sensor #{sensor_id} does not have a calibration parameter, ignoring connected image #{camera_id} inside {camera_filepath}")
            continue

        fx_tag = sensor_calibration.find('fx')
        fy_tag = sensor_calibration.find('fy')
        cx_tag = sensor_calibration.find('cx')
        cy_tag = sensor_calibration.find('cy')
        k1_tag = sensor_calibration.find('k1')
        k2_tag = sensor_calibration.find('k2')
        p1_tag = sensor_calibration.find('p1')
        p2_tag = sensor_calibration.find('p2')

        if fx_tag is None and fy_tag is None:
            f_tag = sensor_calibration.find('f')
            fx_tag = f_tag
            fy_tag = f_tag

        k1 = 0.
        if k1_tag is not None:
            k1 = float(k1_tag.string)
        k2 = 0.
        if k2_tag is not None:
            k2 = float(k2_tag.string)

        p_tag = sensor_calibration.find('p')
        if p1_tag is None:
            p1_tag = p_tag
        if p2_tag is None:
            p2_tag = p_tag

        fx = cast_double_or_default(fx_tag, 0.)
        fy = cast_double_or_default(fy_tag, 0.)
        cx = cast_double_or_default(cx_tag, float(W) / 2.)
        cy = cast_double_or_default(cy_tag, float(H) / 2.)
        p1 = cast_double_or_default(p1_tag, 0.)
        p2 = cast_double_or_default(p2_tag, 0.)

        cam_mat = np.asarray(
            [[fx / img_resize_factor, 0, cx / img_resize_factor], [0, fy / img_resize_factor, cy / img_resize_factor],
             [0, 0, 1.]], dtype=np.float32)
        cam_mat, _ = getOptimalNewCameraMatrix(cam_mat, np.asarray([k1, k2, p1, p2]),
                                               (int(W / img_resize_factor), int(H / img_resize_factor)), 0.)

        img_path = os.path.join(base_dir, img_dirname, image_filename)
        extension = ''
        if len(os.path.splitext(image_filename)[1]) == 0:
            # autodiscover extension
            found = False
            for common_extension in supported_extensions:
                if os.path.exists(img_path + common_extension):
                    img_path = img_path + common_extension
                    extension = common_extension
                    found = True
                    break

            if not found:
                print(f"Skip image because we was unable to find the right extension")
                continue

        if img_dirname != 'undistorted_images':
            metashape_filenames.append(os.path.join(base_dir, 'undistorted_images', image_filename + extension))
        metashape_masks_filenames.append(
            os.path.join(base_dir, 'masks_metashape', os.path.splitext(image_filename)[0] + extension))

        filenames.append(img_path)
        cam2world.append(transform)
        Ks.append(cam_mat)

    # opengl2opencv = np.asarray([[1.,  0.,  0.,  0.],
    #                             [0., -1.,  0.,  0.],
    #                             [0.,  0., -1.,  0.],
    #                             [0.,  0.,  0.,  1.]], dtype=np.float32)
    # opengl2opencv = np.asarray([[1.,  0.,  0.,  0.],
    #                             [0.,  1.,  0.,  0.],
    #                             [0.,  0., -1.,  0.],
    #                             [0.,  0.,  0.,  1.]], dtype=np.float32)
    cam2world = np.stack(cam2world)
    # cam2world[:, :, 2] = cam2world[:, :, 2] * -1
    # cam2world = cam2world @ opengl2opencv[None]
    Ks = np.stack(Ks)

    # Center and scale poses.
    # cam2world = recenter_poses(cam2world, pose_avg=np.asarray(self.DTU_POSES_AVG, dtype=cam2world.dtype, device=cam2world.device))
    cam2world, inv_transformation = recenter_poses(cam2world)
    # cam2world = rescale_poses(cam2world, scale=np.asarray(self.DTU_POSES_SCALE, dtype=cam2world.dtype, device=cam2world.device))
    cam2world, inv_scale = rescale_poses(cam2world)

    return {
        'filenames': filenames,
        'metashape_filenames': metashape_filenames,
        'metashape_masks': metashape_masks_filenames,
        'cam2world': cam2world,
        'Ks': Ks,
        'base_dir': base_dir,
    }, inv_scale, inv_transformation


def recursive_explore_repair(folder, scenes=None, img_resize_factor: float = 1.):
    scenes_dict = []
    groups = os.listdir(folder) if scenes is None else scenes
    groups.sort()
    for group in groups:
        group_dirpath = os.path.join(folder, group)
        if not os.path.isdir(group_dirpath):
            continue
        camera_xml_filepath = os.path.join(group_dirpath, "cameras.xml")
        if os.path.isfile(camera_xml_filepath):
            cameras = load_cameras_xml(camera_xml_filepath, group_dirpath, img_resize_factor=img_resize_factor)
            if len(cameras) != 0:
                scenes_dict.append(cameras)
        else:
            scenes_dict.extend(recursive_explore_repair(group_dirpath, img_resize_factor=img_resize_factor))
    return scenes_dict


if __name__ == "__main__":
    scenes_dict = recursive_explore_repair("/home/mbortolon/data/dataset/repair")
    print(len(scenes_dict))
