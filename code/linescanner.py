"""
Kamera A, alpha - Abstand c - Laserdiode B, beta | Laserpunkt: C, gamma - Laserlinie a
kartesische Position von C: x, z
"""
import numpy as np
import cv2
import math
import open3d as o3d

from libs.laser import find_laser
from libs.calc import triangulate
from libs.pointcloud import estimate_normals, export_pointcloud
# from libs.image import rotate_bound


def main(s):
    # init video
    camera = cv2.VideoCapture(s.video_path)

    # init framebuffer for difference-map
    old_frame = np.zeros((960, 540, 3), np.uint8)  # TODO: size variable

    # load image texture
    if s.texture_path is not None:
        texture = cv2.imread(s.texture_path, 1)
        texture = cv2.resize(texture, s.dims, interpolation=cv2.INTER_LINEAR)
    else:
        texture = None

    # init Camera and Laser position
    camera_pos = np.array([0, 0, 0])
    laser_pos = np.array([s.camera_laser_distance, 0, 0])
    # project imageplane into 3d space
    topleft_corner = np.array([-s.dims[0] / 2, s.dims[1] / 2, s.lens_length])

    # init live 3D-Viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800, left=1000)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud_frame = o3d.geometry.PointCloud()

    # TODO: initial scale is crucial; needs scale-to-fit and  wider clipping plane
    points = np.array([[-50, -50, -50], [50, 50, 50]])
    colors = np.array([[1, 0, 0], [1, 0, 0]])

    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pointcloud)

    # VIDEO PROCESSING LOOP
    frame_number = 0
    while True:
        # calculate current normal-vector of laserplane
        laser_angle_rad = math.radians(s.laser_angle)  # beta
        plane_normal = np.array([-1, 0, math.tan(laser_angle_rad)])

        if s.verbose:
            print(f"frame {frame_number} | laser-angle {s.laser_angle}°")

        # read frame from video and resize
        (grabbed, frame) = camera.read()
        if grabbed is False:  # check if file is finished
            break

        # resize image
        frame = cv2.resize(frame, s.dims, interpolation=cv2.INTER_LINEAR)


        img = cv2.subtract(frame, old_frame)
        B, G, R = cv2.split(img.astype(np.float64))

        average = (R + B + G) / 2
        average = average.clip(max=255).astype(np.uint8)

        img = cv2.merge([average, average, average])


        # search frame for laserline, returns ndarray and preview image.
        # format: ndarray[height, 8]->[[x_2d,y_2d,x,y,z,r,g,b]..] with y_2d as index
        pointlist, preview_img = find_laser(img, channel=2, threshold=180, texture=texture)
        preview_img = cv2.resize(preview_img, s.preview_dims, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('preview', preview_img)

        # run through each row to triangulate a 3D point
        for y, values in enumerate(pointlist):
            x = values[0]
            if x < 0.5:  # skip lines without matches
                continue

            point3d = triangulate((x, y), topleft_corner, camera_pos, laser_pos, plane_normal)

            # add 3D coordinates to pointlist
            pointlist[y][2] = point3d[0]
            pointlist[y][3] = point3d[1] / s.vertical_stretch
            pointlist[y][4] = point3d[2] * -1

        # remove empty rows
        pointlist = pointlist[~np.all(pointlist == 0, axis=1)]

        # # append 3D points of this line to the global pointcloud
        # o3d.utility.Vector3dVector.extend(pointcloud.points, pointlist)
        pointcloud_frame.points = o3d.utility.Vector3dVector(pointlist[:, 2:5])
        pointcloud_frame.colors = o3d.utility.Vector3dVector(pointlist[:, 5:8] / 255)
        pointcloud += pointcloud_frame

        # update 3D-view each line
        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()

        # iterate frame_number and laser_angle
        frame_number += 1
        s.laser_angle += s.angle_step

        old_frame = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vis.destroy_window()

    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pointcloud = estimate_normals(pointcloud)
    export_pointcloud(pointcloud, s.export_path, type="ply")
    export_pointcloud(pointcloud, s.export_path, type="csv")
    export_pointcloud(pointcloud, s.export_path, type="pcd", write_ascii=True)
    print("export successful.")

    o3d.visualization.draw_geometries([pointcloud], width=800, height=800, left=1000,
                                      mesh_show_back_face=False, zoom=0.2, up=[0.0, 1.0, 0.0],
                                      front=[0.0, 0.0, 0.01], lookat=[0.0, 0.0, -1.0])




class Settings:
    """containing specific values"""
    def __init__(self, video_path="", verbose=True, export_path=None, texture_path=None,
                 shrink_x=1, shrink_y=8, shrink_preview=3):
        self.verbose = verbose
        self.video_path = video_path
        self.export_path = export_path

        self.texture_path = texture_path

        # load first frame of video and get width & height
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()
        input_height, input_width = frame.shape[:2]

        self.input_dims = (input_width, input_height)
        self.preview_width = int(input_width / shrink_preview)
        self.preview_height = int(input_height / shrink_preview)
        self.width = int(input_width / shrink_x)
        self.height = int(input_height / shrink_y)

        self.preview_dims = (self.preview_width, self.preview_height)
        self.dims = (self.width, self.height)

        # reduce vertical resolution
        self.vertical_stretch = (input_width / self.width) / (input_height / self.height)

        self.laser_angle = -28.  # initial laser angle
        self.angle_step = 0.5
        self.camera_laser_distance = 10  # cm Camera|Laser

        self.fov_degree = 48  # Camera horizontal Field of View
        self.fov_rad = math.radians(self.fov_degree)
        self.lens_length = self.dims[0] / (2 * math.tan(self.fov_rad / 2))


settings = Settings(video_path="../images/laser1a.mp4", export_path="../export/laser1a",
                    # texture_path="../images/_alt/laser1_rgb.jpg",
                    shrink_x=1, shrink_y=4, shrink_preview=3, verbose=False)

if __name__ == '__main__':
    main(settings)
    cv2.destroyAllWindows()
