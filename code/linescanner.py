"""
░░      ░░ ░░░    ░░ ░░░░░░░ ░░░░░░░  ░░░░░░  ░░░░░  ░░░    ░░ ░░░    ░░ ░░░░░░░ ░░░░░░
▒▒      ▒▒ ▒▒▒▒   ▒▒ ▒▒      ▒▒      ▒▒      ▒▒   ▒▒ ▒▒▒▒   ▒▒ ▒▒▒▒   ▒▒ ▒▒      ▒▒   ▒▒
▒▒      ▒▒ ▒▒ ▒▒  ▒▒ ▒▒▒▒▒   ▒▒▒▒▒▒▒ ▒▒      ▒▒▒▒▒▒▒ ▒▒ ▒▒  ▒▒ ▒▒ ▒▒  ▒▒ ▒▒▒▒▒   ▒▒▒▒▒▒
▓▓      ▓▓ ▓▓  ▓▓ ▓▓ ▓▓           ▓▓ ▓▓      ▓▓   ▓▓ ▓▓  ▓▓ ▓▓ ▓▓  ▓▓ ▓▓ ▓▓      ▓▓   ▓▓
███████ ██ ██   ████ ███████ ███████  ██████ ██   ██ ██   ████ ██   ████ ███████ ██   ██

"""
import numpy as np
import cv2
import math
import open3d as o3d
import json

from libs.laser import find_laser
from libs.pointcloud import export_pointcloud
# from libs.image import rotate_bound


class LineScanner:
    def __init__(self, config_path):
        # Load settings from JSON file
        with open(config_path, 'r') as f:
            configs             = json.load(f)
        
        self.video_path         = configs['video_path']
        self.export_path        = configs['export_path']
        self.export_type        = configs['export_type']
        self.cam_pos            = np.array(configs['cam_pos']) 
        self.laser_pos          = np.array(configs['laser_pos'])
        self.hfov               = configs['horizontal_fov']
        self.laser_angle        = configs['laser_angle_start']
        self.angle_step         = configs['angle_step']
        self.laser_thres        = configs['laser_thres']
        
        self.desaturate_texture = configs['desaturate_texture']
        
        self.shrink_x           = configs['shrink_x']
        self.shrink_y           = configs['shrink_y']
        self.shrink_preview     = configs['shrink_preview']
        self.window_size        = configs['window_size']
        self.verbose            = configs['verbose']
        
        # load first frame of video and get width & height
        self.cap                = cv2.VideoCapture(self.video_path)
        ret, frame              = self.cap.read()
        source_h, source_w      = frame.shape[:2]

        self.input_dims         = (source_w, source_h)
        self.preview_width      = int(source_w / self.shrink_preview)
        self.preview_height     = int(source_h / self.shrink_preview)
        self.width              = int(source_w / self.shrink_x)
        self.height             = int(source_h / self.shrink_y)
        self.dims               = (self.width, self.height)
        self.preview_dims       = (self.preview_width, self.preview_height)
        self.zero               = configs['zero']
        self.KDTree_radius      = configs['KDTree_radius']
        self.KDTree_max_nn      = configs['KDTree_max_nn']

        # load image texture
        texpath                 = configs['texture_path'] 
        self.texture = cv2.resize(cv2.imread(texpath, 1), self.dims, interpolation=cv2.INTER_LINEAR) if texpath != "" else None

        # reduce vertical resolution
        self.vertical_stretch   = (source_w / self.width) / (source_h / self.height)

        self.fov_rad            = math.radians(self.hfov)
        self.lens_length        = self.dims[0] / (2 * math.tan(self.fov_rad / 2))

        # project imageplane into 3d space
        self.topleft_corner = np.array([-self.dims[0] / 2, self.dims[1] / 2, self.lens_length])

        # init live 3D-Viewer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.window_size[0], height=self.window_size[1], left=1000)
        self.pointcloud = o3d.geometry.PointCloud()
        self.pointcloud_frame = o3d.geometry.PointCloud()

        # TODO: initial scale is crucial; needs scale-to-fit and  wider clipping plane
        self.pointcloud.points = o3d.utility.Vector3dVector(np.array([[-50, -50, -50], [50, 50, 50]]))
        self.pointcloud.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [1, 0, 0]]))
        self.vis.add_geometry(self.pointcloud)


    def triangulate(self, pixel, plane_normal):
        # Pixel vector relative to image topleft_corner point
        rayDirection = np.array([pixel[0] + self.topleft_corner[0], self.topleft_corner[1] - pixel[1], self.topleft_corner[2]])

        dotProduct = plane_normal.dot(rayDirection)

        # check if parallel or in-plane
        if abs(dotProduct) < self.zero:
            print("[WARNING] no intersection at line", pixel[1])
            return np.array([0, 0, 0])
        else:
            w = self.cam_pos - self.laser_pos
            si = -plane_normal.dot(w) / dotProduct
            intersection = w + si * rayDirection + self.laser_pos

            if intersection[2] > 0:
                return intersection
            # print("[WARNING] intersection behind camera")
            return np.array([0, 0, 0])


    # def sort_numpy_by_column(array, column=0):
    #     return array[array[:, column].argsort()]


    def scan(self):
        # init framebuffer for difference-map
        old_frame = np.zeros((self.dims[1], self.dims[0], 3), np.uint8)

        # VIDEO PROCESSING LOOP
        frame_number = 0
        while True:
            # calculate current normal-vector of laserplane
            laser_angle_rad = math.radians(self.laser_angle)  # beta
            plane_normal = np.array([-1, 0, math.tan(laser_angle_rad)])

            if self.verbose:
                print(f"frame {frame_number} | laser-angle {self.laser_angle}°")

            # read frame from video and resize
            (grabbed, frame) = self.cap.read()
            if grabbed is False:  # check if file is finished
                break

            # resize image
            frame = cv2.resize(frame, self.dims, interpolation=cv2.INTER_LINEAR)

            difference_map = self.subtract_images(frame, old_frame)

            # use texture if available
            tex = frame if self.texture is None else self.texture

            # search frame for laserline, returns ndarray and preview image.
            # format: ndarray[height, 8]->[[x_2d,y_2d,x,y,z,r,g,b]..] with y_2d as index
            pointlist, preview_img = find_laser(difference_map, channel=2, threshold=self.laser_thres, texture=tex, desaturate_texture=self.desaturate_texture)  # texture=texture

            preview_img = cv2.resize(preview_img, self.preview_dims, interpolation=cv2.INTER_NEAREST)
            cv2.imshow('preview', preview_img)

            # run through each row to triangulate a 3D point
            for y, values in enumerate(pointlist):
                x = values[0]
                if x < 0.5:  # skip lines without matches
                    continue

                point3d = self.triangulate((x, y), plane_normal)

                # add 3D coordinates to pointlist
                pointlist[y][2] = point3d[0]
                pointlist[y][3] = point3d[1] / self.vertical_stretch
                pointlist[y][4] = point3d[2] * -1

            # remove empty rows
            pointlist = pointlist[~np.all(pointlist == 0, axis=1)]

            # # append 3D points of this line to the global pointcloud
            # o3d.utility.Vector3dVector.extend(pointcloud.points, pointlist)
            self.pointcloud_frame.points = o3d.utility.Vector3dVector(pointlist[:, 2:5])
            self.pointcloud_frame.colors = o3d.utility.Vector3dVector(pointlist[:, 5:8] / 255)
            self.pointcloud += self.pointcloud_frame

            # update 3D-view each line
            self.vis.update_geometry(self.pointcloud)
            self.vis.poll_events()
            self.vis.update_renderer()

            # iterate frame_number and laser_angle
            frame_number += 1
            self.laser_angle += self.angle_step

            old_frame = frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.vis.destroy_window()

        # calculate point normals
        # TODO: they are all facing the camera ?!
        self.pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.KDTree_radius, max_nn=self.KDTree_max_nn))
        # pointcloud = estimate_normals(pointcloud)

        # EXPORT PLY, CSV OR PCD MODEL
        # TODO: exports do not overwrite files -> need to delete if existing
        export_pointcloud(self.pointcloud, self.export_path, type=self.export_type, write_ascii=True)  # ply, csv, pcd
        print("export successful.")

        # VISUALIZE
        o3d.visualization.draw_geometries([self.pointcloud], width=self.window_size[0], height=self.window_size[1])


    @staticmethod
    def subtract_images(current_frame, previous_frame):
        img = cv2.subtract(current_frame, previous_frame).astype(np.float64)
        average = np.mean(img, axis=2).clip(max=255).astype(np.uint8)
        return cv2.merge([average, average, average])

 
if __name__ == '__main__':
    linescanner = LineScanner('config.json')
    linescanner.scan()
    cv2.destroyAllWindows()
