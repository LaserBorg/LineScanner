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

from lib.pointcloud import set_verbosity, estimate_point_normals, export_pointcloud
from lib.visualization import init_visualizer, update_visualizer, visualize
from lib.image import find_laser, subtract_images  # , rotate_bound

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
        # self.sweep_direction    = configs['sweep_direction']          # TODO: unused

        self.sweep_step         = configs['sweep_step']
        self.laser_thres        = configs['laser_thres']
        
        self.shrink_x           = configs['shrink_x']
        self.shrink_y           = configs['shrink_y']
        self.shrink_preview     = configs['shrink_preview']
        self.window_size        = configs['window_size']
        self.verbose            = configs['verbose']

        self.sweep_angle        = configs['sweep_startangle']
        self.frame_index        = 0
        self.cap                = cv2.VideoCapture(self.video_path)     # load frame 0 to get dimensions,
        self.iterate_frame()                                            # then increment sweep_angle and frame_index 

        ret, frame              = self.cap.read()
        source_h, source_w      = frame.shape[:2]
        self.input_dims         = (source_w, source_h)
        self.preview_width      = int(source_w / self.shrink_preview)
        self.preview_height     = int(source_h / self.shrink_preview)
        self.width              = int(source_w / self.shrink_x)
        self.height             = int(source_h / self.shrink_y)
        self.dims               = (self.width, self.height)
        self.preview_dims       = (self.preview_width, self.preview_height)
        self.zero_value         = configs['zero_value']
        self.KDTree_radius      = configs['KDTree_radius']
        self.KDTree_max_nn      = configs['KDTree_max_nn']

        # load image texture
        texpath                 = configs['texture_path'] 
        self.texture            = cv2.resize(cv2.imread(texpath, 1), self.dims, interpolation=cv2.INTER_LINEAR) if texpath != "" else None
        self.desaturate         = configs['desaturate']

        # reduce vertical resolution
        self.vertical_stretch   = (source_w / self.width) / (source_h / self.height)

        self.fov_rad            = math.radians(self.hfov)
        self.lens_length        = self.dims[0] / (2 * math.tan(self.fov_rad / 2))

        # project imageplane into 3d space
        self.topleft_corner     = np.array([-self.dims[0] / 2, self.dims[1] / 2, self.lens_length])

        # init interactive 3D-Viewer
        self.vis = init_visualizer(width=self.window_size[0], height=self.window_size[1])

        self.pointcloud = o3d.geometry.PointCloud()
        self.pointcloud_frame = o3d.geometry.PointCloud()

        ss = 50  # scenesize  TODO: scale seems to be sensitive; will require scale-to-fit and wider clipping plane
        self.pointcloud.points = o3d.utility.Vector3dVector(np.array([[-ss, -ss, -ss], [ss, ss, ss]]))
        self.pointcloud.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [1, 0, 0]]))
        self.vis.add_geometry(self.pointcloud)

    def triangulate(self, pixel, plane_normal):
        # Pixel vector relative to image topleft_corner point
        rayDirection = np.array([pixel[0] + self.topleft_corner[0], self.topleft_corner[1] - pixel[1], self.topleft_corner[2]])

        dotProduct = plane_normal.dot(rayDirection)

        # check if parallel or in-plane
        if abs(dotProduct) < self.zero_value:
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
        previous_frame = self.init_framebuffer()

        # MAIN LOOP
        while True:
            # calculate current normal-vector of laserplane
            plane_normal = self.calculate_plane_normal()
            if self.verbose:
                print(f"frame {self.frame_index} | laser-angle {self.sweep_angle}°")
            
            difference_map, frame = self.read_and_resize_frame(previous_frame)
            if difference_map is None:
                break

            pointlist = self.process_frame(difference_map, frame, plane_normal)
            self.append_points(pointlist)

            # update preview for each frame
            update_visualizer(self.vis, self.pointcloud)

            self.iterate_frame()
            previous_frame = frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.vis.destroy_window()
        cv2.destroyAllWindows()

        # calculate point normals
        self.pointcloud = estimate_point_normals(self.pointcloud, radius=self.KDTree_radius, max_nn=self.KDTree_max_nn)
        
        # export PCD, PLY or CSV model
        export_pointcloud(self.pointcloud, self.export_path, type=self.export_type, write_ascii=True)
        print("export successful.")

        # display static scene
        visualize([self.pointcloud], width=self.window_size[0], height=self.window_size[1], left=1000)

    def init_framebuffer(self):
        return np.zeros((self.dims[1], self.dims[0], 3), np.uint8)

    def calculate_plane_normal(self):
        laser_angle_rad = math.radians(self.sweep_angle)  # beta
        return np.array([-1, 0, math.tan(laser_angle_rad)])

    def read_and_resize_frame(self, previous_frame):
        (grabbed, frame) = self.cap.read()
        if grabbed is False:  # check if file is finished
            return None, previous_frame
        frame = cv2.resize(frame, self.dims, interpolation=cv2.INTER_LINEAR)
        difference_map = subtract_images(frame, previous_frame, return_RGB=True)
        return difference_map, frame

    def process_frame(self, difference_map, frame, plane_normal):
        # use texture if available
        tex = frame if self.texture is None else self.texture

        # search frame for laserline, returns ndarray and preview image.
        # format: ndarray[height, 8]->[[x_2d,y_2d,x,y,z,r,g,b]..] with y_2d as index
        pointlist, preview_img = find_laser(difference_map, channel=2, threshold=self.laser_thres, texture=tex, desaturate=self.desaturate)
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
        return pointlist

    def append_points(self, pointlist):
        def to_vector3d(data):
            return o3d.utility.Vector3dVector(data)

        # Update points and colors
        self.pointcloud_frame.points = to_vector3d(pointlist[:, 2:5])
        self.pointcloud_frame.colors = to_vector3d(pointlist[:, 5:8] / 255)
        self.pointcloud += self.pointcloud_frame

    def iterate_frame(self):
        self.frame_index += 1
        self.sweep_angle += self.sweep_step


if __name__ == '__main__':
    set_verbosity()
    
    # config = 'images/laser1a_2048_config.json'
    config = 'images/laser1a_720_config.json'
    # config = 'images/laser1b_720_config.json'

    linescanner = LineScanner(config)
    linescanner.scan()
