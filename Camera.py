from pyrr import Vector3, matrix44, Matrix44, vector, vector3, matrix33, Vector4
from math import sin, cos, radians
import numpy as np
import cv2

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

#camera class to handle all movements and rotations

# https://github.com/KKeishiro/Shape-from-Silhouettes/blob/master/code_exercise/exercise7.m


class Camera:
    def __init__(
        self, 
        position: list, 
        target: list,
        camera_matrix = None,
        camera_rotation = None,
        camera_translation = None,
        camera_projection = None,
        focal_length: np.array = np.array([1.2,1.2]),
        display_width: int = 1200,
        display_height: int = 600,
        has_to_render: bool = True,
        has_to_render_image_plane: bool = True,
        has_to_render_axes: bool = True,
        speed: float = 0.15,
        rot_step: float = 0.025
    ):

        self.position = Vector3(position)
        self.target = Vector3(target)


        # PARAMETTERS FOR PERSPECTIVE PROJECTION
        self.display_width = display_width
        self.display_height = display_height
        self.aspect_ratio = float(self.display_width/self.display_height)
        self.principal_point = [float(self.display_width/2.), float(self.display_height/2.)]
        

        self.camera_matrix = camera_matrix
        self.camera_rotation = camera_rotation
        self.camera_translation = camera_translation
        self.camera_projection = camera_projection

        self.focal_length = focal_length
        self.compute_fov()

        
        self.z_axis = Vector3(vector3.normalize(self.position - self.target))
        self.up = Vector3([0.0, 1.0, 0.0])
        self.x_axis = Vector3(vector3.normalize(vector3.cross( self.up,self.z_axis)))
        self.y_axis = Vector3(vector3.cross(self.x_axis,self.z_axis))
        self.speed = speed
        self.rot_step = rot_step
        self.target = self.position -self.z_axis



        self.update_camera_vectors()
        self.set_projection_matrix()

        # FLAGS FOR CAM RENDERING
        self.has_to_render = has_to_render
        self.has_to_render_image_plane = has_to_render_image_plane
        self.has_to_render_axes = has_to_render_axes

    def compute_fov(self):
        self.fov = np.array([0.,0.])
        self.fov[0] = 2. * np.arctan(self.display_width/(2.*self.focal_length[0]))
        self.fov[1] = 2. * np.arctan(self.display_height/(2.*self.focal_length[1]))

    def get_view_matrix(self):
        return self.look_at_matrix(self.position, self.target, self.y_axis)
    
    def cam_lookat(self):     
        glu_lookat =  gluLookAt(self.position.x,  self.position.y,  self.position.z,  self.target.x,  self.target.y,  self.target.z,  self.y_axis.x,  self.y_axis.y,  self.y_axis.z)

    def get_view_matrix_focused_on(self, target):
        return self.look_at_matrix(self.position, target, self.y_axis)

    def get_local_vector(self,vector):
        view = self.look_at_matrix(self.position,self.target,self.y_axis)
        i_view = matrix44.inverse(view.T)
        rot_ = self.mult_mat_vec(i_view,vector)
        return rot_

    def move(self,delta):
        local_delta = self.get_local_vector(delta)
        self.position = self.position + local_delta
        self.target = self.target + local_delta
        self.update_camera_vectors()

    def rotate(self,angle, axis):
        rot_axis = self.get_local_vector(axis)
        vector.normalise(rot_axis)
        rotation_matrix = matrix33.create_from_axis_rotation(rot_axis, angle)

        self.z_axis = Vector3(vector3.normalise(self.mult_mat_vec(rotation_matrix, self.position - self.target)))
        self.target = self.position - self.z_axis
        self.update_camera_vectors()

    def update_camera_vectors(self):
        self.x_axis = Vector3(vector.normalise(vector3.cross(Vector3([0.0, 1.0, 0.0]),self.z_axis)))
        self.y_axis = Vector3(vector.normalise(vector3.cross( self.z_axis,self.x_axis)))

    def look_at_matrix(self, position, target, world_up):
        # 1.Position = known
        # 2.Calculate cameraDirection
        zaxis = vector.normalise(position - target)
        # 3.Get positive right axis vector
        xaxis = vector.normalise(vector3.cross(vector.normalise(world_up), zaxis))
        # 4.Calculate the camera up vector
        yaxis = vector3.cross(zaxis, xaxis)

        # create translation and rotation matrix
        translation = Matrix44.identity()
        translation[3][0] = -position.x
        translation[3][1] = -position.y
        translation[3][2] = -position.z

        rotation = Matrix44.identity()
        rotation[0][0] = xaxis[0]
        rotation[1][0] = xaxis[1]
        rotation[2][0] = xaxis[2]
        rotation[0][1] = yaxis[0]
        rotation[1][1] = yaxis[1]
        rotation[2][1] = yaxis[2]
        rotation[0][2] = zaxis[0]
        rotation[1][2] = zaxis[1]
        rotation[2][2] = zaxis[2]

        return  rotation * translation

    def mult_mat_vec(self,matrix,vector):
        result = np.empty((np.asarray(vector).shape))
        m = np.asarray(matrix)
        v = np.asarray(vector)
        for x in range(len(vector)):
            result[x] = np.sum(v*m[x,:len(vector)])
        return result

    def set_projection_matrix(self):
        self.projection_matrix =  matrix44.create_perspective_projection(self.fov, (self.display_width/self.display_width), 0.1, 500.0) 

    def get_projection_matrix(self):
        
        return self.projection_matrix

    def project_3d_point_into_image_plane(self, world_point: Vector3):

        point_in_camera_coordinates = self.get_view_matrix() * Vector4([world_point.x, world_point.y, world_point.z, 1])

        f = self.focal_length
        im_center = self.principal_point
        point_in_image_plane = [float((f[0]*point_in_camera_coordinates.x/ point_in_camera_coordinates.z) + im_center[0]),
                                float((f[1]*point_in_camera_coordinates.y/ point_in_camera_coordinates.z) + im_center[1])]

        return point_in_image_plane

    def project_3d_point_into_image_plane_opencv(self, world_points):
        image_points, _ = cv2.projectPoints(
            objectPoints = world_points.T.reshape(-1,1,3), 
            rvec = cv2.Rodrigues(self.camera_rotation)[0], 
            tvec = self.camera_translation, 
            cameraMatrix = self.camera_matrix.T,
            distCoeffs = np.float64([])
            )

        return image_points



    def render_image_plane(self):
        glBegin(GL_QUADS)

        ofset = Vector3(self.position-self.z_axis*self.focal_length[0])
        
        glColor3f(1.0,0.0,0.0)
        up_left_corner = ofset + self.x_axis*self.aspect_ratio + self.y_axis
        glVertex3fv(up_left_corner)
        glColor3f(0.0,1.0,0.0)
        up_right_corner = ofset - self.x_axis*self.aspect_ratio + self.y_axis
        glVertex3fv(up_right_corner)
        glColor3f(0.0,0.0,1.0)
        down_right_corner = ofset - self.x_axis*self.aspect_ratio - self.y_axis
        glVertex3fv(down_right_corner)
        glColor3f(1.0,1.0,1.0)
        down_left_corner = ofset + self.x_axis*self.aspect_ratio - self.y_axis
        glVertex3fv(down_left_corner)
        
        glEnd()

        # RENDER VISUAL FRUSTUM
        l1_proj_coordinates = np.cross(up_left_corner, self.position)
        l2_proj_coordinates = np.cross(up_right_corner, self.position)
        l3_proj_coordinates = np.cross(down_left_corner, self.position)
        l4_proj_coordinates = np.cross(down_right_corner, self.position)
        glBegin(GL_LINES)

        

        glColor3f(0.0,0.0,1.0)
        glVertex3fv(self.position)
        glVertex3fv(up_left_corner)
        glVertex3fv(self.position)
        glVertex3fv(up_right_corner)
        glVertex3fv(self.position)
        glVertex3fv(down_left_corner)
        glVertex3fv(self.position)
        glVertex3fv(down_right_corner)
        glEnd()

    def render_camera_position(self):
        glBegin(GL_POINTS)
        glColor3f(1.0,0.0,1.0)
        glVertex3fv(self.position) 
        glEnd()

    def render_axes(self):
    
        ofset = Vector3(self.position-self.z_axis*self.focal_length[0])
        #ofset = Vector3([1.0,1.0,1.0])
        glBegin(GL_LINES)
        # red X axis
        glColor3f(1.0,0.0,0.0)
        glVertex3fv(ofset)
        glVertex3fv(ofset + self.x_axis)
        # green Y axis
        glColor3f(0.0,1.0,0.0)
        glVertex3fv([ofset.x,ofset.y,ofset.z])
        glVertex3fv(ofset + self.y_axis)
        # blue Z axis
        glColor3f(0.0,0.0,1.0)
        glVertex3fv([ofset.x,ofset.y,ofset.z])
        glVertex3fv(ofset + self.z_axis)

        '''
        #plot camera front begining at 0.0 0.0 0.0
        glColor3f(1.0,1.0,1.0)
        glVertex3fv([.0,.0,.0])
        glVertex3fv(-self.z_axis)
        '''
    

        glEnd()

    def render_camera(self):
        if self.has_to_render_image_plane:
            self.render_image_plane()
        if self.has_to_render_axes:
            self.render_axes()
        self.render_camera_position()