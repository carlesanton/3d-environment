from pyrr import Vector3, matrix44, Matrix44, vector, vector3, matrix33, Vector4
from math import sin, cos, radians
import numpy as np
import cv2

from typing import Tuple

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
        speed: float = 0.5,
        rot_step: float = 0.025,
        shilouette = None,
        shader = None
    ):

        self.position = Vector3(position)
        self.target = Vector3(target)

        # shilouette for space carving
        self.shilouette = shilouette
        if shilouette is None:
            self.shilouette = np.zeros((display_width, display_height))

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

        self.shader = shader
        if self.shader:
            self.init_gl_vertex_buffers()


    def compute_fov(self):
        self.fov = np.array([0.,0.])
        self.fov[0] = 2. * np.arctan(self.display_width/(2.*self.focal_length[0]))
        self.fov[1] = 2. * np.arctan(self.display_height/(2.*self.focal_length[1]))
        self.fov = [50.,50.*self.aspect_ratio]

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
        self.projection_matrix =  matrix44.create_perspective_projection(self.fov[0], self.aspect_ratio, 0.1, 500.0) 

    def get_projection_matrix(self):
        return self.projection_matrix

    def project_3d_point_into_image_plane(self, world_point: Vector3) -> Tuple[float]:

        point_in_camera_coordinates = self.get_view_matrix() * Vector4([world_point.x, world_point.y, world_point.z, 1])

        f = self.focal_length
        im_center = self.principal_point
        point_in_image_plane = (float((f[0]*point_in_camera_coordinates.x/ point_in_camera_coordinates.z) + im_center[0]),
                                float((f[1]*point_in_camera_coordinates.y/ point_in_camera_coordinates.z) + im_center[1]),
                                )

        return point_in_image_plane

    def project_3d_point_into_image_plane_opencv(self, world_points) -> Tuple[float]:
        image_points, _ = cv2.projectPoints(
            objectPoints = world_points.T.reshape(-1,1,3), 
            rvec = cv2.Rodrigues(self.camera_rotation)[0], 
            tvec = self.camera_translation, 
            cameraMatrix = self.camera_matrix.T,
            distCoeffs = np.float64([])
            )

        return image_points

    def check_point_in_silhouette(self, image_point: Tuple[float]) -> bool:
        if self.shilouette[int(image_point[1])-1,int(image_point[0])-1]:
            return True
        else:
            return False

    def init_gl_vertex_buffers(self):

        # create buffers for camera image plane vertices
        self.init_gl_image_plane_buffers()
        
        self.init_gl_frustrum_buffers()

        # create buffers for camera axis
        self.init_gl_axis_buffers()

    def init_gl_image_plane_buffers(self):
        ofset = Vector3(self.position-self.z_axis*self.focal_length[0])
        up_left_corner = ofset + self.x_axis*self.aspect_ratio + self.y_axis
        up_right_corner = ofset - self.x_axis*self.aspect_ratio + self.y_axis
        down_right_corner = ofset - self.x_axis*self.aspect_ratio - self.y_axis
        down_left_corner = ofset + self.x_axis*self.aspect_ratio - self.y_axis
        
        # create vertices and colors array
        self.image_plane_vertices = np.array(np.hstack((up_left_corner, up_right_corner, down_right_corner, down_left_corner)),dtype = np.float32) 
        self.image_plane_colors = np.array(np.hstack(([1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0], [1.0,1.0,1.0])),dtype = np.float32) 

        self.image_plane_primitive = GL_QUADS
        self.vertex_size = 3

        # Create VAO
        self.image_plane_vao = self.shader.init_gl_vertex_and_color_buffers_into_vao(self.image_plane_vertices, self.image_plane_colors, self.vertex_size)

    def init_gl_frustrum_buffers(self):
        ofset = Vector3(self.position-self.z_axis*self.focal_length[0])
        up_left_corner = ofset + self.x_axis*self.aspect_ratio + self.y_axis
        up_right_corner = ofset - self.x_axis*self.aspect_ratio + self.y_axis
        down_right_corner = ofset - self.x_axis*self.aspect_ratio - self.y_axis
        down_left_corner = ofset + self.x_axis*self.aspect_ratio - self.y_axis
    
        # create vertices and colors array
        self.frustrum_vertices = np.array(np.hstack((
                            self.position,
                            up_left_corner,
                            self.position, 
                            up_right_corner,
                            self.position, 
                            down_left_corner,
                            self.position, 
                            down_right_corner, 
                        )),dtype = np.float32) 
        self.frustrum_colors = np.array(np.hstack(([0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0])),dtype = np.float32) 

        self.frustrum_primitive = GL_LINES
        self.vertex_size = 3

        # Create VAO
        self.frustrum_vao = self.shader.init_gl_vertex_and_color_buffers_into_vao(self.frustrum_vertices, self.frustrum_colors, self.vertex_size)
    
    def init_gl_axis_buffers(self):
        ofset = Vector3(self.position-self.z_axis*self.focal_length[0])

        # create vertices and colors array
        self.axis_vertices = np.array(np.hstack((
                            ofset,
                            ofset + self.x_axis,
                            [ofset.x,ofset.y,ofset.z],
                            ofset + self.y_axis,
                            [ofset.x,ofset.y,ofset.z],
                            ofset + self.z_axis,
 
                        )),dtype = np.float32) 
        self.axis_colors = np.array(np.hstack((
                            [1.0,0.0,0.0], 
                            [1.0,0.0,0.0], 
                            [0.0,1.0,0.0], 
                            [0.0,1.0,0.0], 
                            [0.0,0.0,1.0], 
                            [0.0,0.0,1.0], 
                            )),dtype = np.float32) 
        
        self.axis_primitive = GL_LINES
        self.vertex_size = 3

        # Create VAO
        self.axis_vao = self.shader.init_gl_vertex_and_color_buffers_into_vao(self.axis_vertices, self.axis_colors, self.vertex_size)

    def render_image_plane(self):

        self.shader.enable()
        # render image plane
        glBindVertexArray(self.image_plane_vao)
        glDrawArrays(self.image_plane_primitive, 0, int(len(self.image_plane_vertices)/self.vertex_size))
        glBindVertexArray(0)

        # render frustrum
        glBindVertexArray(self.frustrum_vao)
        glDrawArrays(self.frustrum_primitive, 0, int(len(self.frustrum_vertices)/self.vertex_size))
        glBindVertexArray(0)
        self.shader.disable()

    def render_camera_position(self):
        glBegin(GL_POINTS)
        glColor3f(1.0,0.0,1.0)
        glVertex3fv(self.position) 
        glEnd()

    def render_axes(self):
        if self.shader:
            self.shader.enable()
            glBindVertexArray(self.axis_vao)
            glDrawArrays(self.axis_primitive, 0, int(len(self.axis_vertices)/self.vertex_size))
            glBindVertexArray(0)
            self.shader.disable()

        else:
            ofset = Vector3(self.position-self.z_axis*self.focal_length[0])
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
            glEnd()
            



    def render_camera(self):
        if self.has_to_render_image_plane:
            self.render_image_plane()
        if self.has_to_render_axes:
            self.render_axes()
        self.render_camera_position()