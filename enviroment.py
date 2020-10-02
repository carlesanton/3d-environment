import os
import pygame
from pygame.locals import *
import numpy as np
import time
from Camera import Camera
from pyrr import Vector3, matrix44
import pickle
from typing import List
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from visuall_hull_extractor.cam_params_position import get_calibration_matrix_and_external_params
from visuall_hull_extractor.utils.visual_vull import VoxelGrid
from visuall_hull_extractor.utils.visual_vull import VHull
from visuall_hull_extractor.utils.Object import Object
from visuall_hull_extractor.utils.Object import ImagePlane
from visuall_hull_extractor.utils.Object import Scene
from Shaders import Shader
from Shaders import TextureShader
import cv2
from PIL import Image



default_calibartion_images_folder = os.path.join(os.getcwd(), 'visuall_hull_extractor', 'calibration_images')

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )
edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


world_size = 50.0
world_line_spacing = 4.0

figure_render_size = 10.0
figure_voxel_size = 0.5

current_camera_index = 0


ground_points = np.linspace(start = int(-world_size/2.0), stop=int(world_size/2.0), num=int(world_size/world_line_spacing), endpoint=True)
figure_space_points = np.linspace(start = int(-figure_render_size/2.0), stop=int(figure_render_size/2.0), num=int(figure_render_size/figure_voxel_size)+1, endpoint=True)

model_voxel_space = VoxelGrid(figure_space_points)

def create_cube_object(verticies, edges, color):
    vertices_list = []
    colors_list = []
    for edge in edges:
        for vertex in edge:
            vertices_list.append(verticies[vertex])
            colors_list.append(color)

    return Object(vertices=np.array(vertices_list,dtype=np.float32), colors=np.array(colors_list,dtype=np.float32))

def create_ground_object(ground_points, color):
    vertices_list = []
    colors_list = []
        
    max_pos = max(ground_points)
    min_pos = min(ground_points)
    # #############################################
    # ground lines
    for line in ground_points:
        # vertical lines
        vertices_list.append((line,0.,max_pos))
        colors_list.append(color)
        
        vertices_list.append((line,0.,min_pos))
        colors_list.append(color)
        # horizontal lines
        vertices_list.append((max_pos, 0., line))
        colors_list.append(color)
        
        vertices_list.append((min_pos,0.,line))
        colors_list.append(color)

        # #############################################
        # roof lines
        # vertical lines
        vertices_list.append((line,abs(max_pos*2),max_pos))
        colors_list.append(color)
        
        vertices_list.append((line,abs(max_pos*2),min_pos))
        colors_list.append(color)
        # horizontal lines
        vertices_list.append((max_pos,abs(max_pos*2), line))
        colors_list.append(color)
        
        vertices_list.append((min_pos,abs(max_pos*2),line))
        colors_list.append(color)

        # #############################################
        # front wall lines
        # vertical lines
        vertices_list.append((line,abs(max_pos*2),min_pos))
        colors_list.append(color)
        
        vertices_list.append((line,0.,min_pos))
        colors_list.append(color)
        # horizontal lines
        vertices_list.append((max_pos,line+max_pos, min_pos))
        colors_list.append(color)
        
        vertices_list.append((min_pos,line+max_pos,min_pos))
        colors_list.append(color)

        # #############################################
        # back wall lines
        # vertical lines
        vertices_list.append((line,abs(max_pos*2),max_pos))
        colors_list.append(color)
        
        vertices_list.append((line,0.,max_pos))
        colors_list.append(color)
        # horizontal lines
        vertices_list.append((max_pos,line+max_pos, max_pos))
        colors_list.append(color)
        
        vertices_list.append((min_pos,line+max_pos,max_pos))
        colors_list.append(color)

        # #############################################
        # right wall lines
        # vertical lines
        vertices_list.append((min_pos,abs(max_pos*2),line))
        colors_list.append(color)
        
        vertices_list.append((min_pos,0.,line))
        colors_list.append(color)
        # horizontal lines
        vertices_list.append((min_pos,line+max_pos, min_pos))
        colors_list.append(color)
        
        vertices_list.append((min_pos,line+max_pos, max_pos))
        colors_list.append(color)

        # #############################################
        # left wall lines
        # vertical lines
        vertices_list.append((max_pos,abs(max_pos*2),line))
        colors_list.append(color)
        
        vertices_list.append((max_pos, 0., line))
        colors_list.append(color)
        # horizontal lines
        vertices_list.append((max_pos,line+max_pos, min_pos))
        colors_list.append(color)
        
        vertices_list.append((max_pos,line+max_pos, max_pos))
        colors_list.append(color)

    return Object(vertices=np.array(vertices_list,dtype=np.float32), colors=np.array(colors_list,dtype=np.float32))
    
def create_axis_object(center: Vector3):
    vertices_list = []
    colors_list = []
    
    # red X axis
    vertices_list.append([0.0 + center.x,0.0 + center.y,0.0 + center.z])
    colors_list.append([1.,0.,0.])
    vertices_list.append([1.0 + center.x,0.0 + center.y,0.0 + center.z])
    colors_list.append([1.,0.,0.])
    # green Y axis
    vertices_list.append([0.0 + center.x,0.0 + center.y,0.0 + center.z])
    colors_list.append([0.,1.,0.])
    vertices_list.append([0.0 + center.x,1.0 + center.y,0.0 + center.z])
    colors_list.append([0.,1.,0.])
    # blue Z axis
    vertices_list.append([0.0 + center.x,0.0 + center.y,0.0 + center.z])
    colors_list.append([0.,0.,1.])
    vertices_list.append([0.0 + center.x,0.0 + center.y,1.0 + center.z])
    colors_list.append([0.,0.,1.])

    return Object(vertices=np.array(vertices_list,dtype=np.float32), colors=np.array(colors_list,dtype=np.float32))
    
def load_real_cameras(calibration_image_folder: str = default_calibartion_images_folder,force_recompute: bool = False):
    if force_recompute:
        cam_intrisic_parameters, extrinsic_parameters_list = get_calibration_matrix_and_external_params(calibration_image_folder)
    else:
        try:
            with open('camera_data/camera_parameters.pkl', 'rb') as handle:
                cam_intrisic_parameters = pickle.load(handle)
            with open('camera_data/external_parameters_list.pkl', 'rb') as handle:
                extrinsic_parameters_list = pickle.load(handle)
        except:
            print('Error loading camera parameters')
            cam_intrisic_parameters, extrinsic_parameters_list = get_calibration_matrix_and_external_params(calibration_image_folder)
    
    return cam_intrisic_parameters, extrinsic_parameters_list

def create_virtual_cameras_from_real_cameras(cam_intrisic_parameters, extrinsic_parameters_list, shader = None):
            
    camera_list: List[Camera] = []

    cam_width = 640
    cam_height = 480
    '''
    for extrinsic_parameters in extrinsic_parameters_list:
        focal_length = np.array([cam_intrisic_parameters[0,0],cam_intrisic_parameters[1,1]])
        cam_rotation, cam_translation, cam_projection = extrinsic_parameters
        cam_translation = cam_translation/100
        camera = Camera(position=cam_translation,
                        target = [.0,.0,.0],
                        camera_matrix = cam_intrisic_parameters.T,
                        camera_rotation = cam_rotation,
                        camera_translation = cam_translation,
                        camera_projection = cam_projection,
                        display_height=cam_height, 
                        display_width=cam_width,
                        speed=0.,
                        rot_step = 0.,
                        )
        camera_list.append(camera)
        print(f'camera translation: {cam_translation}')
    '''
    cube_shilouette = cv2.imread('/home/carles/repos/3d-environment/visuall_hull_extractor/calibration_images/square_mask.png', cv2.IMREAD_GRAYSCALE)
    cube_shilouette = cv2.imread('/home/carles/repos/3d-environment/visuall_hull_extractor/calibration_images/calibration_image_4.jpg', cv2.IMREAD_GRAYSCALE)
    
    camera = Camera(position=[0.,5.,20.],
                        target = [.0,5.,.0],
                        display_height=cam_height, 
                        display_width=cam_width,
                        speed=0.,
                        rot_step = 0.,
                        shilouette=cube_shilouette,
                        shader = shader,
                        )
    camera_list.append(camera)
    camera = Camera(position=[20.,5.,0.],
                        target = [.0,5.,.0],
                        display_height=cam_height, 
                        display_width=cam_width,
                        speed=0.,
                        rot_step = 0.,
                        shilouette=cube_shilouette,
                        shader = shader,
                        )
    camera_list.append(camera)

    return camera_list

def render_virtual_cameras(camera_list: List[Camera]):
    for camera in camera_list:
        if camera.has_to_render:
            camera.render_camera()

class Frame:
        
    def __init__(self):
        self.display_size = (640,480)

    def init_window(self):
        
        self.web_cam = cv2.VideoCapture(0)
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA) 
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(100, 100)
        wind = glutCreateWindow('Visual Hull')
        # set OpenGL callbacks
        # set display callbacks
        glutDisplayFunc(self.gl_loop_callback)
        glutIdleFunc(self.gl_loop_callback)
        # set keyboard callbacks
        glutKeyboardFunc(self.process_keyboard) # for normal keys
        glutSpecialFunc(self.process_special_keys) # for special keys (as arrow keys) 
        '''
        # set mouse callbacks
        glutMotionFunc(self.proces_mouse_movement)
        glutPassiveMotionFunc(self.proces_mouse_pos)
        glutMouseFunc(self.proces_mouse_pressing)
        '''
        # create vhull shader program
        vhull_vertex_shader = open('./visuall_hull_extractor/shaders/vhull_vertex_shader.glsl','r').read()
        vhull_fragment_shader = open('./visuall_hull_extractor/shaders/vhull_fragment_shader.glsl','r').read()
        self.vhull_shader = Shader(vertex_shader_source = vhull_vertex_shader, fragment_shader_source = vhull_fragment_shader)
        # create regular shader program
        regular_vertex_shader = open('./visuall_hull_extractor/shaders/regular_vertex_shader.glsl','r').read()
        regular_fragment_shader = open('./visuall_hull_extractor/shaders/regular_fragment_shader.glsl','r').read()
        self.regular_shader = Shader(vertex_shader_source = regular_vertex_shader, fragment_shader_source = regular_fragment_shader)
        
        # create textured shader program
        regular_vertex_shader = open('./visuall_hull_extractor/shaders/texture_vertex_shader.glsl','r').read()
        regular_fragment_shader = open('./visuall_hull_extractor/shaders/texture_fragment_shader.glsl','r').read()
        self.texture_shader = TextureShader(vertex_shader_source = regular_vertex_shader, fragment_shader_source = regular_fragment_shader)

        self.cam_intrinsc_params, self.cam_extrinsic_prmtrs_lst = load_real_cameras(force_recompute = True)
        self.camera_list = create_virtual_cameras_from_real_cameras(self.cam_intrinsc_params, self.cam_extrinsic_prmtrs_lst, shader = self.regular_shader)
        self.camera_list.append(Camera(position=[0.0,world_size,-world_size/2], target=[0.0,world_size/2,world_size/2], has_to_render_image_plane=False))
        self.current_camera_index = len(self.camera_list) - 1


        self.vhull_vertex = model_voxel_space.get_voxel_centers_as_np_array()
        # self.vhull_vertex = np.array([-0.5,-0.5,0.0,0.5,0.5,-0.5],dtype=np.float32)
        np.random.shuffle(self.vhull_vertex)
        # load vhull shaders code
        
        self.scene = Scene()
        # create VHULL object
        self.visual_hull = VHull(vhull_vertex = self.vhull_vertex, shader = self.vhull_shader, modeling_cameras = self.camera_list[0:2])

        # Create Cube object
        self.scene.add_object('cube', create_cube_object(vertices,edges, color=[1.,1.,1.]))
        self.scene.object_dict['cube'].shader = self.regular_shader
        self.scene.object_dict['cube'].init_gl_vertex_and_color_buffers()
        
        # Create Image object
        corners = np.array([[world_size/2, world_size, world_size/2],
                        [-world_size/2, world_size, world_size/2],
                        [-world_size/2, 0, world_size/2],
                        [world_size/2, 0, world_size/2]],
                        dtype=np.float32)
        self.new_image = Image.open('/home/carles/repos/3d-environment/visuall_hull_extractor/calibration_images/square_mask.png')
        self.scene.add_object(
            'image_plane', 
            ImagePlane(
                corners,
                self.visual_hull.cube_shilouette_image,
                rendering_primitive=GL_QUADS, 
                shader = self.texture_shader
            )
        )
        
        # Create Ground object
        self.scene.add_object('ground', create_ground_object(ground_points, color=[1.,1.,1.]))
        self.scene.object_dict['ground'].shader = self.regular_shader
        self.scene.object_dict['ground'].init_gl_vertex_and_color_buffers()
        
        # Create Model Box object
        #self.scene.add_object('model_box', create_ground_object(figure_space_points, color=[1.,0.,1.]))
        #self.scene.object_dict['model_box'].shader = self.regular_shader
        #self.scene.object_dict['model_box'].init_gl_vertex_and_color_buffers()

        # Create Axis object
        self.scene.add_object('center_axis', create_axis_object(center = Vector3([0.,0.,0.])))
        self.scene.object_dict['center_axis'].shader = self.regular_shader
        self.scene.object_dict['center_axis'].init_gl_vertex_and_color_buffers()



        self.mouse_pos_x = 0
        self.mouse_pos_y = 0

        self.ttt = time.time()
        self.is_mouse_down = False
        self.background_show = True
        self.show_time = True
        self.print_pos = False
        self.print_modelview = False
        #store initial mouse position
        # (mouse_pos_x,mouse_pos_y) = pygame.mouse.get_pos()

        glutMainLoop()

    def proces_mouse_movement(self, current_mouse_pos_x, current_mouse_pos_y):
        # get mouse delta (do after pump events)
        mouse_delta_x = (self.mouse_pos_x-current_mouse_pos_x)
        mouse_delta_y = (self.mouse_pos_y-current_mouse_pos_y)
        
        #print(str(mouse_delta_x) + ' ' + str(mouse_delta_y))
        #store new position

        # si el boto esquerra està clicat
        if self.is_mouse_down:
            # rotate the pertinent amount in each axis
            self.camera_list[self.current_camera_index].rotate(-mouse_delta_x*self.camera_list[self.current_camera_index].rot_step, Vector3([.0,1.0,.0]))
            self.camera_list[self.current_camera_index].rotate(-mouse_delta_y*self.camera_list[self.current_camera_index].rot_step, Vector3([1.0,.0,.0]))

        self.mouse_pos_x = current_mouse_pos_x
        self.mouse_pos_y = current_mouse_pos_y

    def proces_mouse_pos(self, current_mouse_pos_x, current_mouse_pos_y):
        self.mouse_pos_x = current_mouse_pos_x
        self.mouse_pos_y = current_mouse_pos_y

    def proces_mouse_pressing(self, button, state, mouse_pos_x, mouse_pos_y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self.is_mouse_down = True
        elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
            self.is_mouse_down = False

    def process_special_keys(self, key, mouse_x, mouse_y):        
        if key == GLUT_KEY_LEFT:
            direction_vector = Vector3([-1.,.0,.0])
            self.camera_list[self.current_camera_index].move(direction_vector*self.camera_list[self.current_camera_index].speed)        
        elif key == GLUT_KEY_RIGHT:
            direction_vector = Vector3([1.,.0,.0])
            self.camera_list[self.current_camera_index].move(direction_vector*self.camera_list[self.current_camera_index].speed)        
        elif key == GLUT_KEY_UP:
            direction_vector = Vector3([.0,.0,-1.])
            self.camera_list[self.current_camera_index].move(direction_vector*self.camera_list[self.current_camera_index].speed)        
        elif key == GLUT_KEY_DOWN:
            direction_vector = Vector3([.0,.0,1.])
            self.camera_list[self.current_camera_index].move(direction_vector*self.camera_list[self.current_camera_index].speed)
        elif key == 112: ## LEFT SHIFT, I dont find the GLUT_KEY_* for it
            self.camera_list[self.current_camera_index].speed = self.camera_list[self.current_camera_index].speed*4.
        elif key == 113: ## RIGHT SHIFT, I dont find the GLUT_KEY_* for it
            self.camera_list[self.current_camera_index].speed = self.camera_list[self.current_camera_index].speed/4.

    def process_keyboard(self, key, mouse_x, mouse_y):
        key = key.decode("utf-8").lower()
        
        if key == 'a':
            self.camera_list[self.current_camera_index].rotate(-self.camera_list[self.current_camera_index].rot_step,Vector3([0.0,1.,.0]))
        elif key == 's':
            self.camera_list[self.current_camera_index].rotate(self.camera_list[self.current_camera_index].rot_step,Vector3([1.0,0.0,.0]))
        elif key == 'd':
            self.camera_list[self.current_camera_index].rotate(self.camera_list[self.current_camera_index].rot_step,Vector3([0.0,1.0,.0]))
        elif key == 'w':
            self.camera_list[self.current_camera_index].rotate(-self.camera_list[self.current_camera_index].rot_step,Vector3([1.0,0.0,.0]))

        elif key == '+':
            self.current_camera_index += 1
            if self.current_camera_index >= len(self.camera_list):
                self.current_camera_index = 0
        elif key == '-':
            self.current_camera_index -= 1
            if self.current_camera_index < 0:
                self.current_camera_index = len(self.camera_list) - 1

        elif key == ' ':
            self.background_show = not self.background_show

    def gl_loop_callback(self):    
        '''
        Function executed every time OpenGL needs to redraw the window.
        Binded with:
            glutDisplayFunc(self.gl_loop_callback)
            glutIdleFunc(self.gl_loop_callback)
        methods in the self.init_window() process.
        glutDisplayFunc() sets the callback for everytime the window is resized or opened or smthing like this
        glutIdleFunc() sets the callback to run in loop even when no events must be processed.
                       from OpenGL documentation: 
                            sets the global idle callback to be func so a GLUT program can perform background processing
                            tasks or continuous animation when window system events are not being received. If enabled,
                            the idle callback is continuously called when events are not being received
        '''

        self.update_gl_matrices()
        '''
        new_image = Image.fromarray(self.web_cam.read()[1])
        self.scene.object_dict['image_plane'].update_texture_image_2(new_image)
        '''
        


        if self.print_modelview:
            self.print_gl_and_computed_view_matrices()
        if self.print_pos:
            self.print_camera_position()

        self.render_scene()
        self.check_cam_boundaries()

        if self.show_time:
            self.compute_fps()

    def render_scene(self):
        # clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # render each element
        self.scene.render()
        self.visual_hull.short_render(viewing_camera = self.camera_list[-1], modeling_cameras = self.camera_list[0:-1])
        render_virtual_cameras(self.camera_list)
        # swap buffers i dont actualy know what this does but it seems necessary for the pipeline
        glutSwapBuffers()

    def check_cam_boundaries(self):
   
        if self.camera_list[self.current_camera_index].speed > 0:
            # X boundaries
            if self.camera_list[self.current_camera_index].position.x<-world_size/2:
                self.camera_list[self.current_camera_index].position.x = -world_size/2
            if self.camera_list[self.current_camera_index].position.x>world_size/2:
                self.camera_list[self.current_camera_index].position.x = world_size/2

            # Y boundaries
            if self.camera_list[self.current_camera_index].position.y<0.5:
                self.camera_list[self.current_camera_index].position.y = 0.5
            if self.camera_list[self.current_camera_index].position.y>world_size:
                self.camera_list[self.current_camera_index].position.y = world_size

            # Z boundaries
            if self.camera_list[self.current_camera_index].position.z<-world_size/2:
                self.camera_list[self.current_camera_index].position.z = -world_size/2
            if self.camera_list[self.current_camera_index].position.z>world_size/2:
                self.camera_list[self.current_camera_index].position.z = world_size/2


            self.camera_list[self.current_camera_index].target = self.camera_list[self.current_camera_index].position - self.camera_list[self.current_camera_index].z_axis
            self.camera_list[self.current_camera_index].update_camera_vectors()

    def update_gl_matrices(self):
        # set projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50, self.camera_list[self.current_camera_index].aspect_ratio, 0.1, 100.0)
        
        # set modelview matrix (actualy just view matrix  model view is an identity matrix)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.camera_list[self.current_camera_index].cam_lookat()

    def compute_fps(self):
        d_t = time.time() - self.ttt
        frame_rate = 1/(d_t)
        print(f'Frame rate: {frame_rate} fps')
        self.ttt = time.time()

    def get_gl_view_matrix(self):
        a = (GLfloat * 16)()
        mvm = glGetFloatv(GL_MODELVIEW_MATRIX, a)
        gl_view_matrix = np.asarray(list(a)).reshape((4,4))
        return gl_view_matrix

    def print_gl_and_computed_view_matrices(self):
        gl_view_matrix = self.get_gl_view_matrix()
        print('GL look at matrix: ')
        print(gl_view_matrix)
        print('--------------')
        print('Computed look at matrix: ')
        print(self.camera_list[self.current_camera_index].get_view_matrix())
    
    def print_camera_position(self):
        print('Camera position: ' + str(self.camera_list[self.current_camera_index].position))
        print('----------------------------')

def main():
    frame = Frame()
    frame.init_window()


if __name__ == "__main__":
    main()
    
