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
from visuall_hull_extractor.utils.visual_vull import VHullShader
import cv2



default_calibartion_images_folder = os.path.join(os.getcwd(), 'visuall_hull_extractor', 'calibration_images')

verticies = (
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
figure_voxel_size = 2

current_camera_index = -1


ground_points = np.linspace(start = int(-world_size/2.0), stop=int(world_size/2.0), num=int(world_size/world_line_spacing), endpoint=True)
figure_space_points = np.linspace(start = int(-figure_render_size/2.0), stop=int(figure_render_size/2.0), num=int(figure_render_size/figure_voxel_size), endpoint=False)

model_voxel_space = VoxelGrid(figure_space_points)


def check_boundaries(cam: Camera):
   
    if cam.speed > 0:
        # X boundaries
        if cam.position.x<-world_size/2:
            cam.position.x = -world_size/2
        if cam.position.x>world_size/2:
            cam.position.x = world_size/2

        # Y boundaries
        if cam.position.y<0.5:
            cam.position.y = 0.5
        if cam.position.y>world_size:
            cam.position.y = world_size

        # Z boundaries
        if cam.position.z<-world_size/2:
            cam.position.z = -world_size/2
        if cam.position.z>world_size/2:
            cam.position.z = world_size/2


        cam.target = cam.position - cam.z_axis
        cam.update_camera_vectors()

def square():
    # glColor3f(1.0, 0.0, 3.0)
    glBegin(GL_QUADS)
    glVertex2f(100, 100)
    glVertex2f(200, 100)
    glVertex2f(200, 200)
    glVertex2f(100, 200)
    glEnd()

def Cube():
    glBegin(GL_LINES)
    glColor3f(1.0,1.0,1.0)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

def ground(ground_points):
    glBegin(GL_LINES)
    glColor3f(1.0,1.0,1.0)
    
    max_pos = max(ground_points)
    min_pos = min(ground_points)
    # #############################################
    # ground lines
    for line in ground_points:
        
        # vertical lines
        glVertex3fv((line,0,max_pos))
        glVertex3fv((line,0,min_pos))
        # horizontal lines
        glVertex3fv((max_pos, 0, line))
        glVertex3fv((min_pos,0,line))

        # #############################################
        # roof lines
        # vertical lines
        glVertex3fv((line,abs(max_pos*2),max_pos))
        glVertex3fv((line,abs(max_pos*2),min_pos))
        # horizontal lines
        glVertex3fv((max_pos,abs(max_pos*2), line))
        glVertex3fv((min_pos,abs(max_pos*2),line))

        # #############################################
        # front wall lines
        # vertical lines
        glVertex3fv((line,abs(max_pos*2),min_pos))
        glVertex3fv((line,0,min_pos))
        # horizontal lines
        glVertex3fv((max_pos,line+max_pos, min_pos))
        glVertex3fv((min_pos,line+max_pos,min_pos))

        # #############################################
        # back wall lines
        # vertical lines
        glVertex3fv((line,abs(max_pos*2),max_pos))
        glVertex3fv((line,0,max_pos))
        # horizontal lines
        glVertex3fv((max_pos,line+max_pos, max_pos))
        glVertex3fv((min_pos,line+max_pos,max_pos))
        
        # #############################################
        # right wall lines
        # vertical lines
        glVertex3fv((min_pos,abs(max_pos*2),line))
        glVertex3fv((min_pos,0,line))
        # horizontal lines
        glVertex3fv((min_pos,line+max_pos, min_pos))
        glVertex3fv((min_pos,line+max_pos, max_pos))
    
        # #############################################
        # left wall lines
        # vertical lines
        glVertex3fv((max_pos,abs(max_pos*2),line))
        glVertex3fv((max_pos, 0, line))
        # horizontal lines
        glVertex3fv((max_pos,line+max_pos, min_pos))
        glVertex3fv((max_pos,line+max_pos, max_pos))

    glEnd()

def plot_axes():
    glBegin(GL_LINES)
    # red X axis
    glColor3f(1.0,0.0,0.0)
    glVertex3fv([0.0,0.0,0.0])
    glVertex3fv([1.0,0.0,0.0])
    # green Y axis
    glColor3f(0.0,1.0,0.0)
    glVertex3fv([0.0,0.0,0.0])
    glVertex3fv([0.0,1.0,0.0])
    # blue Z axis
    glColor3f(0.0,0.0,1.0)
    glVertex3fv([0.0,0.0,0.0])
    glVertex3fv([0.0,0.0,1.0])

    glEnd()

def proces_mouse_movement(prev_pos_x,prev_pos_y,is_mouse_down, cam: Camera):
    #get mouse position and delta (do after pump events)
    (pos_x,pos_y) = pygame.mouse.get_pos()
    #compute delta of previous and actual position
    mouse_delta_x = (prev_pos_x-pos_x)
    mouse_delta_y = (prev_pos_y-pos_y)
    #print(str(mouse_delta_x) + ' ' + str(mouse_delta_y))
    #store new position



    # si el boto esquerra està clicat

    if is_mouse_down:
        # rotate the pertinent amount in each axis
        cam.rotate(-mouse_delta_x*cam.rot_step, Vector3([.0,1.0,.0]))
        cam.rotate(-mouse_delta_y*cam.rot_step, Vector3([1.0,.0,.0]))

    return pos_x, pos_y

def proces_inputs(camera_list: List[Camera]):
    
    global current_camera_index
    global background_show
    global mouse_pos_x
    global mouse_pos_y
    global is_mouse_down


    keys = pygame.key.get_pressed()
    # ARROWS
    if keys[pygame.K_LEFT]:
        direction_vector = Vector3([-1.,.0,.0])
        camera_list[current_camera_index].move(direction_vector*camera_list[current_camera_index].speed)        
    if keys[pygame.K_RIGHT]:
        direction_vector = Vector3([1.,.0,.0])
        camera_list[current_camera_index].move(direction_vector*camera_list[current_camera_index].speed)        
    if keys[pygame.K_UP]:
        direction_vector = Vector3([.0,.0,-1.])
        camera_list[current_camera_index].move(direction_vector*camera_list[current_camera_index].speed)        
    if keys[pygame.K_DOWN]:
        direction_vector = Vector3([.0,.0,1.])
        camera_list[current_camera_index].move(direction_vector*camera_list[current_camera_index].speed)        

    # ASDW
    if keys[pygame.K_a]:
        camera_list[current_camera_index].rotate(-camera_list[current_camera_index].rot_step,Vector3([0.0,1.,.0]))
    if keys[pygame.K_s]:
        camera_list[current_camera_index].rotate(camera_list[current_camera_index].rot_step,Vector3([1.0,0.0,.0]))
    if keys[pygame.K_d]:
        camera_list[current_camera_index].rotate(camera_list[current_camera_index].rot_step,Vector3([0.0,1.0,.0]))
    if keys[pygame.K_w]:
        camera_list[current_camera_index].rotate(-camera_list[current_camera_index].rot_step,Vector3([1.0,0.0,.0]))

    # TOGGLE KEYS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        # KEY DOWN
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                background_show = not  background_show
            
            if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                camera_list[current_camera_index].speed = camera_list[current_camera_index].speed*4

            if event.key == pygame.K_PLUS:
                current_camera_index += 1
                if current_camera_index >= len(camera_list):
                    current_camera_index = 0

            if event.key == pygame.K_MINUS:
                current_camera_index -= 1
                if current_camera_index < 0:
                    current_camera_index = len(camera_list) - 1
            

        # MOUSE DOWN
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_mouse_down = True


        # MOUSE UP
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                is_mouse_down = False
        
    (mouse_pos_x, mouse_pos_y) = proces_mouse_movement(mouse_pos_x, mouse_pos_y, is_mouse_down, camera_list[current_camera_index])
   
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

def create_virtual_cameras_from_real_cameras(cam_intrisic_parameters, extrinsic_parameters_list):
            
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
    camera = Camera(position=[0.,5.,20.],
                        target = [.0,5.,.0],
                        display_height=cam_height, 
                        display_width=cam_width,
                        speed=0.,
                        rot_step = 0.,
                        shilouette=cube_shilouette,
                        )
    camera_list.append(camera)
    camera = Camera(position=[20.,5.,0.],
                        target = [.0,5.,.0],
                        display_height=cam_height, 
                        display_width=cam_width,
                        speed=0.,
                        rot_step = 0.,
                        shilouette=cube_shilouette,
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
        glutInit()
        glutInitDisplayMode(GLUT_RGBA) 
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(100, 100)
        wind = glutCreateWindow('Visual Hull')
        glutDisplayFunc(self.gl_loop_callback)
        glutIdleFunc(self.gl_loop_callback)
        self.vhull_vertex = model_voxel_space.get_voxel_centers_as_np_array()
        self.vhull_vertex = np.array([-0.5,-0.5,0.0,0.5,0.5,-0.5],dtype=np.float32)
        self.vhull_shader = VHullShader(vhull_vertex = self.vhull_vertex)

        self.background_show = True
        '''
        self.mouse_pos_x
        self.mouse_pos_y
        self.is_mouse_down
        '''
        self.ttt = time.time()
        self.show_time = True
        self.is_mouse_down = False

        self.cam_intrinsc_params, self.cam_extrinsic_prmtrs_lst = load_real_cameras(force_recompute = True)
        self.camera_list = create_virtual_cameras_from_real_cameras(self.cam_intrinsc_params, self.cam_extrinsic_prmtrs_lst)
        self.camera_list.append(Camera(position=[0.0,world_size,0.1], target=[0.0,0.0,0.0], has_to_render_image_plane=False))
        
        self.current_camera_index = len(self.camera_list) - 1



        self.print_pos = False
        self.print_modelview = False
        #store initial mouse position
        # (mouse_pos_x,mouse_pos_y) = pygame.mouse.get_pos()

        glutMainLoop()
        glMatrixMode(GL_PROJECTION)
        gluPerspective(self.camera_list[self.current_camera_index].fov[1], (self.display_size[0]/self.display_size[1]), 0.1, 500.0)
    
    def _draw_frame(self):    

        if self.print_modelview:
            a = (GLfloat * 16)()
            mvm = glGetFloatv(GL_MODELVIEW_MATRIX, a)
            print('GL look at matrix: ')
            print(np.asarray(list(a)).reshape((4,4)))
            print('--------------')
            print('Computed look at matrix: ')
            print(self.camera_list[self.current_camera_index].get_view_matrix())
            print('Camera position: ' + str(self.camera_list[self.current_camera_index].position))
            print('----------------------------')
        if self.print_pos:
            print(str(self.camera_list[self.current_camera_index].position) + ' ||||x ' + str(self.camera_list[self.current_camera_index].x_axis) + ' ||||y ' +str(self.camera_list[self.current_camera_index].y_axis) + ' ||||z ' + str(self.camera_list[self.current_camera_index].z_axis))


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
        
        # proces_inputs(self.camera_list)
        # check_boundaries(self.camera_list[self.current_camera_index])

        self.update_gl_matrices()

        if self.print_modelview:
            self.print_gl_and_computed_view_matrices()
        if self.print_pos:
            self.print_camera_position()

        self.render_scene()
        
        if self.show_time:
            self.compute_fps()

    def render_scene(self):
        # clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # render each element
        self.vhull_shader.short_render(viewing_camera = self.camera_list[-1], modeling_cameras = self.camera_list[0:-1])
        render_virtual_cameras(self.camera_list)
        square()
        Cube()
        plot_axes()
        if self.background_show:
            ground(ground_points)
            ground(figure_space_points)
        # swap buffers i dont actualy know what this does
        glutSwapBuffers()

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
        print('GL look at matrix: ')
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
    