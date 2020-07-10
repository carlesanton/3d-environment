import pygame
import os
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


#3d enviroment where visual hull wll be represented


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
line_spacing = 4.0

current_camera_index = -1


ground_points = np.linspace(start = int(-world_size/2.0), stop=int(world_size/2.0), num=int(world_size/line_spacing), endpoint=True)



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

def Cube():
    glBegin(GL_LINES)
    glColor3f(1.0,1.0,1.0)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

def ground():
    glBegin(GL_LINES)
    glColor3f(1.0,1.0,1.0)
    
    # #############################################
    # ground lines
    for line in ground_points:
        
        # vertical lines
        glVertex3fv((line,0,world_size/2))
        glVertex3fv((line,0,-world_size/2))
        # horizontal lines
        glVertex3fv((world_size/2,0, line))
        glVertex3fv((-world_size/2,0,line))

        # #############################################
        # roof lines
        # vertical lines
        glVertex3fv((line,world_size,world_size/2))
        glVertex3fv((line,world_size,-world_size/2))
        # horizontal lines
        glVertex3fv((world_size/2,world_size, line))
        glVertex3fv((-world_size/2,world_size,line))

        # #############################################
        # front wall lines
        # vertical lines
        glVertex3fv((line,world_size,-world_size/2))
        glVertex3fv((line,0,-world_size/2))
        # horizontal lines
        glVertex3fv((world_size/2,line+world_size/2, -world_size/2))
        glVertex3fv((-world_size/2,line+world_size/2,-world_size/2))

        # #############################################
        # back wall lines
        # vertical lines
        glVertex3fv((line,world_size,world_size/2))
        glVertex3fv((line,0,world_size/2))
        # horizontal lines
        glVertex3fv((world_size/2,line+world_size/2, world_size/2))
        glVertex3fv((-world_size/2,line+world_size/2,world_size/2))
        
        # #############################################
        # right wall lines
        # vertical lines
        glVertex3fv((-world_size/2,world_size,line))
        glVertex3fv((-world_size/2,0,line))
        # horizontal lines
        glVertex3fv((-world_size/2,line+world_size/2, -world_size/2))
        glVertex3fv((-world_size/2,line+world_size/2, world_size/2))
    
        # #############################################
        # left wall lines
        # vertical lines
        glVertex3fv((world_size/2,world_size,line))
        glVertex3fv((world_size/2,0,line))
        # horizontal lines
        glVertex3fv((world_size/2,line+world_size/2, -world_size/2))
        glVertex3fv((world_size/2,line+world_size/2, world_size/2))

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

def create_virtual_cameras_from_real_cameras(calibration_image_folder: str = default_calibartion_images_folder,
                                             force_recompute: bool = False):
    
    if force_recompute:
        cam_intrisic_parameters, extrinsic_parameters_list = get_calibration_matrix_and_external_params(calibration_image_folder)
    else:
        try:
            with open('camera_data/camera_parameters.pkl', 'rb') as handle:
                cam_intrisic_parameters = pickle.load(handle)
            with open('camera_data/external_parameters_list.pkl', 'rb') as handle:
                extrinsic_parameters_list = pickle.load(handle)

        except:
            cam_intrisic_parameters, extrinsic_parameters_list = get_calibration_matrix_and_external_params(calibration_image_folder)
            
    camera_list: List[Camera] = []

    cam_width = 640
    cam_height = 480
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

    return camera_list

def render_virtual_cameras(camera_list: List[Camera]):
    for camera in camera_list:
        if camera.has_to_render:
            camera.render_camera()


def main():
    pygame.init()
    display = (1200,600)
    
    global background_show
    global mouse_pos_x
    global mouse_pos_y
    global is_mouse_down
    background_show = True
    ttt = time.time()
    show_time = True
    is_mouse_down = False

    global current_camera_index
    camera_list = create_virtual_cameras_from_real_cameras(force_recompute=True)
    camera_list.append(Camera(position=[0.0,0.5,10.0], target=[0.0,0.0,0.0], has_to_render_image_plane=False))
    
    current_camera_index = len(camera_list) - 1



    print_pos = False
    print_modelview = False
    #store initial mouse position
    (mouse_pos_x,mouse_pos_y) = pygame.mouse.get_pos()

    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)


    glMatrixMode(GL_PROJECTION)
    gluPerspective(camera_list[current_camera_index].fov[1], (display[0]/display[1]), 0.1, 500.0)
    while True:
        pygame.time.delay(10)

        proces_inputs(camera_list)
        check_boundaries(camera_list[current_camera_index])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera_list[current_camera_index].cam_lookat()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50, (display[0]/display[1]), 0.1, 100.0)
        if print_modelview:
            a = (GLfloat * 16)()
            mvm = glGetFloatv(GL_MODELVIEW_MATRIX, a)
            print('GL look at matrix: ')
            print(np.asarray(list(a)).reshape((4,4)))
            print('--------------')
            print('Computed look at matrix: ')
            print(camera_list[current_camera_index].get_view_matrix())
            print('Camera position: ' + str(camera_list[current_camera_index].position))
            print('----------------------------')
        if print_pos:
            print(str(camera_list[current_camera_index].position) + ' ||||x ' + str(camera_list[current_camera_index].x_axis) + ' ||||y ' +str(camera_list[current_camera_index].y_axis) + ' ||||z ' + str(camera_list[current_camera_index].z_axis))
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        camera_list[0].project_3d_point_into_image_plane_opencv(Vector3([0.,0.,0.]))

        render_virtual_cameras(camera_list)
        plot_axes()
        Cube()
        if background_show:
            ground()
        pygame.display.flip()

        if show_time:
            d_t = time.time() - ttt
            frame_rate = 1/(d_t)
            # print(f'Frame rate: {frame_rate} fps')
            ttt = time.time()

        ##pygame.time.wait(10)

if __name__ == "__main__":
    main()