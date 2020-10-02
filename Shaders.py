from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from Camera import Camera
import numpy as np
import sys
from typing import List
from pyrr import Vector3, Vector4
from PIL import Image

class Shader:
    
    # initialise opengl
    # from https://rdmilligan.wordpress.com/2016/08/27/opengl-shaders-using-python/
    def __init__(self, vertex_shader_source, fragment_shader_source):
        self.program = self.compile_program(vertex_shader_source, fragment_shader_source)

    # from PySpace
    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        status = c_int()
        glGetShaderiv(shader, GL_COMPILE_STATUS, byref(status))
        if not status.value:
            self.print_log(shader)
            glDeleteShader(shader)
            raise ValueError('Shader compilation failed')
        return shader

    # from PySpace
    def compile_program(self, vertex_source, fragment_source):
        vertex_shader = None
        fragment_shader = None
        program = glCreateProgram()

        if vertex_source:
            print("Compiling Vertex Shader...")
            vertex_shader = self.compile_shader(vertex_source, GL_VERTEX_SHADER)
            glAttachShader(program, vertex_shader)
        if fragment_source:
            print("Compiling Fragment Shader...")
            fragment_shader = self.compile_shader(fragment_source, GL_FRAGMENT_SHADER)
            glAttachShader(program, fragment_shader)

        glLinkProgram(program)
        result = glGetProgramiv(program, GL_LINK_STATUS)
        info_log_len = glGetProgramiv(program, GL_INFO_LOG_LENGTH)
        if info_log_len:
            print("Error linking program...")
            print(glGetProgramInfoLog(program))
            sys.exit(11)


        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return program

    # from PySpace
    def print_log(self, shader):
        length = c_int()
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, byref(length))

        if length.value > 0:
            log = create_string_buffer(length.value)
            print(glGetShaderInfoLog(shader))

    def enable(self):
        glUseProgram(self.program)
        assert(glGetError() == GL_NO_ERROR)

        self.last_slot = 0
    
    def disable(self):
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)
        assert(glGetError() == GL_NO_ERROR)

    def init_gl_vertex_and_color_buffers_into_vao(self, vertices_array, colors_array, vertex_size):
        # store attribute locations
        if 'a_vert_location' not in locals():
            self.a_vert_location = glGetAttribLocation(self.program, "vert")
            self.a_color_location = glGetAttribLocation(self.program, "input_color")

        array_type = (GLfloat * len(vertices_array))

        # Create VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
            
        # Create VBO for vertices and color
        vertices_array_buffer = glGenBuffers(1)
        colors_array_buffer = glGenBuffers(1)
        
        # Fill vertices data
        glBindBuffer(GL_ARRAY_BUFFER, vertices_array_buffer)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(ctypes.c_float) * len(vertices_array), array_type(*vertices_array), GL_STATIC_DRAW)
        glEnableVertexAttribArray(self.a_vert_location)
        glVertexAttribPointer(self.a_vert_location, vertex_size, GL_FLOAT, GL_FALSE, ctypes.sizeof(ctypes.c_float) * vertex_size, None)
            
        # Fill color data
        glBindBuffer(GL_ARRAY_BUFFER, colors_array_buffer)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(ctypes.c_float) * len(colors_array), array_type(*colors_array), GL_STATIC_DRAW)
        glEnableVertexAttribArray(self.a_color_location)
        glVertexAttribPointer(self.a_color_location, vertex_size, GL_FLOAT, GL_FALSE, ctypes.sizeof(ctypes.c_float) * vertex_size, None)

        # unbind buffers
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return vao


class TextureShader(Shader):
    def __init__(self, vertex_shader_source, fragment_shader_source, texture_unit = 0):
        super().__init__(vertex_shader_source, fragment_shader_source)
        self.texture_unit = texture_unit

    def init_gl_buffers(self, vertices_array, uv_array, texture_image_data, vertex_size, texture_image_shape):
        self.enable()
        # store attribute locations
        if 'a_vert_location' not in locals():
            self.a_vert_location = glGetAttribLocation(self.program, "vert")
            self.a_uv_location = glGetAttribLocation(self.program, "vertexUV")
            self.u_texture_location = glGetUniformLocation(self.program, "texture_image")


        array_type = (GLfloat * len(vertices_array))

        # Create VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
            
        # Create VBO for vertices and color
        vertices_array_buffer = glGenBuffers(1)
        uv_array_buffer = glGenBuffers(1)
        texture_image_buffer = glGenBuffers(1)
        
        # Fill vertices data
        glBindBuffer(GL_ARRAY_BUFFER, vertices_array_buffer)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(ctypes.c_float) * len(vertices_array), array_type(*vertices_array), GL_STATIC_DRAW)
        glEnableVertexAttribArray(self.a_vert_location)
        glVertexAttribPointer(self.a_vert_location, vertex_size, GL_FLOAT, GL_FALSE, ctypes.sizeof(ctypes.c_float) * vertex_size, None)
            
        # Fill UV data
        glBindBuffer(GL_ARRAY_BUFFER, uv_array_buffer)
        glBufferData(GL_ARRAY_BUFFER, uv_array.itemsize * len(uv_array), uv_array, GL_STATIC_DRAW)
        glEnableVertexAttribArray(self.a_uv_location)
        glVertexAttribPointer(self.a_uv_location, 2, GL_FLOAT, GL_FALSE, uv_array.itemsize * 2, None)

        # Fill texture buffer
        self.texture_image_buffer = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0 + self.texture_unit)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_image_buffer)
        glTexStorage3D( 
            GL_TEXTURE_2D_ARRAY,
            1,
            GL_RGB8,
            texture_image_shape[0], 
            texture_image_shape[1], 
            2
        )
        glTexSubImage3D( 
            GL_TEXTURE_2D_ARRAY,
            0,
            0,0,0,                 # xoffset, yoffset, zoffset
            texture_image_shape[0],# width
            texture_image_shape[1],# height
            1,                     # depth
            GL_RGB,                # format
            GL_UNSIGNED_BYTE,      # type
            texture_image_data     # data
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # glTexImage3D(GL_TEXTURE_2D_ARRAY, 2, GL_RGB, texture_image_shape[0], texture_image_shape[1], 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_image_data)
        glUniform1i(self.u_texture_location, self.texture_unit)

        # unbind buffers
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        self.disable()
        return vao 

    def update_texture_image(self, new_texture_image):
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_image_buffer)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, new_texture_image.shape[0], new_texture_image.shape[1], 3, 0, GL_RGBA, GL_UNSIGNED_BYTE, new_texture_image.tobytes())
