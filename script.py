from pyrr import Vector3, Vector4, Matrix44
import numpy as np

def project_point(world_point):
    camera_model_matrix = Matrix44([[  1.,   0.,   0.,   0.],
                                    [  0.,   1.,   0.,   -5.],
                                    [  0.,   0.,   1.,   -20],
                                    [  0.,  0., 0.,   1.]])
    projection_matrix = Matrix44([[ 0.675     ,  0.        ,  0.        ,  0.        ],
                                  [ 0.        ,  0.9       ,  0.        ,  0.        ],
                                  [ 0.        ,  0.        , -1.00040008, -1.        ],
                                  [ 0.        ,  0.        , -0.20004001,  0.        ]])
    camera_center = np.array([320.0, 240.0])
    camera_focal_lenght = np.array([0.6, 0.6])
    image_size = camera_center*2

    cam_point = camera_model_matrix.dot(Vector4.from_vector3(world_point, w= 1.))
    point_in_screen = projection_matrix.dot(cam_point)
    normalized_coordinate = np.array([
        cam_point[0] / cam_point[2], 
        cam_point[1] / cam_point[2],
    ])

    normalized_coordinate_times_focal = np.array([
        camera_focal_lenght[0] * normalized_coordinate[0],
        camera_focal_lenght[1] * normalized_coordinate[1],
    ])
    pixel_pos_in_cam = np.array([
            camera_center[0] - normalized_coordinate_times_focal[0],
            camera_center[1] - normalized_coordinate_times_focal[1],
    ])
    pixel_in_uv_coord_cam_1 = np.array([
        pixel_pos_in_cam[0]/image_size[0],
        pixel_pos_in_cam[1]/image_size[1],
    ])
    return point_in_screen

if __name__ == "__main__":
    print(project_point(np.array([0.,5.0,1.5])))
    print(project_point(np.array([1.,5.0,1.5])))
    print(project_point(np.array([10.,5.0,1.5])))
    print(project_point(np.array([100.,5.0,1.5])))
