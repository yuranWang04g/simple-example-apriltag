import numpy as np
import cv2
import apriltag

def main():
    # 设置 AprilTag 检测器
    options = apriltag.DetectorOptions(families='tag36h11')
    detector = apriltag.Detector(options)

    # 加载相机参数
    camera_params = np.loadtxt('camera_matrix.txt')

    # 加载畸变系数
    distortion_coeffs = np.loadtxt('distortion_coeffs.txt')

    # 读取图像
    image = cv2.imread('standard.jpeg')

    # 转换图像到灰度空间
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 在图像中检测 AprilTag
    tags = detector.detect(gray)

    # 如果未检测到 AprilTag，则退出程序
    if len(tags) == 0:
        print('No tag detected')
        return

    # 获取第一个 AprilTag 的角点
    tag_corners = tags[0].corners

    # 检查角点数量
    if len(tag_corners) < 4:
        print('Not enough tag corners detected')
        return

    # 计算位姿矩阵
    fx = camera_params[0, 0]
    fy = camera_params[1, 1]
    cx = camera_params[0, 2]
    cy = camera_params[1, 2]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion_coefficients = distortion_coeffs.reshape(5, 1)
    tag_size = 0.1 # 以米为单位
    _, rvec, tvec = cv2.solvePnP(np.array([
        [0, 0, 0],
        [tag_size, 0, 0],
        [tag_size, tag_size, 0],
        [0, tag_size, 0]
    ]), tag_corners, camera_matrix, distortion_coefficients)
    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape((3, 1))
    pose = np.concatenate((np.concatenate((R, tvec), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    # 计算相机中心线与 Apriltag 平面的夹角
    camera_z = np.array([0, 0, 1, 1]).reshape((4,1))
    tag_plane = np.array([0, 0, 1, 0]).reshape((4,1))
    camera_z_in_tag_frame = np.linalg.inv(pose) @ camera_z
    angle = np.arccos(np.dot(camera_z_in_tag_frame.T, tag_plane) / (np.linalg.norm(camera_z_in_tag_frame) * np.linalg.norm(tag_plane)))
    angle_degrees = angle * 180 / np.pi
    print('Angle: {:.2f} degrees'.format(angle_degrees.item()))

if __name__ == '__main__':
    main()
