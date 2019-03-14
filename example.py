import deepdrive as dd

import cv2
import time
import camera_config


def main():
    print(camera_config.rigs['baseline_rigs'])
    env = dd.start(cameras=camera_config.rigs['three_cam_rig'][0], use_sim_start_command=False, render=False, fps=8)
    forward = dd.action(throttle=1, steering=0, brake=0)
    done = False
    while not done:
        observation, reward, done, info = env.step(forward)
        if observation is not None:
            image = observation['cameras'][0]['image']
            cv2.imshow('image', image)
            image1 = observation['cameras'][1]['image']
            cv2.imshow('image1', image1)
            image2 = observation['cameras'][2]['image']
            cv2.imshow('image2', image2)
            print('show')
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
