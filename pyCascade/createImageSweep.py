# Function that generates CharLES input file code to create a sweep of images that can be 
# used to create an animated movie with moving camera

'''
Sample static usage:
python3 createImageSweep.py --static --N 100 --camera0 -47 19 0 --camera1 -47 19 0 --width0 0.6 --width1 0.6 --target 0 1.02 0 --up 0.354 0.935 0 --sweeptype 'polar' --imgpath '/scratch/users/jhochsch/Charles/SN_RL4_5_x4stresses/images_sweep/Psurf_sweep_360deg' --cmd 'SIZE 1376 782 VAR_ON_SURFACE p RANGE_ON_SURFACE -300 100 HIDE_ZONES 0,1,2,3,4,5,6,7,10' --header 'SURF /home/groups/gorle/jack/Charles/SN_30mdomain_RL4_5.mles \nRESTART data/SN_RL4_5_x4stresses.00500000.sles\n'

Sample dynamic usage
python3 createImageSweep.py --N 1000 --step0 300000 --step1 301000 --camera0 -47 19 0 --camera1 -47 19 0 --width0 0.6 --width1 0.6 --target 0 1.02 0 --up 0.354 0.935 0 --sweeptype 'polar' --imgpath '/scratch/users/jhochsch/Charles/SN_RL4_5_x4stresses/images_sweep/Q_20m_sweep_360deg' --cmd 'SIZE 1376 782 GEOM ISO q_criterion() 20000000 VAR_ON_ISO mag(u) RANGE_ON_ISO 0 20 COLORMAP_ON_ISO HOT_METAL HIDE_ZONES 0,1,2,3,4,5,6,7,10'
'''

import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='True to only save images from a single timestep, False for a range of timesteps')
    parser.add_argument('--step0', type=int, help='First step. Required if static == False')
    parser.add_argument('--step1', type=int, help='Last step. Required if static == False')
    parser.add_argument('--N', type=int, help='How many images to write')
    parser.add_argument('--camera0', type=float, nargs=3, help='Starting camera point')
    parser.add_argument('--camera1', type=float, nargs=3, help='End camera point')
    parser.add_argument('--width0', type=float, help='Starting width')
    parser.add_argument('--width1', type=float, help='End width')
    parser.add_argument('--target', type=float, nargs=3, help='Image target (focal point)')
    parser.add_argument('--up', type=float, nargs=3, help='Image up vector')
    parser.add_argument('--sweeptype', type=str, help='linear | polar')
    parser.add_argument('--imgpath', type=str, help='Path to save the image')
    parser.add_argument('--cmd', type=str, help='Rest of the image command, e.g. VAR, HIDE_ZONES, etc.')
    parser.add_argument('--header', type=str, help='Header for the .in file')
    args = parser.parse_args()

    if args.step0 is None or args.step1 is None:
        if not args.static:
            raise Exception('If not a static animation, must specify start and end timestep')
        
        args.step0 = 0
        args.step1 = args.N-1
    
    if args.sweeptype == 'polar':
        # Only rotation around the y axis is supported here
        theta0 = np.arctan2(args.camera0[2], args.camera0[0])
        theta1 = np.arctan2(args.camera1[2], args.camera1[0])
        if theta0==theta1:
            # Consider this to be a 360 degree sweep
            theta1 = theta0 + 2*np.pi

        r0 = np.sqrt(args.camera0[0] **2 + args.camera0[2] **2)
        r1 = np.sqrt(args.camera1[0] **2 + args.camera1[2] **2)

    steps = np.linspace(args.step0, args.step1, args.N).astype(int)
    with open('image_sweep_code.txt', 'w+') as f:
        if args.header is not None:
            f.write(args.header)
            f.write('\n')

        # Loop through images:
        for i in range(args.N):
            cur_width = args.width0 + i/(args.N-1) * (args.width1 - args.width0)
            
            # Calculate coordinates:
            if args.sweeptype == 'linear':
                cur_up = args.up # doesn't change
                cur_camera = []
                for j in [0, 1, 2]:
                    cur_camera.append(args.camera0[j] + i/(args.N-1) * (args.camera1[j] - args.camera0[j]))
            elif args.sweeptype == 'polar':
                cur_r = r0 + i/(args.N-1) * (r1 - r0)
                cur_theta = theta0 + i/(args.N-1) * (theta1 - theta0)
                cur_y = args.camera0[1] + i/(args.N-1) * (args.camera1[1] - args.camera0[1])
                cur_camera = [cur_r * np.cos(cur_theta), cur_y, cur_r * np.sin(cur_theta)]
                
                rot_mat = np.matrix([[np.cos(cur_theta-np.pi), 0, np.sin(cur_theta-np.pi)], [0, 1, 0], [-np.sin(cur_theta-np.pi), 0, np.cos(cur_theta-np.pi)]])
                
                cur_up =  np.matmul(args.up, rot_mat).flat
                # print('CAMERA %.2f, %.2f, %.2f UP %.2f, %.2f, %.2f' %(cur_camera[0], cur_camera[1], cur_camera[2], cur_up[0], cur_up[1], cur_up[2]))
            else:
                raise Exception('Unrecognized sweep_type; should be "linear" or "polar"')
            
            # Write image commands:
            if not args.static:
                f.write('WHEN (step==%d)\n' %steps[i])
            
            if args.static:
                f.write('WRITE_IMAGE NAME=%s.%05d \\\n' %(args.imgpath, i))
            else:
                f.write('WRITE_IMAGE NAME=%s \\\n' %args.imgpath)
            f.write('  INTERVAL 1 \\\n')
            f.write('  TARGET %.6f %.6f %.6f \\\n' %(args.target[0], args.target[1], args.target[2]))
            f.write('  CAMERA %.6f %.6f %.6f \\\n' %(cur_camera[0], cur_camera[1], cur_camera[2]))
            f.write('  UP %.6f %.6f %.6f WIDTH %.6f \\\n' %(cur_up[0], cur_up[1], cur_up[2], cur_width))
            f.write('  %s \n' %args.cmd)

            if not args.static:
                f.write('ENDWHEN\n\n')
            else:
                f.write('\n')

    f.close()

if __name__ == "__main__":
    main()

