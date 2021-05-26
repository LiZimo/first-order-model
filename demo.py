import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import cv2

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import face_alignment


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def single_pass(source_image, initial_image, target_image, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        #predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        target_image = torch.tensor(target_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        initial_image = torch.tensor(initial_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(initial_image)

        #print(source.shape)
        #input('source shape')

        #print(target_image.shape)
        #input('target_image_shape')


        driving_frame = target_image
        if not cpu:
            driving_frame = driving_frame.cuda()
        kp_driving = kp_detector(driving_frame)
        kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                               kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
        prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

    return prediction
        #predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])


def auto_crop_resize_face(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        #return []
        return frame
    #return np.array(bboxes)[:, :-1] * scale_factor
    #print(bboxes)
    #input('bboxes')
    left, top, right, bot, scale = bboxes[0]

    ### custom bbox expansion code based on size

    length_increase = 1.45
    height_increase = 1.45

    cur_width = right - left
    target_width = length_increase * cur_width
    length_diff = target_width - cur_width
    new_left = max(0,round(left - length_diff/2))
    new_right = min(frame.shape[1], round(right + length_diff/2))


    cur_height = bot - top
    target_height = height_increase * cur_height
    height_diff = target_height - cur_height
    new_top = max(0, round(top - height_diff/2))
    new_bot = min(frame.shape[0], round(bot + height_diff/2))

    return frame[new_top:new_bot, new_left:new_right, :]



def stream_to_webcam(source_image, reader, generator, kp_detector, face_alignment_instance, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        #predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        for i, frame in tqdm(enumerate(reader)):
            ### get first frame as basis for relative motion
            if i == 0:
                first_frame = frame.copy()
                first_frame = auto_crop_resize_face(first_frame, face_alignment_instance)
                first_frame = resize(first_frame, (256, 256))[..., :3]

            #### crop and detect relevant part of face
            frame = auto_crop_resize_face(frame, face_alignment_instance)


            frame = resize(frame, (256, 256))[..., :3]
            output = single_pass(source_image, first_frame, frame, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_movement_scale, cpu=cpu)
            #output_bgr = output[..., ::-1]

            #print(frame.shape)
            #print(output.shape)
            #input('shapes')

            concatenated = np.concatenate((frame,output), axis = 1)

            cv2.imshow('image', concatenated[..., ::-1])
            #cv2.imshow('image', frame[..., ::-1])
            #cv2.imshow('image', output[..., ::-1])
            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--webcam", action="store_true", default = False, help = "stream result to webcam")


    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    # reader = imageio.get_reader(opt.driving_video)
    # fps = reader.get_meta_data()['fps']
    # driving_video = []
    # try:
    #     for im in reader:
    #         driving_video.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    # driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)


    if not opt.webcam:

        reader = imageio.get_reader(opt.driving_video)
            fps = reader.get_meta_data()['fps']
            driving_video = []
            try:
                for im in reader:
                    driving_video.append(im)
            except RuntimeError:
                pass
            reader.close()
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        if opt.find_best_frame or opt.best_frame is not None:
            i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


    else:
        device = 'cpu' if opt.cpu else 'cuda'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
        reader = imageio.get_reader("<video0>")
        stream_to_webcam(source_image, reader, generator, kp_detector, fa, relative=True, adapt_movement_scale=True, cpu=False)