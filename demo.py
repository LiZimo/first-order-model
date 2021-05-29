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
from glob import glob
from PIL import Image
from random import randint

from pynput.keyboard import Key, Listener ## get keyboard input

# import sys
# sys.path.append("./pixel2style2pixel")
# from pSpInfer import pSpInfer



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

def single_pass(source, target_image, generator, kp_detector, kp_source, kp_driving_initial, relative=True, adapt_movement_scale=True, cpu=False):
	with torch.no_grad():
		#predictions = []
		#source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		target_image = torch.tensor(target_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		#initial_image = torch.tensor(initial_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		#if not cpu:
			#source = source.cuda()

		#kp_source = kp_detector(source)
		#kp_driving_initial = kp_detector(initial_image)

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
		#prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

	#return prediction
	return out['prediction'].data
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
		#return frame
		return []
	#return np.array(bboxes)[:, :-1] * scale_factor
	#print(bboxes)
	#input('bboxes')
	left, top, right, bot, scale = bboxes[0]

	### custom bbox expansion code based on size

	length_increase = 1.65
	height_increase = 1.65

	cur_width = right - left
	target_width = length_increase * cur_width
	length_diff = target_width - cur_width
	new_left = int(max(0,round(left - length_diff/2)))
	new_right = int(min(frame.shape[1], round(right + length_diff/2)))


	cur_height = bot - top
	target_height = height_increase * cur_height
	height_diff = target_height - cur_height
	new_top = int(max(0, round(top - height_diff/2)))
	new_bot = int(min(frame.shape[0], round(bot + height_diff/2)))

	#print(frame.shape)
	#print(new_top,new_bot, new_left,new_right)
	#input('new slices')

	#return frame[new_top:new_bot, new_left:new_right, :]
	out_box = [new_left, new_top, new_right, new_bot]
	#print(out_box)
	#input('outbox')

	return out_box

def stream_to_webcam(source_image, reader, generator, kp_detector, face_alignment_instance, relative=True, adapt_movement_scale=True, cpu=False, bbox_avg_frames = 7, check_ref_frame_rate = 3):
	bbox_queue = []
	with torch.no_grad():
		#predictions = []
		source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		if not cpu:
			source = source.cuda()

		norm = float('inf')
		for i, frame in tqdm(enumerate(reader)):
			reader_frame = frame
			

			#### crop and detect relevant part of face
			new_bbox = auto_crop_resize_face(frame, face_alignment_instance)
			if len(new_bbox) != 0:
				bbox_queue.append(new_bbox)
			if len(bbox_queue) > bbox_avg_frames:
				bbox_queue = bbox_queue[-bbox_avg_frames:]

			out_box= np.mean(np.array(bbox_queue), axis = 0)
			[left,top,right,bot] = out_box.astype(np.uint64)

			#print(bbox_queue)
			#input('bbox queue')

			#print(out_box)
			#input('coords')
			frame = frame[top:bot, left:right, :]

			### get first frame as basis for relative motion
			# if i == 0:
			# 	first_frame = frame.copy()
			# 	first_frame = auto_crop_resize_face(first_frame, face_alignment_instance)
			# 	first_frame = resize(first_frame, (256, 256))[..., :3]

			frame = resize(frame, (256, 256))[..., :3]
			if i == 0:
				reference_frame = frame

			### check once per while to see if we can get a better reference_frame
			### only using mouth keypoints for reference frame
			if i % check_ref_frame_rate == 0:

				kp_source_norm = area_normalize_kp ( face_alignment_instance.get_landmarks(255 * source_image)[0] )[48:60,:]
				kp_driving_norm = area_normalize_kp ( face_alignment_instance.get_landmarks(255 * frame)[0] )[48:60,:]
	
				new_norm = (np.abs(kp_source_norm - kp_driving_norm) ** 2).sum()
				if new_norm < norm:
					norm = new_norm
					reference_frame = frame
					print('using new reference_frame')

			output = single_pass(source, reference_frame, frame, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_movement_scale, cpu=cpu)
			#output_bgr = output[..., ::-1]

			#print(frame.shape)
			#print(output.shape)
			#input('shapes')

			#concatenated = np.concatenate((frame,output), axis = 1)

			#cv2.imshow('image', concatenated[..., ::-1])
			#cv2.imshow('image', frame[..., ::-1])
			cv2.imshow('image', output[..., ::-1])
			k = cv2.waitKey(20)
			if (k & 0xff == ord('q')):
				break

			# cv2.imshow('src', reader_frame[..., ::-1])
			# m = cv2.waitKey(20)
			# if (m & 0xff == ord('z')):
			# 	break

			# cv2.imshow('cropped', frame[..., ::-1])
			# m = cv2.waitKey(20)
			# if (m & 0xff == ord('a')):
			# 	break

def stream_to_webcam_multisource(source_dir, reader, generator, kp_detector, face_alignment_instance, relative=True, adapt_movement_scale=True, cpu=False, bbox_avg_frames = 9, check_ref_frame_rate = 5, cycle_time = 30, pspInfer_net = None):
	bbox_queue = []
	source_dir_basic = source_dir + '/basic'
	source_dir_priority = source_dir + '/priority'
	source_files_ind = 0
	with torch.no_grad():

		norm = float('inf')
		for i, frame in tqdm(enumerate(reader)):

			if i % cycle_time == 0:

				if randint(0,2) == 2:
					source_dir = source_dir_basic
				else:
					source_dir = source_dir_priority

				source_files = glob(source_dir + '/**/*.*', recursive=True)
				source_image = imageio.imread(source_files[source_files_ind % len(source_files)])
				source_image = resize(source_image, (256, 256))[..., :3]
				source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
				source_files_ind += 1
				#source_files_ind = source_files_ind % len(source_files)
				if not cpu:
					source = source.cuda()

				kp_source = kp_detector(source)

			reader_frame = frame
			

			#### crop and detect relevant part of face
			new_bbox = auto_crop_resize_face(frame, face_alignment_instance)
			#bbox_queue.append(new_bbox)
			if len(new_bbox) != 0:
				bbox_queue.append(new_bbox)
			if len(bbox_queue) > bbox_avg_frames:
				bbox_queue = bbox_queue[-bbox_avg_frames:]

			out_box= np.mean(np.array(bbox_queue), axis = 0)
			[left,top,right,bot] = out_box.astype(np.uint64)

			#print(bbox_queue)
			#input('bbox queue')

			#print(out_box)
			#input('coords')
			frame = frame[top:bot, left:right, :]

			### get first frame as basis for relative motion
			# if i == 0:
			# 	first_frame = frame.copy()
			# 	first_frame = auto_crop_resize_face(first_frame, face_alignment_instance)
			# 	first_frame = resize(first_frame, (256, 256))[..., :3]

			frame = resize(frame, (256, 256))[..., :3]
			if i == 0:
				reference_frame = frame
				reference_frame = torch.tensor(reference_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
				kp_driving_initial = kp_detector(reference_frame)

			### check once per while to see if we can get a better reference_frame
			if i % check_ref_frame_rate == 0:
				#print('checking reference_frame')

				indices = list(range(48,60)) #+ [3,4,12,13]

				#try:
				#	kp_source_norm = area_normalize_kp ( face_alignment_instance.get_landmarks(255 * source_image)[0] )[indices,:]
				#	kp_driving_norm = area_normalize_kp ( face_alignment_instance.get_landmarks(255 * frame)[0] )[indices,:]
#
				#	#kp_source_norm = area_normalize_kp ( face_alignment_instance.get_landmarks(255 * source_image)[0] )#[indices,:]
				#	#kp_driving_norm = area_normalize_kp ( face_alignment_instance.get_landmarks(255 * frame)[0] )#[indices,:]
#
				#	#source_ratio = np.abs(kp_source_norm[48,:] - kp_source_norm[54,:])/np.abs(kp_source_norm[51,:] - kp_source_norm[57,:])
				#	#driving_ratio = np.abs(kp_driving_norm[48,:] - kp_driving_norm[54,:])/np.abs(kp_driving_norm[51,:] - kp_driving_norm[57,:])
				#	
				#	#print(kp_source_norm.shape)
				#	#print(kp_driving_norm.shape)
				#	#input('keypoint shape')
		#
				#	new_norm = (np.abs(kp_source_norm - kp_driving_norm) ** 2).sum()
				#	#ew_norm = (np.abs(source_ratio - driving_ratio) ** 2).sum()
				#	if new_norm < norm:
				#		norm = new_norm
				#		reference_frame = frame
				#		print("current norm is:", norm)
				#		print('using new reference_frame')
				#except:
				#	print("detected no new landmarks for this frame")

			output = single_pass(source, frame, generator, kp_detector, kp_source, kp_driving_initial, relative=relative, adapt_movement_scale=adapt_movement_scale, cpu=cpu)
			#print(output.shape)
			#input('output_shape')
			#np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
			output = np.transpose(output.cpu().numpy(), [0, 2, 3, 1])[0]
			#print(output.max())
			#print(output.min())
			#input('output shape')
			if pspInfer_net is not None:
				output = Image.fromarray( (output * 255).astype(np.uint8))
				#output = (output - 0.5)/0.5
				output = pspInfer_net.process(output)
				output = np.array(output)
			#else:
				
			
			
			#print(output.shape)
			#input('output shape after psp net')

			#concatenated = np.concatenate((frame,output), axis = 1)

			#cv2.imshow('image', concatenated[..., ::-1])
			#cv2.imshow('image', frame[..., ::-1])
			cv2.imshow('image', output[..., ::-1])
			k = cv2.waitKey(20)
			if (k & 0xff == ord('q')):
				break

			# cv2.imshow('src', reader_frame[..., ::-1])
			# m = cv2.waitKey(20)
			# if (m & 0xff == ord('z')):
			# 	break

			# cv2.imshow('cropped', frame[..., ::-1])
			# m = cv2.waitKey(20)
			# if (m & 0xff == ord('a')):
			# 	break


def area_normalize_kp(kp):
	kp = kp - kp.mean(axis=0, keepdims=True)
	area = ConvexHull(kp[:, :2]).volume
	area = np.sqrt(area)
	kp[:, :2] = kp[:, :2] / area
	return kp

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
	parser.add_argument("--source_dir", default=None, help="path to directory of source images")
	parser.add_argument("--cycle_time", default=25, help="time in seconds of changing source image, if source_dir is given")
	parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
	parser.add_argument("--result_video", default='result.mp4', help="path to output")
 
	parser.add_argument("--relative", dest="relative", default = True, action="store_true", help="use relative or absolute keypoint coordinates")
	parser.add_argument("--adapt_scale", dest="adapt_scale", default = True, action="store_true", help="adapt movement scale based on convex hull of keypoints")

	parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
						help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

	parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
						help="Set frame to start from.")
 
	parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
	parser.add_argument("--webcam", action="store_true", default = False, help = "stream result to webcam")
	parser.add_argument("--psp", action="store_true", default = False, help = "use pixel2style2pixel conversion in output")
	parser.add_argument("--psp_size", type=int, default=1024, help = "size of psp output")

	parser.set_defaults(relative=False)
	parser.set_defaults(adapt_scale=False)

	opt = parser.parse_args()



	generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)


	if not opt.webcam:

		source_image = imageio.imread(opt.source_image)
		source_image = resize(source_image, (256, 256))[..., :3]

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
		pspInfer_net = None
		if opt.psp:
			ckpt_path = "./pixel2style2pixel/checkpoints/psp_ffhq_encode.pt"
			print('loading pspInfer_net...')
			pspInfer_net = pSpInfer(ckpt_path, opt.psp_size)


		device = 'cpu' if opt.cpu else 'cuda'
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
		reader = imageio.get_reader("<video0>")

		if opt.source_dir is None:
			source_image = imageio.imread(opt.source_image)
			source_image = resize(source_image, (256, 256))[..., :3]
			stream_to_webcam(source_image, reader, generator, kp_detector, fa, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
		else: 
			stream_to_webcam_multisource(opt.source_dir, reader, generator, kp_detector, fa, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu, cycle_time= int (opt.cycle_time * 10), pspInfer_net = pspInfer_net)