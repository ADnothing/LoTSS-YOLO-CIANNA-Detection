import numpy as np
from astropy.io import fits
from astropy import wcs as WCS
from astropy import units as u
from astropy.wcs import utils
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs.utils import pixel_to_skycoord

from tqdm import tqdm
import glob, os, re ,sys, time
from numba import jit

from multiprocessing import Pool

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")


fits_path = "/minerva/Anthore/LoTSS/"
circular_field = True #Set "True" if the mosaic is a disk strictly registered in a 2D array (i.e. like LoTSS mosaics)

######	  GLOBAL VARIABLES AND DATA	  #####
data_path = "./SDC1_data/"

#####    NETWORK RELATED GLOBAL VARIABLES     #####
image_size 	= 256
nb_param  	= 5
nb_box 		= 9
nb_class	= 0
max_nb_obj_per_image = int(340*((image_size*image_size)/(256*256)))

#Size priors for all possible boxes per grid. element
prior_w = f_ar([6.0,6.0,6.0,6.0,6.0,6.0,12.0,12.0,24.0])
prior_h = f_ar([6.0,6.0,6.0,6.0,6.0,6.0,12.0,12.0,24.0])
prior_size = np.vstack((prior_w, prior_h))

#No obj probability prior to rebalance the size distribution
prior_noobj_prob = f_ar([0.2,0.2,0.2,0.2,0.2,0.2,0.02,0.02,0.02])

#Relative scaling of each extra paramater
param_ind_scales = f_ar([2.0,2.0,1.0,0.5,0.5])

box_prior_class = np.array([0,0,0,0,0,0,1,1,4], dtype="int")

#####   INFERENCE RELATED GLOBAL VARIABLES    #####
fwd_image_size = 512
c_size = 16 #Grid element size / reduction factor
yolo_nb_reg = int(fwd_image_size/c_size) #Number of grid element per dimension

overlap 	= c_size*2
patch_shift = fwd_image_size - overlap #240

val_med_lims = np.array([0.6,0.3,0.1])
val_med_obj  = np.array([0.9,0.7,0.5])

first_nms_thresholds 	 = np.array([0.05,-0.1,-0.3,-0.5]) #lower is stricter
first_nms_obj_thresholds = np.array([1.0,0.70,0.50,0.30])
second_nms_threshold 	 = -0.15

prob_obj_cases = np.array([0.2314, 0.1449, 0.2602, 0.1289, 0.2454, 0.2183, 0.0602, 0.0677, 0.0536])
prob_obj_edges = prob_obj_cases + 0.0


@jit(nopython=True, cache=True, fastmath=False)
def fct_DIoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0])*abs(box1[3] - box1[1]) + \
			 abs(box2[2]-box2[0])*abs(box2[3] - box2[1]) - inter_2d
	enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))
	enclose_2d = enclose_w*enclose_h

	cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
	cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
	dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
	diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h)

	return float(inter_2d)/float(uni_2d) - float(dist_cent)/float(diag_enclose)

@jit(nopython=True, cache=True, fastmath=False)
def tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, patch, val_med_lim, val_med_obj, hist_count):
	c_nb_box = 0
	for i in range(0,yolo_nb_reg):
		for j in range(0,yolo_nb_reg):
			kept_count = 0
			for k in range(0,nb_box):
				offset = int(k*(8+nb_param))
				c_box[4] = c_pred[offset+6,i,j] #probability
				c_box[5] = c_pred[offset+7,i,j] #objectness
				#Manual objectness penality on the edges of the images (help for both obj selection and NMS)
				if((j == 0 or j == yolo_nb_reg-1 or i == 0 or i == yolo_nb_reg-1)):
					c_box[4] = max(0.03,c_box[4]-0.05)
					c_box[5] = max(0.03,c_box[5]-0.05)
				
				if(c_box[5] >= prob_obj_cases[k]):
					bx = (c_pred[offset+0,i,j] + c_pred[offset+3,i,j])*0.5
					by = (c_pred[offset+1,i,j] + c_pred[offset+4,i,j])*0.5
					bw = max(5.0, c_pred[offset+3,i,j] - c_pred[offset+0,i,j])
					bh = max(5.0, c_pred[offset+4,i,j] - c_pred[offset+1,i,j])
					
					c_box[0] = bx - bw*0.5; c_box[1] = by - bh*0.5
					c_box[2] = bx + bw*0.5; c_box[3] = by + bh*0.5
					
					xmin = max(0,int(c_box[0]-5)); xmax = min(fwd_image_size,int(c_box[2]+5))
					ymin = max(0,int(c_box[1]-5)); ymax = min(fwd_image_size,int(c_box[3]+5))
					
					#Remove false detections over very large and very bright sources
					med_val_box = np.median(patch[ymin:ymax,xmin:xmax])
					if((med_val_box > val_med_lim[0] and c_box[5] < val_med_obj[0]) or\
					   (med_val_box > val_med_lim[1] and c_box[5] < val_med_obj[1]) or\
					   (med_val_box > val_med_lim[2] and c_box[5] < val_med_obj[2])):
						continue
					
					c_box[6] = k
					c_box[7:7+nb_param] = c_pred[offset+8:offset+8+nb_param,i,j]
					c_box[-1] = i*yolo_nb_reg+j
					c_tile[c_nb_box,:] = c_box[:]
					c_nb_box += 1
					kept_count += 1
					
			hist_count[kept_count] += 1
			
	return c_nb_box


@jit(nopython=True, cache=True, fastmath=False)
def first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, nms_thresholds, obj_thresholds):
	c_nb_box_final = 0
	is_match = 1
	c_box_size_prev = c_nb_box
	
	while(c_nb_box > 0):
		max_objct = np.argmax(c_tile[:c_box_size_prev,5])
		c_box = np.copy(c_tile[max_objct])
		c_tile[max_objct,5] = 0.0
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1; c_nb_box -= 1; i = 0
		
		for i in range(0,c_box_size_prev):
			if(c_tile[i,5] < 0.0000000001):
				continue
			IoU = fct_DIoU(c_box[:4], c_tile[i,:4])
			if(((IoU > nms_thresholds[0] and c_tile[i,5] < obj_thresholds[0]) or
			    (IoU > nms_thresholds[1] and c_tile[i,5] < obj_thresholds[1]) or
			    (IoU > nms_thresholds[2] and c_tile[i,5] < obj_thresholds[2]) or
			    (IoU > nms_thresholds[3] and c_tile[i,5] < obj_thresholds[3]))):
				c_tile[i,5] = 0.0
				c_nb_box -= 1
				
	return c_nb_box_final


@jit(nopython=True, cache=True, fastmath=False)
def second_NMS_local(boxes, comp_boxes, c_tile, direction, nms_threshold):
	c_tile[:,:] = 0.0
	nb_box_kept = 0
	
	mask_keep = np.where((boxes[:,0] > overlap) & (boxes[:,2] < patch_shift) &\
					(boxes[:,1] > overlap) & (boxes[:,3] < patch_shift))[0]
	mask_remain = np.where((boxes[:,0] <= overlap) | (boxes[:,2] >= patch_shift) |\
					(boxes[:,1] <= overlap) | (boxes[:,3] >= patch_shift))[0]
	
	nb_box_kept = np.shape(mask_keep)[0]
	c_tile[0:nb_box_kept,:] = boxes[mask_keep,:]
	shift_array = np.array([direction[0],direction[1],direction[0],direction[1]])
	comp_boxes[:,0:4] += shift_array[:]*patch_shift
	
	comp_mask_keep = np.where((comp_boxes[:,0] < fwd_image_size) & (comp_boxes[:,2] > 0) &\
					(comp_boxes[:,1] < fwd_image_size) & (comp_boxes[:,3] > 0))[0]
	
	for b_ref in mask_remain:
		found = 0
		for b_comp in comp_mask_keep:
			IoU = fct_DIoU(boxes[b_ref,:4], comp_boxes[b_comp,:4])
			if(IoU > nms_threshold and boxes[b_ref,5] < comp_boxes[b_comp,5]):
				found = 1
				break
		if(found == 0):
			c_tile[nb_box_kept,:] = boxes[b_ref,:]
			nb_box_kept += 1
		   
	return nb_box_kept

def poolingOverlap(mat, f, stride=None, pad=False):
	"""
	Overlapping pooling on 2D or 3D data.
	
	Args:
		mat (ndarray): input array to do pooling on the first 2 dimensions.
		f (int): pooling kernel size.
	Keyword Args:
		stride (int or None): stride in row/column. If None, same as <f>,
		i.e. non-overlapping pooling.
		pad (bool): pad <mat> or not. If true, pad <mat> at the end in
			y-axis with (f-n%f) number of nans, if not evenly divisible,
			similar for the x-axis.
	Returns:
		result (ndarray): pooled array.
	"""
	
	m, n = mat.shape[:2]
	if stride is None:
		stride = f
	_ceil = lambda x, y: x//y + 1
	if pad:
		ny = _ceil(m, stride)
		nx = _ceil(n, stride)
		size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
		mat_pad = np.full(size, 0)
		mat_pad[:m, :n, ...] = mat
	else:
		mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]
		
	view = asStride(mat_pad, (f, f), stride)
	
	result = np.nanmean(view, axis=(2, 3))

	return result
	
		
def asStride(arr, sub_shape, stride):
	"""
	Get a strided sub-matrices view of an ndarray.

	Args:
		arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
		sub_shape (tuple): window size: (m2, n2).
		stride (int): stride of windows in both y- and x- dimensions.
	Returns:
		subs (view): strided window view.

	See also skimage.util.shape.view_as_windows()
	"""
	s0, s1 = arr.strides[:2]
	m1, n1 = arr.shape[:2]
	m2, n2 = sub_shape[:2]

	view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
	strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
	subs = np.lib.stride_tricks.as_strided(
		arr, view_shape, strides=strides, writeable=False)

	return subs
