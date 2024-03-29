#!/usr/bin/env python

#Adrien ANTHORE, 06 Feb 2024
#Env: Python 3.6.7
#fwd_process.py

from aux_fct import *

if not os.path.exists("./fwd_res"):
	os.makedirs("./fwd_res")


cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=1, out_dim=1+max_nb_obj_per_image*(7+nb_param+1),
                bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP32C_FP32A", adv_size=35)

##### YOLO parameters tuning #####

#Relative scaling of each error "type" : 
error_scales = cnn.set_error_scales(position = 36.0, size = 0.20, probability = 0.5, objectness = 2.0, parameters = 5.0)

#Various IoU limit conditions
IoU_limits = cnn.set_IoU_limits(good_IoU_lim = 0.4, low_IoU_best_box_assoc = -1.0, min_prob_IoU_lim = -0.3,
                                                                        min_obj_IoU_lim = -0.3, min_param_IoU_lim = 0.1, diff_IoU_lim = 0.4, diff_obj_lim = 0.4)

#Activate / deactivate some parts of the loss
fit_parts = cnn.set_fit_parts(position = 1, size = 1, probability = 1, objectness = 1, parameters = 1)


slopes_and_maxes = cnn.set_slopes_and_maxes(
                                        position    = cnn.set_sm_single(slope = 0.5, fmax = 6.0, fmin = -6.0),
                                        size        = cnn.set_sm_single(slope = 0.5, fmax = 1.6, fmin = -1.4),
                                        probability = cnn.set_sm_single(slope = 0.2, fmax = 6.0, fmin = -6.0),
                                        objectness  = cnn.set_sm_single(slope = 0.5, fmax = 6.0, fmin = -6.0),
                                        parameters  = cnn.set_sm_single(slope = 0.5, fmax = 1.5, fmin = -0.2))

strict_box_size = 1

nb_yolo_filters = cnn.set_yolo_params(nb_box = nb_box, nb_class = nb_class, nb_param = nb_param, max_nb_obj_per_image = max_nb_obj_per_image,
                                prior_size = prior_size, prior_noobj_prob = prior_noobj_prob, IoU_type = "DIoU", prior_dist_type = "SIZE", 
                                error_scales = error_scales, param_ind_scales = param_ind_scales, slopes_and_maxes = slopes_and_maxes, IoU_limits = IoU_limits,
                                fit_parts = fit_parts, strict_box_size = strict_box_size, min_prior_forced_scaling = 0.0, diff_flag = 1,
                                rand_startup = 0, rand_prob_best_box_assoc = 0.0, class_softmax = 1, error_type = "natural", no_override = 1, raw_output = 1)
                                
#Load here the network that you will use for inference
cnn.load("./net0_s1800.dat", 0, bin=1)

content = os.listdir(fits_path)
list_fits = [f for f in content if os.path.isfile(os.path.join(fits_path,f))]

for fits_file in list_fits:


	print("\nMosaic:", fits_file,"\n")

	hdul = fits.open(fits_path+fits_file)
	full_img = np.squeeze(hdul[0].data)
	hdr = hdul[0].header
	hdul.close()

	f = int(np.round(hdr["BMAJ"]/(1.5/3600)))
	full_img = poolingOverlap(full_img, f)
	
	hdr["NAXIS1"] = full_img.shape[0]
	hdr["NAXIS2"] = full_img.shape[1]

	hdr["CRPIX1"] = full_img.shape[0]/2
	hdr["CRPIX2"] = full_img.shape[1]/2

	hdr["CDELT1"] = hdr["CDELT1"]*f
	hdr["CDELT2"] = hdr["CDELT2"]*f
	
	min_pix = np.nanpercentile(full_img, 85)
	max_pix = np.nanpercentile(full_img, 99)
	
	wcs_img = WCS.WCS(hdr)

	map_pixel_size = np.shape(full_img)[0]
	size_px = map_pixel_size
	size_py = np.shape(full_img)[1]

	nb_area_w = int((map_pixel_size-orig_offset)/patch_shift) + 1
	nb_area_h = int((map_pixel_size-orig_offset)/patch_shift) + 1

	nb_images_all = nb_area_w*nb_area_h
	targets = np.zeros((nb_images_all,1+max_nb_obj_per_image*(7+nb_param+1)), dtype="float32")

	full_data_norm = np.clip(full_img,min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) /(max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)

	pred_all = np.zeros((nb_images_all, image_size*image_size), dtype="float32")
	patch = np.zeros((image_size,image_size), dtype="float32")

	for i_d in range(0,nb_images_all):

		p_y = int(i_d/nb_area_w)
		p_x = int(i_d%nb_area_w)

		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + image_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + image_size - orig_offset

		px_min = 0; px_max = image_size
		py_min = 0; py_max = image_size

		set_zero = 0

		if(xmin < 0):
			px_min = -xmin
			xmin = 0
			set_zero = 1
		if(ymin < 0):
			py_min = -ymin
			ymin = 0
			set_zero = 1
		if(xmax > size_px):
			px_max = image_size - (xmax-size_px)
			xmax = size_px
			set_zero = 1
		if(ymax > size_py):
			py_max = image_size - (ymax-size_py)
			ymax = size_py
			set_zero = 1

		if(set_zero):
			patch[:,:] = 0.0

		patch[px_min:px_max,py_min:py_max] = np.flip(full_data_norm[xmin:xmax,ymin:ymax],axis=0)

		if(np.count_nonzero(~np.isnan(patch[px_min:px_max,py_min:py_max]))==0):
			patch[px_min:px_max,py_min:py_max] = np.nan_to_num(patch[px_min:px_max,py_min:py_max], nan=0)
		else:
			med = np.nanmedian(patch[px_min:px_max,py_min:py_max])
			patch[px_min:px_max,py_min:py_max] = np.nan_to_num(patch[px_min:px_max,py_min:py_max], nan=med)

		pred_all[i_d,:] = patch.flatten("C")

	input_data = pred_all

	#Forward
	cnn.create_dataset("TEST", nb_images_all, input_data[:,:], targets[:,:])
	cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

	os.rename("./fwd_res/net0_0000.dat", "./fwd_res/net0_"+fits_file[:-5]+".dat")
