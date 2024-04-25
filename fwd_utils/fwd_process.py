#!/usr/bin/env python

#Adrien ANTHORE, 25 Apr 2024
#Env: Python 3.6.7
#fwd_process.py

from aux_fct import *

sys.path.insert(0,glob.glob("/minerva/Anthore/CIANNA/src/build/lib.*/")[-1])
import CIANNA as cnn

if not os.path.exists("./fwd_res"):
	os.makedirs("./fwd_res")


cnn.init(in_dim=i_ar([fwd_image_size,fwd_image_size]), in_nb_ch=1, out_dim=1+max_nb_obj_per_image*(7+nb_param),
	bias=0.1, b_size=5, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A", adv_size=30, inference_only=1)

nb_yolo_filters = cnn.set_yolo_params(raw_output=0)

#Load here the network that you will use for inference
if(not os.path.isfile(data_path+"YOLO_CIANNA_ref_SDC1_i3600_s480k_MINERVA_Cornu2024.dat")):
		os.system("wget -P %s https://share.obspm.fr/s/GELFJjBFtwC4g5A/download/YOLO_CIANNA_ref_SDC1_i3600_s480k_MINERVA_Cornu2024.dat"%(data_path))

cnn.load(data_path+"YOLO_CIANNA_ref_SDC1_i3600_s480k_MINERVA_Cornu2024.dat", 0, bin=1)

content = os.listdir(fits_path)
list_fits = [f for f in content if os.path.isfile(os.path.join(fits_path,f))]

for fits_file in list_fits:


	print("\nMosaic:", fits_file,"\n")

	hdul = fits.open(fits_path+fits_file)
	full_img = np.squeeze(hdul[0].data)
	hdr = hdul[0].header
	hdul.close()

	f = int(np.round(hdr["BMAJ"]/(1.5/3600)))
	frame = poolingOverlap(full_img, f)

	hdr["NAXIS1"] = frame.shape[0]
	hdr["NAXIS2"] = frame.shape[1]

	hdr["CRPIX1"] = frame.shape[0]/2
	hdr["CRPIX2"] = frame.shape[1]/2

	hdr["CDELT1"] *= f
	hdr["CDELT2"] *= f
	
	min_pix = np.nanpercentile(frame, 90)
	max_pix = np.nanpercentile(frame, 95)

	wcs_img = WCS.WCS(hdr)

	size_py = np.shape(frame)[0]
	size_px = np.shape(frame)[1]
	map_pixel_size = max(size_py, size_px)

	orig_offset = patch_shift - ((int(map_pixel_size/2) - int(fwd_image_size/2) + patch_shift)%patch_shift)

	nb_area_w = int((size_px+2*orig_offset)/patch_shift)
	nb_area_h = int((size_py+2*orig_offset)/patch_shift)
	nb_images_all = int(nb_area_w*nb_area_h)

	targets = np.zeros((nb_images_all,1+max_nb_obj_per_image*(7+nb_param)), dtype="float32")

	full_data_norm = np.clip(frame,min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) /(max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)

	pred_all = np.zeros((nb_images_all,fwd_image_size*fwd_image_size), dtype="float32")
	patch = np.zeros((fwd_image_size,fwd_image_size), dtype="float32")

	for i_d in range(0,nb_images_all):

		p_y = int(i_d/nb_area_h)
		p_x = int(i_d%nb_area_w)

		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + fwd_image_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + fwd_image_size - orig_offset

		px_min = 0; px_max = fwd_image_size
		py_min = 0; py_max = fwd_image_size

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
			px_max = fwd_image_size - (xmax-size_px)
			xmax = size_px
			set_zero = 1
		if(ymax > size_py):
			py_max = fwd_image_size - (ymax-size_py)
			ymax = size_py
			set_zero = 1

		if(set_zero):
			patch[:,:] = 0.0

		patch[py_min:py_max,px_min:px_max] = full_data_norm[ymin:ymax,xmin:xmax]

		if(np.count_nonzero(~np.isnan(patch[py_min:py_max,px_min:px_max]))==0):
			patch[py_min:py_max,px_min:px_max] = np.nan_to_num(patch[py_min:py_max,px_min:px_max], nan=0)
		else:
			med = np.nanmedian(patch[py_min:py_max,px_min:px_max])
			patch[py_min:py_max,px_min:px_max] = np.nan_to_num(patch[py_min:py_max,px_min:px_max], nan=med)

		pred_all[i_d,:] = patch.flatten("C")

	input_data = pred_all

	#Forward
	cnn.create_dataset("TEST", nb_images_all, input_data[:,:], targets[:,:])
	cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

	os.rename("./fwd_res/net0_0000.dat", "./fwd_res/net0_"+fits_file[:-5]+".dat")
