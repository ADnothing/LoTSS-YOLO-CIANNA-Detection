#!/usr/bin/env python

#Adrien ANTHORE, 25 Apr 2024
#Env: Python 3.6.7
#post_process.py


from aux_fct import *


if not os.path.exists("./cat_res"):
	os.makedirs("./cat_res")

fwd_path = "./fwd_res/"
content = os.listdir(fwd_path)
list_fwd = [f for f in content if f[-4:]=='.dat']

print("Number of forwards:", len(list_fwd),"\n")

def run(fwd_dat):

	flag = 1

	with open("done.txt", "r") as done_file:
		if fwd_dat in done_file.read():
			flag = 0

	if flag:
		processing(fwd_dat)

		with open("done.txt", "a") as done_file:
			done_file.write(fwd_dat + "\n")

def processing(fwd_dat):

	cat_file = "./cat_res/pred_"+fwd_dat[5:-4]+".txt"
	fits_file = fwd_dat[5:-4]+".fits"

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

	hdr["CDELT1"]*=f
	hdr["CDELT2"]*=f
	
	min_pix = np.nanpercentile(frame, 90)
	max_pix = np.nanpercentile(frame, 99)
	
	wcs_img = WCS.WCS(hdr)

	size_py = np.shape(frame)[0]
	size_px = np.shape(frame)[1]
	map_pixel_size = max(size_py, size_px)
	
	orig_offset = patch_shift - ((int(map_pixel_size/2) - int(fwd_image_size/2) + patch_shift)%patch_shift)
	
	nb_area_w = int((map_pixel_size+2*orig_offset)/patch_shift)
	nb_area_h = int((map_pixel_size+2*orig_offset)/patch_shift)

	#full_data_norm is used for the high flux in box rejection criteria
	full_data_norm = np.clip(frame, min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) / (max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)

	final_boxes = []

	pred_data = np.fromfile(fwd_path+fwd_dat, dtype="float32")

	predict = np.reshape(pred_data, (nb_area_h, nb_area_w, nb_box*(8+nb_param),yolo_nb_reg,yolo_nb_reg))
	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
	c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
	c_box = np.zeros((6+1+nb_param+1),dtype="float32")
	patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")

	box_count_per_reg_hist = np.zeros((nb_box+1), dtype="int")

	for ph in range(0,nb_area_h):
		for pw in range(0, nb_area_w):
				
			c_tile[:,:] = 0.0
			c_tile_kept[:,:] = 0.0

			p_x = pw; p_y = ph

			xmin = p_x*patch_shift - orig_offset
			xmax = p_x*patch_shift + fwd_image_size - orig_offset
			ymin = p_y*patch_shift - orig_offset
			ymax = p_y*patch_shift + fwd_image_size - orig_offset

			if(ph == 0 or ph == nb_area_h-1 or pw == 0 or pw == nb_area_w-1):
				patch[:,:] = 0.0
			else:
				if(np.count_nonzero(~np.isnan(full_data_norm[ymin:ymax,xmin:xmax]))==0):
					patch[:,:] = np.nan_to_num(full_data_norm[ymin:ymax,xmin:xmax], nan=0)
				else:
					med = np.nanmedian(full_data_norm[ymin:ymax,xmin:xmax])
					patch[:,:] = np.nan_to_num(full_data_norm[ymin:ymax,xmin:xmax], nan=med)
				
			c_pred = predict[ph,pw,:,:,:]
			c_nb_box = tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, patch, val_med_lims, val_med_obj, box_count_per_reg_hist)

			c_nb_box_final = c_nb_box
			c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, first_nms_thresholds, first_nms_obj_thresholds)
				
			out_range = 0
			if(ph < out_range or ph >= nb_area_h-out_range or pw < out_range or pw >= nb_area_w-out_range):
			    c_nb_box_final = 0
				
			final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

	final_boxes = np.reshape(np.array(final_boxes, dtype="object"), (nb_area_h, nb_area_w))

	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")

	dir_array = np.array([[-1,0],[+1,0],[0,-1],[0,+1],[-1,+1],[+1,+1],[-1,-1],[+1,-1]])
		
	#Second NMS over all the overlapping patches
	for ph in range(0, nb_area_h):
		for pw in range(0, nb_area_w):
			boxes = np.copy(final_boxes[ph,pw])
			for l in range(0,8):
				if(ph+dir_array[l,1] >= 0 and ph+dir_array[l,1] <= nb_area_h-1 and\
					pw+dir_array[l,0] >= 0 and pw+dir_array[l,0] <= nb_area_w-1):
					comp_boxes = np.copy(final_boxes[ph+dir_array[l,1],pw+dir_array[l,0]])
					c_nb_box = second_NMS_local(boxes, comp_boxes, c_tile, dir_array[l], second_nms_threshold)
					boxes = np.copy(c_tile[0:c_nb_box,:])
				
			final_boxes[ph,pw] = np.copy(boxes)
		
		
	#Convert back to full image pixel coordinates
	final_boxes_scaled = np.copy(final_boxes)
	for p_h in range(0, nb_area_h):
		box_h_offset = p_h*patch_shift - orig_offset
		for p_w in range(0, nb_area_w):
			box_w_offset = p_w*patch_shift - orig_offset
			final_boxes_scaled[p_h,p_w][:,0] = box_w_offset + final_boxes_scaled[p_h,p_w][:,0]
			final_boxes_scaled[p_h,p_w][:,2] = box_w_offset + final_boxes_scaled[p_h,p_w][:,2]
			final_boxes_scaled[p_h,p_w][:,1] = box_h_offset + final_boxes_scaled[p_h,p_w][:,1]
			final_boxes_scaled[p_h,p_w][:,3] = box_h_offset + final_boxes_scaled[p_h,p_w][:,3]

	#Order predictions by objectness score and convert to SDC scorer format
	flat_kept_scaled = np.vstack(final_boxes_scaled.flatten())
	flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:,5].argsort(),:][::-1]

	#Pixel coordinates are shift by 0.5 due to the difference of pixel coordinate system between CIANNA and classical FITS format
	x_y_flat_kept = np.copy(flat_kept_scaled[:,0:2])
	x_y_flat_kept[:,0] = (flat_kept_scaled[:,0]+flat_kept_scaled[:,2])*0.5 - 0.5
	x_y_flat_kept[:,1] = (flat_kept_scaled[:,1]+flat_kept_scaled[:,3])*0.5 - 0.5


	#Convert all the predicted parameters to the scorer format and fill non-predicted values using default settings
	cls = utils.pixel_to_skycoord(x_y_flat_kept[:,0], x_y_flat_kept[:,1], wcs_img)
	ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])

	w, h = flat_kept_scaled[:,2]-flat_kept_scaled[:,0], flat_kept_scaled[:,3]-flat_kept_scaled[:,1]

	catalog_size = np.shape(flat_kept_scaled)[0]
	box_catalog = np.zeros((catalog_size,10), dtype="float32")

	lims = np.loadtxt("train_cat_norm_lims.txt")

	box_catalog[:,[0,1]] = ra_dec_coords.T
	box_catalog[:,[2,3]] = np.array([w[:], h[:]]).T
	box_catalog[:,4] = flat_kept_scaled[:,4]
	box_catalog[:,5] = flat_kept_scaled[:,5]
	box_catalog[:,6] = np.exp(flat_kept_scaled[:,7]*(lims[0,0] - lims[0,1]) + lims[0,1])
	box_catalog[:,7] = np.exp(flat_kept_scaled[:,8]*(lims[1,0] - lims[1,1]) + lims[1,1])
	box_catalog[:,8] = np.exp(flat_kept_scaled[:,9]*(lims[2,0] - lims[2,1]) + lims[2,1])
	box_catalog[:,9] =  np.clip(np.arctan2(np.clip(flat_kept_scaled[:,11],0.0,1.0)*2.0 - 1.0, np.clip(flat_kept_scaled[:,10],0.0,1.0))*180.0/np.pi,-90,90)

	if circular_field:
		center_point = pixel_to_skycoord(size_py/2, size_px/2, wcs=wcs_img)
		box_skycoord = SkyCoord(box_catalog[:,0], box_catalog[:,1], unit="deg")
		R = (map_pixel_size/2)*hdr["CDELT2"]
		box_catalog = box_catalog[center_point.separation(box_skycoord).deg <= R]

	f = open(cat_file, 'w')
	np.savetxt(f, box_catalog, delimiter='\t')
	f.close()
	
if __name__ == "__main__":

	if not(os.path.exists("./done.txt")):
		done = open("done.txt", "w")
		done.close()

	with Pool(processes=12) as pool:
		pool.map(run, list_fwd)

	"""for fwd in list_fwd:
		run(fwd)"""
