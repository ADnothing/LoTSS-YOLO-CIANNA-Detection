#!/usr/bin/env python

#Adrien ANTHORE, 05 Feb 2024
#Env: Python 3.6.7
#post_process_LoTSS.py

from aux_fct import *

if not os.path.exists("./cat_res"):
	os.makedirs("./cat_res")

fwd_path = "./fwd_res/"
content = os.listdir(fwd_path)
list_fwd = [f for f in content if f[-4:]=='.dat']

c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param)),dtype="float32")
c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param)),dtype="float32")
c_box = np.zeros((6+1+nb_param),dtype="float32")
cat_boxes = np.zeros((max_nb_obj_per_image,5))

patch = np.zeros((image_size,image_size), dtype="float32")

box_count_per_reg_hist = np.zeros((nb_box+1), dtype="int")

print("Number of forwards:", len(list_fwd),"\n")

for ind, fwd_dat in enumerate(list_fwd):

	print("Data: ", fwd_dat,"\t","%i/%i"%(ind+1, len(list_fwd)))

	start = time.time()

	cat_file = "./cat_res/pred_"+fwd_dat[5:-4]+".txt"

	hdul = fits.open(LoTSS_path+fwd_dat[11:-4]+".fits")
	full_img = np.squeeze(hdul[0].data)
	wcs_img = WCS(hdul[0].header)
	hdul.close()

	min_pix = np.nanpercentile(full_img, 70.)
	max_pix = np.nanpercentile(full_img, 99.)

	full_data_norm = np.clip(full_img,min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) /(max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)

	map_pixel_size = np.shape(full_img)[0]
	size_px = map_pixel_size
	size_py = np.shape(full_img)[1]

	nb_area_w = int((map_pixel_size-orig_offset)/patch_shift) + 1
	nb_area_h = int((map_pixel_size-orig_offset)/patch_shift) + 1

	nb_images_all = nb_area_w*nb_area_h



	#Post-process

	pred_data = np.fromfile(fwd_path+fwd_dat, dtype="float32")

	predict = np.reshape(pred_data, (nb_area_h, nb_area_w, nb_box*(8+nb_param),yolo_nb_reg,yolo_nb_reg))

	final_boxes = []

	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param)),dtype="float32")
	c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param)),dtype="float32")
	c_box = np.zeros((6+1+nb_param),dtype="float32")
	patch = np.zeros((image_size, image_size), dtype="float32")

	box_count_per_reg_hist = np.zeros((nb_box+1), dtype="int")

	print("1st NMS:")
	for ph in tqdm(range(0,nb_area_h)):
		for pw in range(0, nb_area_w):
			c_tile[:,:] = 0.0
			c_tile_kept[:,:] = 0.0

			xmin = pw*patch_shift - orig_offset
			xmax = pw*patch_shift + image_size - orig_offset
			ymin = ph*patch_shift - orig_offset
			ymax = ph*patch_shift + image_size - orig_offset

			if(ph == 0 or ph == nb_area_h-1 or pw == 0 or pw == nb_area_w-1):
				patch[:,:] = 0.0
			else:
				if(np.count_nonzero(~np.isnan(full_data_norm[xmin:xmax,ymin:ymax]))==0):
					temp = np.nan_to_num(full_data_norm[xmin:xmax,ymin:ymax], nan=0)
				else:
					med = np.nanmedian(full_data_norm[xmin:xmax,ymin:ymax])
					temp = np.nan_to_num(full_data_norm[xmin:xmax,ymin:ymax], nan=med)
					
				patch[:,:] = np.flip(temp,axis=0)

			c_pred = predict[ph,pw,:,:,:]
			c_nb_box = tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, prob_obj_edges, patch, val_med_lims, val_med_obj, box_count_per_reg_hist, -1)

			c_nb_box_final = c_nb_box
			c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, box_prior_class, first_nms_thresholds, first_nms_obj_thresholds)

			c_tile = np.copy(c_tile_kept)
			c_tile_kept[:,:] = 0.0
			c_nb_box_final = remove_extended(patch, c_tile, c_tile_kept, c_box, c_nb_box_final, nb_box, val_med_lims, val_med_obj)

			final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

	final_boxes = np.reshape(np.array(final_boxes, dtype="object"), (nb_area_h, nb_area_w))

	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param)),dtype="float32")

	dir_array = np.array([[-1,0],[+1,0],[0,-1],[0,+1],[-1,+1],[+1,+1],[-1,-1],[+1,-1]])
	shift_array = np.array([[-1, 0,-1, 0],[ 1, 0, 1, 0],[ 0, 1, 0, 1],[ 0,-1, 0,-1],\
	                                                [-1,-1,-1,-1],[ 1,-1, 1,-1],[-1, 1,-1, 1],[ 1, 1, 1, 1]])


	#Second NMS over all the overlapping patches
	print("2nd NMS:")
	for ph in tqdm(range(0, nb_area_w)):
		for pw in range(0, nb_area_h):

			boxes = np.copy(final_boxes[ph,pw])

			for l in range(0,8):
				if(ph+dir_array[l,0] >= 0 and ph+dir_array[l,0] <= nb_area_h-1 and\
					pw+dir_array[l,1] >= 0 and pw+dir_array[l,1] <= nb_area_w-1):

					comp_boxes = np.copy(final_boxes[ph+dir_array[l,0],pw+dir_array[l,1]])
					c_nb_box = second_NMS_local(boxes, comp_boxes, c_tile, shift_array[l], second_nms_threshold)
					boxes = np.copy(c_tile[0:c_nb_box,:])

			final_boxes[ph,pw] = np.copy(boxes)


	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param)),dtype="float32")

	boxes = np.copy(c_tile[0:0,:])

	for pw in range(0, nb_area_h):
		final_boxes[0,pw] = np.copy(boxes)
		final_boxes[nb_area_h-1,pw] = np.copy(boxes)
	for ph in range(0, nb_area_w):
		final_boxes[ph,0] = np.copy(boxes)
		final_boxes[ph,nb_area_w-1] = np.copy(boxes)

	final_boxes_scaled = np.copy(final_boxes)
	for p_h in range(0, nb_area_h):
		box_w_offset = p_h*patch_shift - orig_offset
		for p_w in range(0, nb_area_w):
			box_h_offset = p_w*patch_shift - orig_offset
			final_boxes_scaled[p_h,p_w][:,0] = box_w_offset + final_boxes_scaled[p_h,p_w][:,0]
			final_boxes_scaled[p_h,p_w][:,2] = box_w_offset + final_boxes_scaled[p_h,p_w][:,2]
			final_boxes_scaled[p_h,p_w][:,1] = box_h_offset - final_boxes_scaled[p_h,p_w][:,1] + image_size
			final_boxes_scaled[p_h,p_w][:,3] = box_h_offset - final_boxes_scaled[p_h,p_w][:,3] + image_size


	flat_kept_scaled = np.vstack(final_boxes_scaled.flatten())
	flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:,5].argsort(),:][::-1]


	x_y_flat_kept = np.copy(flat_kept_scaled[:,0:2])
	x_y_flat_kept[:,0] = np.clip((flat_kept_scaled[:,0]+flat_kept_scaled[:,2])*0.5 - 0.5, 0, map_pixel_size)
	x_y_flat_kept[:,1] = np.clip((flat_kept_scaled[:,1]+flat_kept_scaled[:,3])*0.5 - 0.5, 0, map_pixel_size)

	#Result file from the training, change the path accordingly!
	lims = np.loadtxt("train_cat_lims.txt")

	cls = utils.pixel_to_skycoord(x_y_flat_kept[:,0], x_y_flat_kept[:,1], wcs_img)
	ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])

	w, h = flat_kept_scaled[:,2]-flat_kept_scaled[:,0], flat_kept_scaled[:,3]-flat_kept_scaled[:,1]

	catalog_size = np.shape(flat_kept_scaled)[0]

	box_catalog = np.zeros((catalog_size,10), dtype="float32")
	box_catalog[:,[0,1]] = ra_dec_coords.T
	box_catalog[:,[2,3]] = np.array([w[:], h[:]]).T
	box_catalog[:,4] = flat_kept_scaled[:,4]
	box_catalog[:,5] = flat_kept_scaled[:,5]
	box_catalog[:,6] = np.exp(flat_kept_scaled[:,7]*(lims[0,0] - lims[0,1]) + lims[0,1])
	box_catalog[:,7] = np.exp(flat_kept_scaled[:,8]*(lims[1,0] - lims[1,1]) + lims[1,1])
	box_catalog[:,8] = np.exp(flat_kept_scaled[:,9]*(lims[2,0] - lims[2,1]) + lims[2,1])
	box_catalog[:,9] =  np.clip(np.arctan2(np.clip(flat_kept_scaled[:,11],0.0,1.0)*2.0 - 1.0, np.clip(flat_kept_scaled[:,10],0.0,1.0))*180.0/np.pi,-90,90)

	f = open(cat_file, 'w')
	np.savetxt(f, box_catalog, delimiter='\t')
	f.close()

	end = time.time()

	print("time:", end - start,"s")
	print("Number of sources detected:", box_catalog.shape[0],"\n")

