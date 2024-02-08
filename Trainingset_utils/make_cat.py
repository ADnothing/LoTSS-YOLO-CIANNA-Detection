#!/usr/bin/env python

#Adrien Anthore, 08 Feb 2024
#Env: Python 3.6.7
#make_cat.py

import numpy as np

from tqdm import tqdm

from astropy.io import fits
from astropy import wcs as WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.nddata import Cutout2D

from astrodendro import Dendrogram, pp_catalog


def fill_im_hole(image):
	"""
	Fill the NaN values inside an image with the median value of the image.
	If the image is full of NaN, fill the image with 0 only.

	Args:
		image (ndarray(dtype=float)): A 2D array of flux/beam values.

	Returns:
		fill_image (ndarray(dtype=float)): A 2D array with NaN values replaced by
						the overall background signal.
	"""
	
	center = np.nanmedian(image)
	
	if(np.count_nonzero(~np.isnan(image))==0):
		fill_image = np.nan_to_num(image, nan=0)
	else:
		fill_image = np.nan_to_num(image, nan=center)
	
	return fill_image


def patch_gen(image, wcs, patch_size=512, patch_shift=480, orig_offset=60):
	"""
	Divide an image from a fits file into patches with respect to its wcs.

	Args:
		image (ndarray(dtype=float)): The image from the fits file.
		wcs (astropy object wcs): The wcs corresponding to the fits header.
		patch_size (int, optional, default=512): Size of the image patches to be generated.
		patch_shift (int, optional, default=480): Amount to shift each patch by.
		orig_offset (int, optional, default=60): Offset to the original image.

	Returns:
		patches (list[astropy object Cutout2D]): List of patches, each element is an
							astropy object from which you can get attributes
							such as data (patches[i].data)
							or wcs (patches[i].wcs).
	"""

	#Get the map size in pixel from the
	#first dimension of the input data ndarray
	mapx_pixel_size = np.shape(image)[1]
	mapy_pixel_size = np.shape(image)[0]
	
	#Calculate the number of areas in the width and height directions
	nb_area_w = int((mapx_pixel_size-orig_offset)/patch_shift) + 1
	nb_area_h = int((mapy_pixel_size-orig_offset)/patch_shift) + 1
	
	#Calculate the total number of patches to be generated
	nb_patches = nb_area_w*nb_area_h
	
	#initialisation of the list of patches
	patches = []
	
	print("Patches generation...")
	
	for i_d in range(0,nb_patches):
	
		#Calculate the x and y indices for the current patch
		p_y = int(i_d/nb_area_w)
		p_x = int(i_d%nb_area_w)
		
		#Initialize the min and max x and y coordinates for the patch
		px_min = 0
		px_max = patch_size
		py_min = 0
		py_max = patch_size
		
		#Calculate the min and max x and y coordinates for the patch based on the current x and y indices	
		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + patch_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + patch_size - orig_offset

		# If any of the patch coordinates are out of bounds, 
		#set the corresponding min and/or max values
		if(xmin < 0):
			px_min = -xmin
			xmin = 0

		if(ymin < 0):
			py_min = -ymin
			ymin = 0

		if(xmax > mapx_pixel_size):
			px_max = patch_size - (xmax-mapx_pixel_size)
			xmax = mapx_pixel_size

		if(ymax > mapy_pixel_size):
			py_max = patch_size - (ymax-mapy_pixel_size)
			ymax = mapy_pixel_size


		#making the cutout and append to the list
		cutout = Cutout2D(image, (xmin+patch_size/2, ymin+patch_size/2), patch_size*u.pix, wcs=wcs.celestial)
		patches.append(cutout)
		
	print("Done.")
	
	return patches
			
#===============================================================================================================

def crea_dendrogram(fits_file, delta, promt=False):
	"""
	Generate dendrograms and catalogs from a fits file using the library astrodendro.
	(C.f. https://dendrograms.readthedocs.io/en/stable/)

	Args:
		fits_file (str): Path to the fits file.
		delta (float): Parameter from the astrodendro package. Step between iterations of the detection.
		prompt (bool, optional, default: False): If True, prompt info.

	Returns:
		None.
	"""

	#Get the image withits wcs
	hdul = fits.open(fits_file)
	image = np.squeeze(hdul[0].data)
	hdr = hdul[0].header
	hdul.close()
	wcs = WCS.WCS(hdr)
	
	#Definition of the meta data that will be use to generate the pp_catalog.
	#data_unit correspond to the data unit of the pixels
	#Spatial_scale correspond to the reolution of the instrument.
	#beam_major/beam_minor correspond to the beam size along its 2 axis.
	#see the doc for more details.
	metadata = {}
	metadata['data_unit'] = u.Jy/u.beam
	metadata['spatial_scale'] =  (hdr["CDELT2"]* u.deg).to(u.arcsec)
	metadata['beam_major'] =  (hdr["BMAJ"]* u.deg).to(u.arcsec)
	metadata['beam_minor'] = (hdr["BMIN"]* u.deg).to(u.arcsec)

	#Get the "min_value" parameter for astrodendro.
	flat_image = np.sort(image[~np.isnan(image)])
	min_val=np.percentile(flat_image, 65)
	
	if promt:
		print("min_value :",min_val)
		print("delta :", delta)
	
	patches = patch_gen(image, wcs)
	nb_patch = len(patches)

	if promt:
		print("Catalogs generation...")
	#The catalog will be stored in a single file
	#you may change the path of the destination here.
	name_file = "./dendrocat/"+fits_file.split("/")[-1][:-5]+"_DendCat.txt"
	f = open(name_file, 'w')
	f.write("_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\n")
	f.close()
	
	#min number of pixel to be taken into account to compute a dendrogram.
	#see compute function parameters.
	min_pix_nb = 2*int(np.sqrt(hdr["BMAJ"]*hdr["BMIN"]/(hdr["CDELT2"]**2)))

	for i in tqdm(range(nb_patch)):
	
		patch = patches[i].data
		patch_wcs = patches[i].wcs.celestial
	
		#Checking if the current patch is empty
		#if it's not, compute the dendrogram
		#and generate the catalog.
		if (np.count_nonzero(~np.isnan(patch))>=2*min_pix_nb):
			#Note: filling the holes (nan values) makes
			#the compute function to run properly.
			patch = fill_im_hole(patch)
			
			#Computation of the dendrogram
			#min_value refers to the minimum pixel value taken for the detection.
			#min_delta correspond to the step size for the detection itterations.
			#min_pix correspind to the minimium amount of pixel to be taken into account to compute structures.
			d = Dendrogram.compute(patch, min_value=min_val, min_delta=3*delta, min_npix=min_pix_nb, wcs=patch_wcs, verbose=promt)

			#Note: the structure we chose are the leaves.
			#taking trunk isn't irrelevant but provide different
			#results, expecially in area with artifacts or with 
			#particular morphologies of signal.
			if len(d.leaves)>0:

				cat = pp_catalog(d.leaves, metadata, verbose=promt)
				if promt:
					cat.pprint(show_unit=True, max_lines=10)
				

				#The information we keep in the catalog
				RA,DEC = patch_wcs.wcs_pix2world(cat["x_cen"][:], cat["y_cen"][:], 0)
				flux = cat["flux"][:]*1e3
				Maj = cat["major_sigma"][:]
				Min = cat["minor_sigma"][:]
				PA = cat["position_angle"][:]
				
				e = np.sqrt(1 - (Min**2)/(Maj**2))
				mask1 = e <= .95

				RA = RA[mask1]
				DEC = DEC[mask1]
				flux = flux[mask1]
				Min = Min[mask1]
				Maj = Maj[mask1]
				PA = PA[mask1]
				 
				data = np.array([RA,DEC,flux,Maj,Min, PA]).T
				fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f']
				

				#Append to the end of the file.
				f = open(name_file, 'a')
				np.savetxt(f, data, fmt=fmt , delimiter='\t', comments='')
				f.close()
			elif promt:
				print("No objects found in patch")

	if promt:	
		print("Catalogs generated !")

