#!/usr/bin/env python

#Adrien ANTHORE, 05 Feb 2024
#Env: Python 3.6.7
#final_process.py

from aux_fct import *

def update_progress(progress):

	bar_length = 100
	filled_length = int(bar_length * progress)
	bar = '#' * filled_length + '-' * (bar_length - filled_length)
	percentage = int(progress * 100)
	print(f'\rProgress : |{bar}| {percentage}% ', end='')

def check_overlap(file1, file2):

	hdul1 = fits.open(file1)
	hdul2 = fits.open(file2)

	side1 = np.max(hdul1[0].data.shape)*hdul1[0].header["CDELT2"]
	d1 = np.sqrt(2)*side1/2
	
	side2 = np.max(hdul2[0].data.shape)*hdul2[0].header["CDELT2"]
	d2 = np.sqrt(2)*side2/2

	ra1 = hdul1[0].header["CRVAL1"]
	dec1 = hdul1[0].header["CRVAL2"]
	c1 = SkyCoord(ra1, dec1, unit='deg',frame='icrs')


	ra2 = hdul2[0].header["CRVAL1"]
	dec2 = hdul2[0].header["CRVAL2"]
	c2 = SkyCoord(ra2, dec2, unit='deg',frame='icrs')

	sep = c1.separation(c2).deg

	if sep<=d1+d2:
		return True
	else:
		return False

def get_overlap_sources(cat, field_fits):
	
	field_hdul = fits.open(field_fits)
	
	side = np.max(field_hdul[0].data.shape)*field_hdul[0].header["CDELT2"]
	d = np.sqrt(2)*side/2
	
	Cra = field_hdul[0].header["CRVAL1"]
	Cdec = field_hdul[0].header["CRVAL2"]
	center = SkyCoord(Cra, Cdec, unit='deg',frame='icrs')
	
	Sra = cat[:, 0]
	Sdec = cat[:, 1]
	Spos = SkyCoord(Sra, Sdec, unit='deg',frame='icrs')
	
	sep = center.separation(Spos).deg
	
	mask_inferior_d = sep <= d
	mask_residual = ~mask_inferior_d
	
	cat_overlaped = cat[mask_inferior_d]
	cat_residual = cat[mask_residual]
	
	return cat_overlaped, cat_residual
	
def third_NMS(cat, reject, col=2):

	if col==-1:
		pass
	else:
		cat = cat[np.argsort(-cat[:,col])]
		
	i = 0
	while i<cat.shape[0]-1:

		c1 = SkyCoord(cat[i,0]*u.deg, cat[i,1]*u.deg, frame='icrs')
		c2 = SkyCoord(cat[i+1:,0]*u.deg, cat[i+1:,1]*u.deg, frame='icrs')

		sep_array = c1.separation(c2)

		excl_ind = np.where(sep_array.arcsecond < reject)[0] + i + 1

		for ind in sorted(excl_ind, reverse=True):
			cat = np.delete(cat, ind, 0)
			
		i+=1
		
	return cat


if __name__ == "__main__":

	nbsou = 0
	f = open("full_YOLOcat.txt", "w")

	list_fits = os.listdir(fits_path)
	
	list_cat = []
	for ind, f1 in enumerate(list_fits):
		flag=False
		temp_list = list_fits[:ind] + list_fits[ind+1:] 
		cat1 = "./cat_res/pred_"+f1.split("/")[-1][:-5]+".txt"
		
		list_cat.append(cat1)
		
		ctot = np.loadtxt(cat1, skiprows=1)

			for f2 in temp_list:
			fits1 = fits_path + f1
			fits2 = fits_path + f2

			cat2 = "./cat_res/pred_"+f2.split("/")[-1][:-5]+".txt"

			if check_overlap(fits1,fits2):
				flag=True
				c2 = np.loadtxt(cat2, skiprows=1)
				overlap_c2, residual_cat = get_overlap_sources(c2, fits1)
				ctot = np.vstack((ctot, overlap_c2))
				n_ovp+=overlap_c2.shape[0]
					
				f_temp = open(cat2, "w")
				np.savetxt(f_temp, residual_cat)
				f_temp.close()
					
		if flag:
			ctot = third_NMS(ctot, 6, 4)
				
		f_temp = open(cat1, "w")
		np.savetxt(f_temp, ctot)
		f_temp.close()

	for cat_name in tqdm(list_cat):
		cat = np.loadtxt(cat_name, skiprows=1)
		nbsou += cat.shape[0]
		np.savetxt(f, cat)


	f.close()
	
print("END")
print(nbsou, "sources catalogued")
