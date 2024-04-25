#!/usr/bin/env python

#Adrien ANTHORE, 25 Apr 2024
#Env: Python 3.6.7
#final_process.py

from aux_fct import *

def check_overlap(file1, file2):

	hdul1 = fits.open(file1)
	hdul2 = fits.open(file2)

	side1 = np.max(hdul1[0].data.shape)*hdul1[0].header["CDELT2"]
	d1 = side1/2
	
	side2 = np.max(hdul2[0].data.shape)*hdul2[0].header["CDELT2"]
	d2 = side2/2

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
	"""
	Extract sources from a catalog that overlap and do not overlap with a field.

	Args:
		cat (numpy.ndarray): Input catalog containing source positions.
		field_fits (str): File path of the FITS file representing the field.

	Returns:
		tuple: Two numpy arrays - the first containing sources that overlap with the field,
			and the second contains sources that do not overlap.

	This function extracts sources from a catalog based on whether they overlap with
	a specified field represented by a FITS file. It calculates the diagonal distance
	of the field and compares it with the separation between the center of the field
	and the positions of sources in the catalog.
	"""
	
	field_hdul = fits.open(field_fits)
	
	side = np.max(field_hdul[0].data.shape)*field_hdul[0].header["CDELT2"]
	d = 1.1*(side/2)
	
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

def third_NMS(f1, list_fits, path, reject, col=7):
	"""
	Apply a Non-Max Suppression (NMS) on catalogs from images that have an overlap.
	The method use for the suppression is a Nearest Neigbourg suppression.
	
	Args:
		f1 (str): File name of the first FITS file.
		list_fits (list): List of file names of FITS files to compare with.
		path (str): Path to the FITS files.
		reject (float): Threshold distance for rejecting redundant sources, in arcseconds.
		col (int, optional): Index of the column in the catalog containing the confidence score. Default is 7.

	The resulting catalog from the image in the fits f1 is saved as a text file in the 'Catalogs' directory.
	"""
	
	fits1 = path + f1
	cat1 = "./cat_res/pred_"+f1[:-5]+".txt"
	c1 = np.loadtxt(cat1, comments='#')
		
	for f2 in tqdm(list_fits):
		
		fits2 = path + f2
		cat2 = "./cat_res/pred_"+f2[:-5]+".txt"
			
		if check_overlap(fits1,fits2):
			
			overlap_c1, residual_c1 = get_overlap_sources(c1, fits2)
			
			c2 = np.loadtxt(cat2, comments='#')
			overlap_c2, residual_c2 = get_overlap_sources(c2, fits1)
					
			excl_ind = []
				
			for idx, source in enumerate(overlap_c1):
					
				pos = SkyCoord(source[0]*u.deg, source[1]*u.deg, frame='icrs')
				pos_test = SkyCoord(overlap_c2[:,0]*u.deg, overlap_c2[:,1]*u.deg, frame='icrs')
					
				sep_array = pos.separation(pos_test)
					
				test_ind = np.where(sep_array.arcsecond < reject)[0]
					
				if any(source[col] >= overlap_c2[test_ind,col]):
					excl_ind.append(idx)
						
			for ind in sorted(excl_ind, reverse=True):
				overlap_c1 = np.delete(overlap_c1, ind, 0)
						
					
			c1 = np.vstack((residual_c1, overlap_c1))
			
		
	f = open("./Catalogs/"+f1[:-5]+"_Cat.txt", "w")
	np.savetxt(f, c1, delimiter='\t')
	f.close()


def overlap(fits_file):

	flag = 1
	temp_list = [f2 for f2 in list_fits if f2 != fits_file]

	with open("NMSed.txt", "r") as done_file:
		if fits_file in done_file.read():
			flag = 0

	if fits_file[-5:] != ".fits":
		flag = 0

	if flag:
		third_NMS(fits_file, temp_list, "../LoTSS/", 6, col=5)

		with open("NMSed.txt", "a") as done_file:
			done_file.write(fits_file + "\n")


if __name__ == "__main__":

	if not(os.path.exists("./Catalogs/")):
		os.makedirs("./Catalogs/")

	if not(os.path.exists("./NMSed.txt")):
		done = open("NMSed.txt", "w")
		done.close()

	global list_fits
	done_cats = os.listdir("./cat_res")
	list_fits = [f[5:-4] + ".fits" for f in done_cats]
	print(list_fits)
	with Pool(processes=12) as pool:
		pool.map(overlap, list_fits)

print("END")
