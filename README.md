# LoTSS YOLO Detection

## Introduction

This project takes place in an international effort to develop new data processing and analysis tools for the upcoming Square Kilometer Array (SKA), the first light of which is scheduled for 2028.
This instrument will reach an unprecedented sensitivity, allowing it to set new constraints on the early stages of the universe and better constrain the evolution of astronomical objects over cosmological times. However, the analysis of the data produced by the SKA will be very challenging. The current forecasts predict up to 700 PB of archived data per year and a raw data output of around 1 TB per second, which must be processed in real time to produce science data products. With such a data rate and with data products that are large and highly dimensional, current widely adopted analysis methods exhibit significant limitations. For this reason, new innovative analysis methods must be developed alongside the SKA deployment. For this, the astronomical community makes use of current instruments that are considered "pathfinders" or "precursors" for the SKA (e.g., ASKAP, MeerKAT, LOFAR, etc.).

In this context, the SKAO (SKA Observatory) started a series of data challenges (SDC), intending to provide simulated data products that should be representative of the SKA.
These challenges aim to compare analysis tools on controlled datasets and encourage the development of new data analysis methods.
While these SDCs seek to represent a variety of analysis tasks, the first two editions of the challenges (SDC1 Bonaldi et al. (2020) and SDC2 Hartley et al. (2023)) focused on source detection and characterization.
The first edition was about detecting sources in simulated 2D continuum images, and the second edition in a simulated 3D HI emission cube.

To participate in the SDC2, members of the MINERVA project from the Paris Observatory developed a deep-learning approach for source detection and characterization that would be applied to both 2D and 3D datasets.
This new innovative method demonstrated state-of-the-art performance on both types of simulated data.
The team noticeably reached first place in the SDC2 and achieved much better results than all the solutions submitted during the SDC1.
The method must now be generalized on observational radio datasets, which can be done using various surveys from precursor instruments.

This project lies in this global effort of applying the method to existing radio datasets.
I noticeably focused on trying to apply the detector trained on the first SDC on observational continuum surveys, namely the LoFAR Two-metre Sky Survey (LoTSS DR2 Shimwell, et al. (2022)).
This survey is already associated with a source catalog derived using the same classical detection method.
This catalog will be considered as a reference to which our detection results will be systematically compared.
Because MINERVA's source detector is based on supervised machine learning, it must be trained for the survey it will be applied to.
This can be done in two different ways: i) by using simulated examples, or ii) by using actual observations with labels from another method or confirmed observationally using other data.

This project is separated into two steps.
The first step is to evaluate MINERVA's method capabilities on the selected survey by performing source detection using the network trained on the simulated data from the SDC1.
By doing this, we will evaluate if the SDC1 is a good enough approximation of observational data for the LoTSS.
The second step consists of using observational examples to perform complementary network training to better account for the survey's specificities.
This approach requires the construction of a comprehensive and high-quality source catalog based on a different method than the one used to define the reference source catalog.

## Content

### Trainingset_utils

This folder contains the function used to build the training dataset to perform the fine training of our neural network.

| File Name            | Description                                                   |
| -------------------- | --------------------------------------------------------------|
| make_cat.py          | Contains the functions to build the initial catalogs.         |
| corr_cat.py          | Contains the functions to clean the produced catalogs         |
| CrossMatch.py        | Contains the function to perform cross-matches                |
| main.py              | Main python file                                              |
| config.ini           | Example of the config file (used as a parameter of main.py)   |
| config_LoTSS.ini     | Config file used for LoTSS                                    |

#### make_cat.py

fill_im_hole(image)

  Fill the NaN values inside an image with the median value of the image.
	If the image is full of NaN, fill the image with 0 only.

  Args:
		image (ndarray(dtype=float)): A 2D array of flux/beam values.

  Returns:
		fill_image (ndarray(dtype=float)): A 2D array with NaN values replaced by
						the overall background signal.
      
========================================================================================

patch_gen(image, wcs, patch_size=512, patch_shift=480, orig_offset=60)

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

========================================================================================

crea_dendrogram(fits_file, delta, promt=False)

Generate dendrograms and catalogs from a fits file using the library astrodendro.
(C.f. https://dendrograms.readthedocs.io/en/stable/)

Args:
	fits_file (str): Path to the fits file.
	delta (float): Parameter from the astrodendro package. Step between iterations of the detection.
	prompt (bool, optional, default: False): If True, prompt info.

Returns:
	None.

========================================================================================

#### coor_cat.py

update_progress(progress)

Function to update the progress bar in the console.

========================================================================================

Rexcl(flux, P1, R1, P2, R2)

Rejection radius function:
Around bright sources, we often find artifacts due to interferometry.
To reject those artifacts, we defined a radius that is a function of
the integrated flux of the source studied. As the extent of the artifacts
seems to scale very fast with the integrated flux, we chose an exponential
law which is parameterized by typical integrated flux values "P" (low and high)
with their expected rejection radius "R". The parameters P and R are
chosen from guesses by analyzing the fields studied.

Args:
	flux (float): Integrated flux value for which the rejection radius is calculated.
	P1 (float): Typical low integrated flux (sources with no artifacts).
	R1 (float): Typical radius of exclusion for the low integrated flux.
	P2 (float): Typical high integrated flux (sources with a lot of artifacts).
	R2 (float): Typical radius of exclusion for high integrated flux sources.

Returns:
	float: Rejection radius calculated based on the input parameters.

========================================================================================

clean_cat(name_file, res, R1, P1, R2, P2, survey)

Clean the catalog generated with crea_dendrogram to suppress multiple detections
as well as supposedly false detections.
This process results in the writing of 2 files:
	- The cleaned catalog of a field (overwriting the input file).
	- The "To Test Sources Catalog" (TTSC) contains all rejected sources that could be True detections.
		
The name of the TTSC is: TTSC_{survey}.txt

Args:
	name_file (str): File path of the catalog generated with crea_dendrogram.
	res (float): Resolution of the instrument.
	R1 (float): Parameter R1 for rejection radius calculation.
	P1 (float): Parameter P1 for rejection radius calculation.
	R2 (float): Parameter R2 for rejection radius calculation.
	P2 (float): Parameter P2 for rejection radius calculation.
	survey (str): Name of the survey.

Returns:
	None.

Information about the number of sources cleaned and sources to test are printed at the end.

========================================================================================

check_overlap(file1, file2)

Check if two FITS files have spatial overlap.

Args:
	file1 (str): File path of the first FITS file.
	file2 (str): File path of the second FITS file.

Returns:
	bool: True if the two files have spatial overlap, False otherwise.

This function checks whether two FITS files have spatial overlap. It computes
the diagonal distance of the images and compares it with the separation between
their central coordinates. If the separation is less than or equal to the diagonal
distance, it indicates that the images overlap.

========================================================================================

get_overlap_sources(cat, field_fits)

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

========================================================================================

clean_overlap(data, res, R1, P1, R2, P2, survey)

Clean a catalog by removing multiple detections and excluding artifacts
around bright sources based on spatial overlap.
This function has the same process as "clean_cat" but is 
specific for the case when we clean overlapping regions.
The expected data in output is typically the output "cat_overlaped"
from the function "get_overlap_sources".

Args:
	data (numpy.ndarray): Input catalog containing source information.
	res (float): Resolution of the instrument.
	R1 (float): Parameter R1 for rejection radius calculation.
	P1 (float): Parameter P1 for rejection radius calculation.
	R2 (float): Parameter R2 for rejection radius calculation.
	P2 (float): Parameter P2 for rejection radius calculation.
	survey (str): Name of the survey.

Returns:
	numpy.ndarray: Cleaned catalog containing sources that pass the cleaning criteria.

========================================================================================

 flux_NMS(cat, reject)

 Perform a Non-Maximum Suppression (NMS) process on a catalog.
The parameter taken into account for the suppression is the integrated flux of the sources.

Args:
	cat (numpy.ndarray): Input catalog containing source information.
	reject (float): Separation threshold for exclusion.

Returns:
	numpy.ndarray: Catalog after the Third NMS process.

#### CrossMatch.py

match_coord(cat2cross, refcoord, sep)

Perform coordinate matching between two catalogs.

Args:
	cat2cross (numpy.ndarray): Catalog to cross-match.
	refcoord (numpy.ndarray): Reference catalog for cross-matching.
	sep (float): Separation threshold for matching in radians.

Returns:
	Tuple[numpy.ndarray, numpy.ndarray]: Matched indices in 'cat2cross' and corresponding indices in 'refcoord'.

This function performs coordinate matching between two catalogs using the Haversine formula.
It compares the celestial coordinates of sources in 'cat2cross' with the coordinates in 'refcoord'.
If the separation between a source in 'cat2cross' and any source in 'refcoord' is within the specified
separation threshold ('sep'), the indices of the matched sources in 'cat2cross' and 'refcoord' are returned.

========================================================================================

NN_Xmatch(cat_test, cat_ref, maxsep, colRA_test, colDec_test, colRA_ref, colDec_ref, sort_test_by="")

Perform nearest-neighbor cross-matching between two catalogs.

Args:
	cat_test (astropy.table.Table): Catalog to be tested.
	cat_ref (astropy.table.Table): Reference catalog for cross-matching.
	maxsep (Quantity): Maximum separation allowed for matching. (dimension of an angle)
	colRA_test (str): Column name containing Right Ascension in the catalog to be tested.
	colDec_test (str): Column name containing Declination in the catalog to be tested.
	colRA_ref (str): Column name containing Right Ascension in the reference catalog.
	colDec_ref (str): Column name containing Declination in the reference catalog.
	sort_test_by (str, optional): Column name for sorting the catalog to be tested. Defaults to "".

Returns:
	Tuple[astropy.table.Table, astropy.table.Table]: Matched sub-catalogs from the test and reference catalogs.

This function performs nearest-neighbor cross-matching between a catalog and a reference catalog.
The catalogs are assumed to be Astropy Tables, and the matching is done based on Right Ascension (RA) and Declination (Dec).
The maximum separation allowed for a match is specified by the 'maxsep' parameter.
