# The Microlensing Driftscan Project

Driftscan techniques have not been widely used so far for fast photometry, and a fair amount of development of new techniques has been required to extract photometric time series with sub-arcsecond sampling from photometric data.
The functions presented in this repository are designed to accomplish the drift scan time series extraction, and work on a single selected field of interest without dithering patterns.

For this science case, the input to the pipeline is two pointed optical colour images of a field (in this case DECam g and r-band images) and a series of 20 second rapid-cadence (VR) images of the field with a constant drift rate in Declination. 
The output is a catalogue of light curves for stars across the field, with brightness measurements subsampled along the drifted image, producing an effective candence much shorter than the exposure duration. 
The driftscan image, or DSI, can be searched for sub-second transient signals. The Driftscan techique is especially powerful in searching for very short duration microlensing signals produced by a population of asteroid-mas primordial black holes.

Other science applications include the ability to screen for milli-second optical transient phenomena (e.g counterparts to GRBs, FRBs) by distinguishing between DSI trails from field stars, and point sources from out-of-nowhere signals.

# Contents

1. Main Catalogue Generation in MainCat.py
   
Generates a deep catalogue from pointed g, r images from the sky. Use is for linking coordinates of DSIs, WCS corrections across the detector plane.

2. Reference Star Selection in RefStar.py

Finds a number of bright, isolated reference stars per CCD chip. Use is for WCS corrections and measuring DSI singal-to-noise contamination from atmospheric seeing.

3. DSI centroiding in DriftAstrom.py
   
Locates the centroids of DSI on CCD chip by convolution with a DSI template and centroiding function. Centroids are matched to reference stars to WCS correct each exposure for tangent plane distortions. 

4. DSI extraction in DriftExtract.py
   
Extraction function is run on all located DSI, and then matched to reference star with the most similar drift patten. The best match reference star is used to normalise out atmospheric seeing pattens in the DSI.

5. Point Source detection in DriftClassify.py WORK IN PROGRESS
    
Uses the reference catalgoue and the DSI shape to distinguish between drifted stars, millisecond optical transients and other phenomena (like satellites)

6. Drift Recalibration of Astrometry WORK IN PROGRESS

# Generalised Flow of Pipeline

![driftIO](https://github.com/rgkey/DriftScanning/assets/45152240/8d6e5026-2efe-4a16-afe5-10efcb837edc)
