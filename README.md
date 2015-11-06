# TAIPAN tiling code

Original by Marc White
Copyright 2015, TAIPAN Survey Team (http://www.taipan-survey.org)

## Disclaimer

This code is **BETA**. Use it at your own risk. Please notify (or fork and commit!) if you locate any bugs. Any use of this code must acknowledge the TAIPAN Survey Team.

## Introduction

This is the initial version of the tiling code for the TAIPAN galaxy survey, to start in 2016. The aim of this code is to investigate the ideal tiling (that is, creation of tiles and assignment of targets to tiles) for a large, multi-fibre spectroscopic survey.

## Using the code

### Importing

The code can be imported simply into Python by placing the taipan folder in your current working directory or on your Python path, and calling:
`import taipan`
In the test scripts, `taipan.core` is imported as `tp`, and `taipan.tiling` is imported as `tl`.

### Running the test scripts

**NOTE:** Some of the test scripts may be periodically modified to test things other than what their name suggests (e.g. computing target difficulties). Be aware!

The test scripts are designed to run on an arbitrary catalogue of astronomical targets, guides and standards. It is **your** responsibility to update the test script code to load the data you wish to test on. To request access to the catalogues used for testing during development, please contact me directly.