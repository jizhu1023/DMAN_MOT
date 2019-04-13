Multiple Object Tracking Challenge
==================================
---- http://motchallenge.net -----
----------------------------------

Version 1.1

This development kit provides scripts to evaluate tracking results.
Please report bugs to anton.milan@adelaide.edu.au


Requirements
============
- MATLAB
- Benchmark data 
  e.g. 2DMOT2015, available here: http://motchallenge.net/data/2D_MOT_2015/
  
  

Usage
=====

To compute the evaluation for the included demo, which corresponds to 
the results of the CEM tracker (continuous energy minimization) on the 
training set of the '2015 MOT 2DMark', start MATLAB, cd into devkit 
directory and run

	benchmarkDir = '../data/2DMOT2015/train/';
	allMets = evaluateTracking('c2-train.txt', 'res/data/', benchmarkDir);

Replace the value for benchmarkDir accordingly.

You should see the following output (be patient, it may take a minute):

Sequences: 
    'TUD-Stadtmitte'
    'TUD-Campus'
    'PETS09-S2L1'
    'ETH-Bahnhof'
    'ETH-Sunnyday'
    'ETH-Pedcross2'
    'ADL-Rundle-6'
    'ADL-Rundle-8'
    'KITTI-13'
    'KITTI-17'
    'Venice-2'
Evaluating ...
        ... TUD-Stadtmitte
*** 2D (Bounding Box overlap) ***
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 60.9  94.0  0.25| 10   5   4   1|   45   452    7    6|  56.4  65.4  56.9

        ... TUD-Campus
*** 2D (Bounding Box overlap) ***
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 58.2  94.1  0.18|  8   1   6   1|   13   150    7    7|  52.6  72.3  54.3

..................
..................
..................

 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 45.3  71.7  1.30|500  81 161 258| 7129 21842  220  338|  26.8  72.4  27.4 



Details
=======
The evaluation script accepts 3 arguments:

1)
sequence map (e.g. `c2-train.txt` contains a list of all sequences to be 
evaluated in a single run. These files are inside the ./seqmaps folder.

2)
The folder containing the tracking results. Each one should be saved in a
separate .txt file with the name of the respective sequence (see ./res/data)

3)
The folder containing the benchmark sequences.

The results will be shown for each individual sequence, as well as for the
entire benchmark.




Directory structure
===================
	

./res
----------
This directory contains 
  - the tracking results for each sequence in a subfolder data  
  - eval.txt, which shows all metrics for this demo
  
  
  
./utils
-------
Various scripts and functions used for evaluation.


./seqmaps
---------
Sequence lists for different benchmarks




Version history
===============
1.1.1 - Oct 10, 2016
  - Included camera projections scripts
	
1.1 - Feb 25, 2016
  - Included evaluation for the new MOT16 benchmark

1.0.5 - Nov 10, 2015
  - Fixed bug where result has only one frame
  - Fixed bug where results have extreme values for IDs
  - Results may now contain invalid frames, IDs, which will be ignored

1.0.4 - Oct 08, 2015
  - Fixed bug where result has more frames than ground truth

1.0.3 - Jul 04, 2015
  - Removed spurious frames from ETH-Pedcross2 result (thanks Nikos)
  
1.0.2 - Mar 11, 2015
  - Fix to exclude small bounding boxes from ground truth
  - Special case of empty mapping fixed

1.0.1 - Feb 06, 2015
  - Fixes in 3D evaluation (thanks Michael)

1.00 - Jan 23, 2015
  - initial release