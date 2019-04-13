%benchmarkDir = '/home/tensend/data/2DMOT2015/train/';
%allMets = evaluateTracking('c2-train.txt', 'res/data/', benchmarkDir);
%fprintf('\nnweights  = [%.2f, %.2f, %.2f]; Weights for combining correlation filter responses\n', nweights)

 benchmarkDir = '/home/tensend/data/MOT16/train/';
 allMets = evaluateTracking('c5-train.txt', '/home/tensend/MDP_Tracking_vllab/results/cong/', benchmarkDir);
 %fprintf('\nnweights  = [%.2f, %.2f, %.2f]; Weights for combining correlation filter responses\n', nweights)

 
 %Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 %29.0  79.8  2.19| 54   7  19  28| 1311 12660   55   87|  21.3  74.4  21.6 
