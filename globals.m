% global parameters (modified based on the original MDP code)
function opt = globals()

opt.root = pwd;

% path for MOT benchmark
mot_paths = {'data/', ''};
for i = 1:numel(mot_paths)
    if exist(mot_paths{i}, 'dir')
        opt.mot = mot_paths{i};
        break;
    end
end

opt.mot2d = 'MOT16';
opt.results_dir = 'results/';
opt.mot2d_train_seqs = {'MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'};
opt.mot2d_train_nums = [600, 1050, 837, 525, 654, 900, 750];
opt.mot2d_test_seqs = {'MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'};
opt.mot2d_test_nums = [450, 1500, 1194, 500, 625, 900, 750];

addpath(fullfile(opt.root, 'devkit', 'utils'));
addpath([opt.root '/3rd_party/libsvm-3.20/matlab']);
addpath([opt.root '/3rd_party/Hungarian']);

% tracking parameters
opt.aspect_ratio_thre = 0.6;    % aspect ratio threshold in target association
opt.distance_thre = 2;        % distance threshold in target association, multiple of the width of target
opt.det_overlap_thre = 0.5;        % overlap with detection
opt.tracking_score_thre = 0.18;
opt.termination_thre = 2;
opt.association_score_thre = 0.6;
opt.exit_thre = 0.3;
opt.initialization_thre = 0.25;
opt.overlap_sup = 0.7;

% scene context
opt.backgrounds = cell(1,7);
opt.backgrounds{1} = [240, 570, 540, 760; 1, 1, 1920, 400]; % [x1, y1, x2, y2]
opt.backgrounds{2} = [1, 1, 155, 460; 1610, 670, 1710, 770; 700, 445, 760, 520; 1270, 210, 1360, 300; 1760, 900, 1830, 1040; 1455, 770, 1515, 890];
opt.backgrounds{7} = [1, 1, 1920, 450];
opt.h_max = cell(1,7);
opt.h_max{2} = 315; 
opt.h_min = cell(1,7);
opt.fps = [30, 30, 14, 30, 30, 30, 25];

% tracking state: START -> ACTIVATED -> TRACKED -> LOST -> STOP
opt.STATE_START = 4;
opt.STATE_ACTIVATED = 2;
opt.STATE_TRACKED = 1;
opt.STATE_LOST = 3;
opt.STATE_STOP = 5;
