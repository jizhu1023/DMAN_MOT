function metrics = track_seq(seq_idx, seq_set, tracker, opt)

is_print = 1;   % set is_print to 1 to display the printed info

opt.is_print = is_print;
opt.exit_thre = 0.3;

if strcmp(seq_set, 'train') == 1
    seq_name = opt.mot2d_train_seqs{seq_idx};
    seq_len = opt.mot2d_train_nums(seq_idx);
else
    seq_name = opt.mot2d_test_seqs{seq_idx};
    seq_len = opt.mot2d_test_nums(seq_idx);
end

% mkdir img_traj
if exist('img_traj') ~= 7
    mkdir('img_traj');
end

if exist(['img_traj/' seq_name]) ~= 7
    mkdir(['img_traj/' seq_name]);
end

% read detections
det_file = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'det', 'det.txt');
bboxes_det = read_bboxes(det_file);

if strcmp(seq_set, 'train') == 1
    % read ground truth
    gt_file = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'gt', 'gt.txt');
    bboxes_gt = read_bboxes(gt_file);
    index_considered = find(bboxes_gt.r == 1);
    bboxes_gt = sub_bboxes(bboxes_gt, index_considered);
end

% initialize frame settings
frame_path = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'img1', sprintf('%06d.jpg', 1));
frame_image = imread(frame_path);
frame_size.x = 1;
frame_size.y = 1;
frame_size.w = size(frame_image, 2);
frame_size.h = size(frame_image, 1);

% intialize tracker
tracker.image_width = frame_size.w;
tracker.image_height = frame_size.h;
tracker.max_width = max(bboxes_det.w);
tracker.max_height = max(bboxes_det.h);
tracker.max_score = max(bboxes_det.r);
tracker.min_score = min(bboxes_det.r);
tracker.num_tracked = 0;
tracker.fps = opt.fps(seq_idx);

% for each frame
trackers = [];
id = 0;
for fr = 1:seq_len
    if is_print
        fprintf('%s/%s frame %d\n', seq_set, seq_name, fr);
    else
        fprintf('.');
        if mod(fr, 100) == 0
            fprintf('\n');
        end        
    end

    % read frame image
    frame_path = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'img1', sprintf('%06d.jpg', fr));
    frame_image = imread(frame_path);
    
    % extract detection
    sub_idx = find(bboxes_det.fr == fr);
    bboxes = sub_bboxes(bboxes_det, sub_idx);

    % Detection preprocessing based on the scene context
    if strcmp(seq_set, 'test')
        sub_idx = [];
        for idx = 1:numel(bboxes.x)
            FP_flag = 0;
            % filter detections which are too large to be pedestrians
            if ~isempty(opt.h_max{seq_idx})
                if bboxes.h(idx) > opt.h_max{seq_idx}
                    FP_flag = 1;
                end
            end

            % filter detections which are too small to be pedestrians
            if ~isempty(opt.h_min{seq_idx})
                if bboxes.h(idx) < opt.h_min{seq_idx}
                    FP_flag = 1;
                end
            end

            % filter detections in impossible locations
            if ~isempty(opt.backgrounds{seq_idx})
                cx_det = bboxes.x(idx) + bboxes.w(idx) / 2; % center x of the detection
                cy_det = bboxes.y(idx) + bboxes.h(idx) / 2; % center y of the detection
                for bg_idx = 1:size(opt.backgrounds{seq_idx}, 1)
                    bg = opt.backgrounds{seq_idx}(bg_idx,:);
                    if cx_det > bg(1) && cx_det < bg(3) && cy_det > bg(2) && cy_det < bg(4)
                        FP_flag = 1;
                        continue;
                    end
                end
            end
            if FP_flag == 0
                sub_idx = [sub_idx, idx];
            end
        end
        bboxes = sub_bboxes(bboxes, sub_idx);
    end
    
    % nms
     boxes = [bboxes.x bboxes.y bboxes.x+bboxes.w bboxes.y+bboxes.h bboxes.r];
     sub_idx = nms(boxes, 0.6);
     bboxes = sub_bboxes(bboxes, sub_idx);
    
    % sort trackers
    [index1, index2] = sort_trackers(trackers, opt);
    index_processed = [];
    for k = 1:2
        % process trackers in the first class or the second class
        if k == 1
            index_track = index1;
        else
            index_track = index2;
        end
        
        % process trackers
        for i = 1:numel(index_track)
            ind = index_track(i);
            if trackers{ind}.state == opt.STATE_TRACKED || trackers{ind}.state == opt.STATE_ACTIVATED
                % track target
                [bboxes_tmp, index] = find_candidate_detections(trackers(index_processed), bboxes, opt);
                bboxes_eco_track = sub_bboxes(bboxes_tmp, index);
                trackers{ind} = MOT_track(fr, frame_image, frame_size, bboxes_eco_track, trackers{ind}, opt, seq_name);
                
                if trackers{ind}.state == opt.STATE_TRACKED || trackers{ind}.state == opt.STATE_ACTIVATED
                    index_processed = [index_processed; ind];
                end
            end
        end
        for i = 1:numel(index_track)
            ind = index_track(i);
            if trackers{ind}.state == opt.STATE_LOST
                % associate target
                [bboxes_tmp, index] = find_candidate_detections(trackers(index_processed), bboxes, opt);
                bboxes_associate = sub_bboxes(bboxes_tmp, index);    
                trackers{ind} = MOT_associate(fr, frame_image, frame_size, bboxes_associate, trackers{ind}, opt, seq_name);              
                index_processed = [index_processed; ind];              
            end
        end
    end
    
    % find detections for initialization
    [bboxes, index] = find_candidate_detections(trackers, bboxes, opt);
    
    for i = 1:numel(index)
        % extract features
        bbox = sub_bboxes(bboxes, index(i));
          
        % filter detections using SVM
        det_feat = extract_det_feat(tracker, bbox);
        label = svmpredict(1, det_feat, tracker.w_active, '-q');
        if label < 0
           continue;
        end
        
        % reset tracker for the new object identity
        tracker.state = opt.STATE_START;            
        id = id + 1;
        trackers{end+1} = initialize_tracker(fr, frame_image, id, bboxes, index(i), tracker, opt);  
    end

    % resolve tracker conflict
    trackers = handle_conflicting_trackers(trackers, bboxes, opt);   

    bboxes_track = [];
    for i = 1:numel(trackers)
        if isempty(bboxes_track)
            bboxes_track = trackers{i}.bboxes;
        else
            bboxes_track = concatenate_bboxes(bboxes_track, trackers{i}.bboxes);
        end
    end

    for i = 1:numel(trackers)
        if trackers{i}.state == opt.STATE_START || trackers{i}.state == opt.STATE_TRACKED || trackers{i}.state == opt.STATE_ACTIVATED
            bbox = sub_bboxes(trackers{i}.bboxes, numel(trackers{i}.bboxes.fr));
            x1 = floor(max(1, bbox.x));
            y1 = floor(max(1, bbox.y));
            x2 = ceil(min(frame_size.w, bbox.x+bbox.w-1));
            y2 = ceil(min(frame_size.h, bbox.y+bbox.h-1));
            img_traj = frame_image(y1:y2, x1:x2, :);           
            if exist(['img_traj/' seq_name '/' num2str(trackers{i}.target_id)]) ~= 7
                mkdir(['img_traj/' seq_name '/' num2str(trackers{i}.target_id)]);
            end
            imwrite(img_traj, ['img_traj/' seq_name '/' num2str(trackers{i}.target_id) '/' num2str(fr) '.jpg']);
        end
    end
end

% write tracking results
file_name = sprintf('%s/%s.txt', opt.results_dir, seq_name);
fprintf('write results: %s\n', file_name);
save_results(seq_name, file_name, bboxes_track, tracker.fps*opt.initialization_thre, opt);
