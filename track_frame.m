
function [tracker, qscore, f] = track_frame(tracker, frame_id, frame_image, bboxes_det, index_det, seq_name, opt)

    global client_tcp;
    
    % tracked, decide to tracked or occluded
    if tracker.state == opt.STATE_TRACKED || tracker.state == opt.STATE_ACTIVATED
        tracker = ECO_tracking(frame_id, frame_image, bboxes_det, tracker, opt);
        is_tracking_score_good = tracker.eco.is_confident;
        is_overlap_ratio_good = mean(tracker.eco.bb_overlaps);
        tracker.bb = tracker.eco.bb';
        qscore = 0;
        if is_tracking_score_good == 1
            if seq_name == 'MOT16-03'
                overlap_threshold = 0.1;
            else
                overlap_threshold = 0.8;
            end  
            if tracker.state == opt.STATE_TRACKED && is_overlap_ratio_good > overlap_threshold
                label = 1;
            elseif tracker.state == opt.STATE_ACTIVATED %&& is_overlap_ratio_good > 0.8
                label = 1;
            else
                label = -1;
            end            
        else
            label = -1;
        end
    
        % make a decision
        if label > 0
            if tracker.num_tracked > (0.25 * tracker.fps)
                tracker.state = opt.STATE_TRACKED;
            else
                tracker.state = opt.STATE_ACTIVATED;
            end
        else
            if tracker.state == opt.STATE_ACTIVATED
                tracker.state = opt.STATE_STOP;
                fprintf('Target %d end.\n', tracker.target_id);
            else 
                tracker.state = opt.STATE_LOST;
            end     
        end
    
        % build the bboxes structure new
        if tracker.state == opt.STATE_TRACKED || tracker.state == opt.STATE_ACTIVATED        
            bbox.x = tracker.bb(1);
            bbox.y = tracker.bb(2);
            bbox.w = tracker.bb(3) - tracker.bb(1);
            bbox.h = tracker.bb(4) - tracker.bb(2);
            bbox.r = 1;
            bbox.eco_x = tracker.eco.bb(1);
            bbox.eco_y = tracker.eco.bb(2);
            bbox.eco_w = tracker.eco.bb(3) - tracker.eco.bb(1);
            bbox.eco_h = tracker.eco.bb(4) - tracker.eco.bb(2);
    
            if ~isempty(bboxes_det.fr)
                o = calc_overlap(bbox, 1, bboxes_det, 1:numel(bboxes_det.fr));
                tracker.eco.bb_overlaps(1:end-1) = tracker.eco.bb_overlaps(2:end);
                if max(o) > 0.5
                    tracker.eco.bb_overlaps(end) = 1;
                else
                    tracker.eco.bb_overlaps(end) = 0;
                end
            else
                tracker.eco.bb_overlaps(1:end-1) = tracker.eco.bb_overlaps(2:end);
                tracker.eco.bb_overlaps(end) = 0;
            end
        else
            bbox = sub_bboxes(tracker.bboxes, numel(tracker.bboxes.fr));
        end
        if tracker.state == opt.STATE_ACTIVATED && tracker.eco.bb_overlaps(end) == 0
            tracker.state = opt.STATE_STOP;
        end
        bbox.fr = frame_id;
        bbox.id = tracker.target_id;
        bbox.state = tracker.state;
        tracker.bboxes = concatenate_bboxes(tracker.bboxes, bbox);
    
    % occluded, decide to tracked or occluded
    elseif tracker.state == opt.STATE_LOST
        % association
        if isempty(index_det)
            qscore = 0;
            label = -1;
            f = [];
        else
            bboxes = sub_bboxes(bboxes_det, index_det);
            ctrack = motion_predict(frame_id, tracker, opt);
            bboxes_predict.w = tracker.bboxes.w(end);
            bboxes_predict.h = tracker.bboxes.h(end);
            bboxes_predict.x = ctrack(1) - (bboxes_predict.w - 1) / 2;
            bboxes_predict.y = ctrack(2) - (bboxes_predict.h - 1) / 2;
            o_predict = calc_overlap(bboxes_predict, 1, bboxes, 1:numel(bboxes.fr));
            [motion_score, motion_ind] = max(o_predict);
            bboxes_tracker = tracker.bboxes;
            index_state = find(bboxes_tracker.state == opt.STATE_TRACKED | bboxes_tracker.state == opt.STATE_ACTIVATED);
            bboxes_tracker = sub_bboxes(bboxes_tracker, index_state);
            if ~isempty(bboxes_tracker.fr)
                fr_tracker = bboxes_tracker.fr(end);
            else % for target which is suppressed since the first frame it appears
                fr_tracker = 1;
            end
            traj_dir = ['img_traj/' seq_name '/' num2str(tracker.target_id) '/'];
            frame_id_double = double(frame_id);
            save('mot_py.mat', 'traj_dir', 'bboxes', 'frame_id_double', 'seq_name');
            
            fwrite(client_tcp, 'client ok');
            fread(client_tcp, 9); % size is 9 for 'server ok' message
            load('similarity.mat');
            
            [ass_score, ind] = max(similarity);
            index_tracked = find(tracker.bboxes.state == opt.STATE_TRACKED | tracker.bboxes.state == opt.STATE_ACTIVATED);
            bboxes_tracked = sub_bboxes(tracker.bboxes, index_tracked);
            fr_tracked = bboxes_tracked.fr(end);
            if frame_id - fr_tracker == 1
                prev_c = [bboxes_tracked.x(end) + bboxes_tracked.w(end) / 2, bboxes_tracked.y(end) + bboxes_tracked.h(end) / 2];
                prev_w = bboxes_tracked.w(end);
                prev_h = bboxes_tracked.h(end);
                bboxes_det_one = sub_bboxes(bboxes_det, index_det(ind));
                cur_c = [bboxes_det_one.x + bboxes_det_one.w/2, bboxes_det_one.y + bboxes_det_one.h/2];
                dis = norm(cur_c - prev_c) / prev_w;
                ratio = bboxes_det_one.h / prev_h;
                ratio = min(ratio, 1/ratio);
            end
    
            if ass_score > opt.association_score_thre
                label = 1;
                bboxes_det_one = sub_bboxes(bboxes_det, index_det(ind));
                fprintf('Target %d associated by appearance.\n', tracker.target_id);
           
            elseif frame_id - fr_tracker < 5 && ass_score > 0.4 && motion_score > 0.5
                label = 1;
                bboxes_det_one = sub_bboxes(bboxes, motion_ind);
                fprintf('Target %d associated by motion.\n', tracker.target_id);
            else
                label = -1;
            end 
    
            if label == 1
                tracker.eco.rect_position = double(round([bboxes_det_one.x, bboxes_det_one.y, bboxes_det_one.w, bboxes_det_one.h]));
                tracker.eco.pos = single([bboxes_det_one.y + (bboxes_det_one.h - 1)/2, bboxes_det_one.x + (bboxes_det_one.w - 1)/2]); 
                tracker.eco.target_sz = double([bboxes_det_one.h, bboxes_det_one.w]);
                tracker.eco.base_target_sz = double(tracker.eco.target_sz / tracker.eco.currentScaleFactor);
                tracker.eco.base_target_sz(2) = tracker.eco.base_target_sz(2)* 0.5;
                tracker = ECO_tracking(frame_id, frame_image, bboxes_det_one, tracker, opt);
            end
        end
    
        if label > 0
            tracker.bb = tracker.eco.bb';
    
            % association
            if tracker.num_tracked > 0.25 * tracker.fps
                tracker.state = opt.STATE_TRACKED;
            else
                tracker.state = opt.STATE_ACTIVATED;
            end
            
            % build the bboxes structure
            bbox = [];
            bbox.fr = frame_id;
            bbox.id = tracker.target_id;
            bbox.x = tracker.bb(1);
            bbox.y = tracker.bb(2);
            bbox.w = tracker.bb(3) - tracker.bb(1);
            bbox.h = tracker.bb(4) - tracker.bb(2);
            bbox.r = 1;
            bbox.state = tracker.state;
            bbox.eco_x = tracker.eco.bb(1);
            bbox.eco_y = tracker.eco.bb(2);
            bbox.eco_w = tracker.eco.bb(3) - tracker.eco.bb(1);
            bbox.eco_h = tracker.eco.bb(4) - tracker.eco.bb(2);
              
     
            if tracker.bboxes.fr(end) == frame_id
                bboxes = tracker.bboxes;
                index = 1:numel(bboxes.fr)-1;
                tracker.bboxes = sub_bboxes(bboxes, index);            
            end
    
            tracker.bboxes = interpolate_traj(tracker.bboxes, bbox, tracker.fps, opt);
    
            if isempty(bboxes_det.fr) == 0
                o = calc_overlap(bbox, 1, bboxes_det, 1:numel(bboxes_det.fr));
                tracker.eco.bb_overlaps(1:end-1) = tracker.eco.bb_overlaps(2:end);
                if max(o) > 0.5
                    tracker.eco.bb_overlaps(end) = 1;
                else
                    tracker.eco.bb_overlaps(end) = 0;
                end
            else
                tracker.eco.bb_overlaps(1:end-1) = tracker.eco.bb_overlaps(2:end);
                tracker.eco.bb_overlaps(end) = 0;
            end 
        else
            % no association
            tracker.state = opt.STATE_LOST;
            bbox = sub_bboxes(tracker.bboxes, numel(tracker.bboxes.fr));
            bbox.fr = frame_id;
            bbox.id = tracker.target_id;
            bbox.state = tracker.state;
            
            if tracker.bboxes.fr(end) == frame_id
                bboxes = tracker.bboxes;
                index = 1:numel(bboxes.fr)-1;
                tracker.bboxes = sub_bboxes(bboxes, index);
            end        
            tracker.bboxes = concatenate_bboxes(tracker.bboxes, bbox);          
        end
    end