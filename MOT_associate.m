% associate a lost target
function tracker = MOT_associate(fr, frame_image, frame_size, bboxes_associate, tracker, opt, seq_name)

if tracker.state == opt.STATE_LOST
    tracker.num_occluded = tracker.num_occluded + 1;

    % find a set of detections for association
    [bboxes_associate, index_det] = find_detections_for_association(tracker, fr, bboxes_associate, opt);
    tracker = track_frame(tracker, fr, frame_image, bboxes_associate, index_det, seq_name, opt);
    if tracker.state == opt.STATE_TRACKED || tracker.state == opt.STATE_ACTIVATED
        tracker.num_occluded = 0;
    end

    % terminate tracking if the target is lost for a long time
    if tracker.num_occluded > opt.termination_thre * tracker.fps
        tracker.state = opt.STATE_STOP;
        if opt.is_print
            fprintf('target %d exits due to long time occlusion\n', tracker.target_id);
        end
    end
   
    % check if target outside image
    [~, overlap] = calc_overlap(tracker.bboxes, numel(tracker.bboxes.fr), frame_size, 1);
    if overlap < opt.exit_thre
        if opt.is_print
            fprintf('target outside image by checking boarders\n');
        end
        tracker.state = opt.STATE_STOP;
    end    
end