% track a target
function tracker = MOT_track(frame_idx, frame_image, frame_size, bboxes, tracker, opt, seq_name)
  
if tracker.state == opt.STATE_TRACKED || tracker.state == opt.STATE_ACTIVATED
    tracker.num_occluded = 0;
    tracker.num_tracked = tracker.num_tracked + 1;
    tracker = track_frame(tracker, frame_idx, frame_image, bboxes, [], seq_name, opt);

    % check if target exits from the camera view
    [~, ov] = calc_overlap(tracker.bboxes, numel(tracker.bboxes.fr), frame_size, 1);
    if ov < opt.exit_thre
        if opt.is_print
            fprintf('The target exits from the camera view.\n');
        end
        tracker.state = opt.STATE_STOP;
    end    
end