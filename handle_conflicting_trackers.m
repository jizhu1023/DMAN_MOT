% resolve conflict between trackers
function trackers = handle_conflicting_trackers(trackers, bboxes_det, opt)

% collect bboxes from trackers
bboxes_track = [];
for i = 1:numel(trackers)
    tracker = trackers{i};
    bboxes = sub_bboxes(tracker.bboxes, numel(tracker.bboxes.fr));
    
    if tracker.state == opt.STATE_TRACKED || tracker.state == opt.STATE_ACTIVATED
        if isempty(bboxes_track)
            bboxes_track = bboxes;
        else
            bboxes_track = concatenate_bboxes(bboxes_track, bboxes);
        end
    end

    if tracker.state == opt.STATE_STOP
        tracker.eco = [];
    end
end   

% compute overlaps
num_det = numel(bboxes_det.fr);
if isempty(bboxes_track)
    num_track = 0;
else
    num_track = numel(bboxes_track.fr);
end

flag = zeros(num_track, 1);
for i = 1:num_track
    if flag(i) == 1
        continue;
    end
    [~, o] = calc_overlap(bboxes_track, i, bboxes_track, 1:num_track);
    o(i) = 0;
    o(flag == 1) = 0;
    [max_ov, j] = max(o);
    if max_ov > opt.overlap_sup       
        num1 = trackers{bboxes_track.id(i)}.num_tracked;
        num2 = trackers{bboxes_track.id(j)}.num_tracked;
        o1 = max(calc_overlap(bboxes_track, i, bboxes_det, 1:num_det));
        o2 = max(calc_overlap(bboxes_track, j, bboxes_det, 1:num_det));
        
        if num1 > num2
            suppressed_idx = j;
            winner_idx = i;
        elseif num1 < num2
            suppressed_idx = i;
            winner_idx = j;
        else
            if o1 > o2
                suppressed_idx = j;
                winner_idx = i;
            else
                suppressed_idx = i;
                winner_idx = j;
            end
        end
        
        if numel(trackers{bboxes_track.id(suppressed_idx)}.bboxes.fr) == 1
            trackers{bboxes_track.id(suppressed_idx)}.state = opt.STATE_STOP;
            trackers{bboxes_track.id(suppressed_idx)}.bboxes.state(end) = opt.STATE_STOP;
        else
            trackers{bboxes_track.id(suppressed_idx)}.state = opt.STATE_LOST;
            trackers{bboxes_track.id(suppressed_idx)}.bboxes.state(end) = opt.STATE_LOST;
        end
        if opt.is_print
            fprintf('target %d suppressed by %d\n', bboxes_track.id(suppressed_idx), bboxes_track.id(winner_idx));
        end
        flag(suppressed_idx) = 1;
    end
end