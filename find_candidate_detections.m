% find candidate detections which have not been covered by any existing trackers
function [bboxes_det, index_det] = find_candidate_detections(trackers, bboxes_det, opt)

if isempty(bboxes_det)
    index_det = [];
    return;
end

% collect bboxes from trackers
bboxes_tracker = [];
for i = 1:numel(trackers)
    tracker = trackers{i};
    bboxes = sub_bboxes(tracker.bboxes, numel(tracker.bboxes.fr));
    
    if tracker.state == opt.STATE_TRACKED || tracker.state == opt.STATE_ACTIVATED
        if isempty(bboxes_tracker)
            bboxes_tracker = bboxes;
        else
            bboxes_tracker = concatenate_bboxes(bboxes_tracker, bboxes);
        end
    end
end

% compute overlap ratios and occluded ratios between bboxes_det and bboxes_tracker
num_det = numel(bboxes_det.fr);
if isempty(bboxes_tracker)
    num_track = 0;
else
    num_track = numel(bboxes_tracker.fr);
end

if num_track
    ov = zeros(num_det, 1); % overlap ratios
    occ = zeros(num_det, 1); % occluded ratios
    for i = 1:num_det
        [o1, o2] = calc_overlap(bboxes_det, i, bboxes_tracker, 1:num_track);
        ov(i) = max(o1);
        occ(i) = sum(o2);
    end
    index_det = find(ov < 0.5 & occ < 0.5);
else
    index_det = 1:num_det;
end