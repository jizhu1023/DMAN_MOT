% find nearby detections with appropriate size for association
function [bboxes_det, index_det, center_pred] = find_detections_for_association(tracker, frame_id, bboxes_det, opt)

center_pred = motion_predict(frame_id, tracker, opt);
num_det = numel(bboxes_det.fr);
center_dets = [bboxes_det.x + bboxes_det.w/2, bboxes_det.y + bboxes_det.h/2];

% compute distances and height ratios
distances = zeros(num_det, 1);
ratios = zeros(num_det, 1);
for i = 1:num_det
    distances(i) = norm(center_dets(i,:) - center_pred) / tracker.bboxes.w(end);

    ratio = tracker.bboxes.h(end) / bboxes_det.h(i);
    ratios(i) = min(ratio, 1/ratio);
end

index_det = find(distances < opt.distance_thre & ratios > opt.aspect_ratio_thre);
bboxes_det.ratios = ratios;
bboxes_det.distances = distances;