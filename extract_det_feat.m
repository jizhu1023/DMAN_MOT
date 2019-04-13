% extract features for the detection bbox
function feat = extract_det_feat(tracker, bbox)

num = numel(bbox.fr);
feat_dim = 6;
feat = zeros(num, feat_dim);
feat(:,1) = bbox.x / tracker.image_width;
feat(:,2) = bbox.y / tracker.image_height;
feat(:,3) = bbox.w / tracker.max_width;
feat(:,4) = bbox.h / tracker.max_height;
feat(:,5) = bbox.r / tracker.max_score;
feat(:,6) = 1;