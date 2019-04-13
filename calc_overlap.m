% caculate the overlap ratios and occluded ratios between only one bbox in bboxes1(idx1) and all the bboxes in bboxes2(idx2_list(:))
% idx1 should be a scalar
% idx2_list should be an array or scalar
function [ov, occ1, occ2] = calc_overlap(bboxes1, idx1, bboxes2, idx2_list)

idx2_list = idx2_list(:)';
n = length(idx2_list);

x1_1 = bboxes1.x(idx1);
x2_1 = bboxes1.x(idx1) + bboxes1.w(idx1) - 1;
y1_1 = bboxes1.y(idx1);
y2_1 = bboxes1.y(idx1) + bboxes1.h(idx1) - 1;

x1_2_list = bboxes2.x(idx2_list);
x2_2_list = bboxes2.x(idx2_list) + bboxes2.w(idx2_list) - 1;
y1_2_list = bboxes2.y(idx2_list);
y2_2_list = bboxes2.y(idx2_list) + bboxes2.h(idx2_list) - 1;

area1 = bboxes1.h(idx1) .* bboxes1.w(idx1);
area2_list = bboxes2.h(idx2_list) .* bboxes2.w(idx2_list);

% find the overlapping area
x1_inter = max(x1_1, x1_2_list);
y1_inter = max(y1_1, y1_2_list);
x2_inter = min(x2_1, x2_2_list);
y2_inter = min(y2_1, y2_2_list);
w_inter = x2_inter - x1_inter + 1;
h_inter = y2_inter - y1_inter + 1;

index_overlapped = find((w_inter > 0) .* (h_inter > 0));  
ov = zeros(1, n);
occ1 = zeros(1, n);
occ2 = zeros(1, n);
if ~isempty(index_overlapped)
    area_inter = w_inter(index_overlapped) .* h_inter(index_overlapped); % intersection area
    area_union = area1  +  area2_list(index_overlapped)  -  w_inter(index_overlapped) .* h_inter(index_overlapped); % union area
    ov(index_overlapped) = area_inter ./ area_union; % overlap ratio
    occ1(index_overlapped) = area_inter / area1; % occluded ratio of bbox1
    occ2(index_overlapped) = area_inter ./ area2_list(index_overlapped); % occluded ratio of bbox2
end
