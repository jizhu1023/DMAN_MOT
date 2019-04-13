% Non-maximum suppression
function det = nms(boxes, overlap_thre)

if isempty(boxes)
    det = [];
else
    x1 = boxes(:,1);
    y1 = boxes(:,2);
    x2 = boxes(:,3);
    y2 = boxes(:,4);
    score = boxes(:,end);
    area = (x2-x1+1) .* (y2-y1+1);

    [~, idx_list] = sort(score, 'descend');
    num_bbox = numel(idx_list);
    det = ones(num_bbox, 1);
    for i = 2:num_bbox
        ii = idx_list(i);
        for j = 1:i-1
            jj = idx_list(j);
            if det(jj)
                x1_inter = max(x1(ii), x1(jj));
                y1_inter = max(y1(ii), y1(jj));
                x2_inter = min(x2(ii), x2(jj));
                y2_inter = min(y2(ii), y2(jj));
                w_inter = x2_inter-x1_inter+1;
                h_inter = y2_inter-y1_inter+1;
                if w_inter > 0 && h_inter > 0
                    % compute overlap and occluded ratio 
                    area_inter = w_inter * h_inter;
                    area_union = area(ii) + area(jj) - area_inter;
                    overlap_ratio =  area_inter / area_union;
                    occluded_ratio1 = area_inter / area(ii);
                    occluded_ratio2 = area_inter / area(jj);
                    if overlap_ratio > overlap_thre || occluded_ratio1 > 0.9 || occluded_ratio2 > 0.9
                        det(ii) = 0;
                        break;
                    end
                end
            end
        end
    end
    det = find(det == 1);
end