% save the tracking results to txt required by the MOTChallenge
function save_results(seq_name, file_name, bboxes, initialization_threshold, opt)

num_target = max(bboxes.id);
len_traj = zeros(num_target, 1);
for i = 1:num_target
    len_traj(i) = numel(find(bboxes.id == i & (bboxes.state == opt.STATE_TRACKED | bboxes.state == opt.STATE_ACTIVATED)));
end

fid = fopen(file_name, 'w');
num_bbox = numel(bboxes.x);

for i = 1:num_bbox
    % <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    if len_traj(bboxes.id(i)) > initialization_threshold && (bboxes.state(i) == opt.STATE_TRACKED || bboxes.state(i) == opt.STATE_ACTIVATED)
        fprintf(fid, '%d,%d,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
            bboxes.fr(i), bboxes.id(i), bboxes.x(i), bboxes.y(i), bboxes.w(i), bboxes.h(i), -1, -1, -1, -1);
    end
end

fclose(fid);