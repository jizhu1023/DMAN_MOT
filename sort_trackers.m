% heuristically sort the reliability of trackers based on the tracking state and the number of tracked frames
function [index1, index2] = sort_trackers(trackers, opt)

num = numel(trackers);
len = zeros(num, 1);
state = zeros(num, 1);
for i = 1:num
    len(i) = trackers{i}.num_tracked;
    state(i) = trackers{i}.state;
end

% give high priority to trackers with > 10 tracked frames
index1 = find(len > 10);
index2 = find(len <= 10);

% priority order: opt.STATE_TRACKED -> opt.STATE_ACTIVATED -> opt.STATE_LOST -> opt.STATE_START -> opt.STATE_STOP
[~, ind] = sort(state(index1));
index1 = index1(ind);
[~, ind] = sort(state(index2));
index2 = index2(ind);
index = [index1; index2];