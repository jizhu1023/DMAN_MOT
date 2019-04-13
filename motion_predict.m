% predict the target location based on locations in the past K tracked frames
function prediction = motion_predict(frame_current, tracker, opt)

bboxes = tracker.bboxes;
index = find(bboxes.state == opt.STATE_TRACKED | bboxes.state == opt.STATE_ACTIVATED);
bboxes = sub_bboxes(bboxes, index);
center_x = bboxes.x + bboxes.w/2;
center_y = bboxes.y + bboxes.h/2;
frames_past = double(bboxes.fr);
frame_current = double(frame_current);

num = numel(frames_past);
K = 10;
%K = 0.3 * tracker.fps;
if num > K
    center_x = center_x(num-K+1:num);
    center_y = center_y(num-K+1:num);
    frames_past = frames_past(num-K+1:num);
end

% compute velocity
vx = 0;
vy = 0;
num = numel(center_x);
count = 0;
for j = 2:num
    vx = vx + (center_x(j)-center_x(j-1)) / (frames_past(j) - frames_past(j-1));
    vy = vy + (center_y(j)-center_y(j-1)) / (frames_past(j) - frames_past(j-1));
    count = count + 1;
end
if count
    vx = vx / count;
    vy = vy / count;
end

if isempty(center_x)
    bboxes = tracker.bboxes;
    center_x_pred = bboxes.x(end) + bboxes.w(end)/2;
    center_y_pred = bboxes.y(end) + bboxes.h(end)/2;
else
    center_x_pred = center_x(end) + vx * (frame_current - frames_past(end));
    center_y_pred = center_y(end) + vy * (frame_current - frames_past(end));
end
prediction = [center_x_pred center_y_pred];