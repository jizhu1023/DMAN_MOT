% read bboxes from txt file given by the MOTChallege
function bboxes = read_bboxes(file_name)

fid = fopen(file_name, 'r');
% <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
line = textscan(fid, '%d %d %f %f %f %f %f %f %f %f', 'Delimiter', ',');
fclose(fid);

% build the bboxes structure for detections
bboxes.fr = line{1};
bboxes.id = line{2};
bboxes.x = line{3};
bboxes.y = line{4};
bboxes.w = line{5};
bboxes.h = line{6};
bboxes.r = line{7};