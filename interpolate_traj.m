% add traj2 to traj1 and interpolate
function traj_interpolate = interpolate_traj(traj1, traj2, fps, opt)

if isempty(traj2) == 1
    traj_interpolate = traj1;
    return;
end

index = find(traj1.state == opt.STATE_TRACKED | traj1.state == opt.STATE_ACTIVATED);
if isempty(index) == 0
    ind = index(end);
    fr1 = double(traj1.fr(ind));
    fr2 = double(traj2.fr(1));

    if fr2 - fr1 <= fps && fr2 - fr1 > 1
        traj1 = sub_bboxes(traj1, 1:ind);

        % box1
        x1 = traj1.x(end);
        y1 = traj1.y(end);
        w1 = traj1.w(end);
        h1 = traj1.h(end);
        r1 = traj1.r(end);

        % box2
        x2 = traj2.x(1);
        y2 = traj2.y(1);
        w2 = traj2.w(1);
        h2 = traj2.h(1);
        r2 = traj2.r(1);

        % linear interpolation
        n = fieldnames(traj1);
        for fr = fr1+1:fr2-1
            bbox = sub_bboxes(traj2, 1);
            bbox.fr = fr;
            bbox.x = x1 + ((x2 - x1) / (fr2 - fr1)) * (fr - fr1);
            bbox.y = y1 + ((y2 - y1) / (fr2 - fr1)) * (fr - fr1);
            bbox.w = w1 + ((w2 - w1) / (fr2 - fr1)) * (fr - fr1);
            bbox.h = h1 + ((h2 - h1) / (fr2 - fr1)) * (fr - fr1);
            bbox.r = r1 + ((r2 - r1) / (fr2 - fr1)) * (fr - fr1);

            bbox.eco_x = x1 + ((x2 - x1) / (fr2 - fr1)) * (fr - fr1);
            bbox.eco_y = y1 + ((y2 - y1) / (fr2 - fr1)) * (fr - fr1);
            bbox.eco_w = w1 + ((w2 - w1) / (fr2 - fr1)) * (fr - fr1);
            bbox.eco_h = h1 + ((h2 - h1) / (fr2 - fr1)) * (fr - fr1);

            for i = 1:length(n),
                f = n{i};
                traj1.(f) = [traj1.(f); bbox.(f)];
            end    
        end
    end
end

% concatenation
n = fieldnames(traj1);
for i = 1:length(n),
    f = n{i};
    traj_interpolate.(f) = [traj1.(f); traj2.(f)];
end