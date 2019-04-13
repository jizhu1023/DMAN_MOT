% apply the ECO tracking for each target (modified based on the original ECO code)
function tracker = ECO_tracking(frame_id, im, bboxes_det, tracker, opt)

tracker.eco.n_frame = tracker.eco.n_frame + 1;
frame = tracker.eco.n_frame;

% variables from ECO_initialize.m
params = tracker.eco.params;
max_train_samples = tracker.eco.max_train_samples;
features = tracker.eco.features;
global_fparams = tracker.eco.global_fparams;
pos = tracker.eco.pos;
target_sz = tracker.eco.target_sz;
currentScaleFactor = tracker.eco.currentScaleFactor;
base_target_sz = tracker.eco.base_target_sz;
img_support_sz = tracker.eco.img_support_sz;
feature_dim = tracker.eco.feature_dim;
num_feature_blocks = tracker.eco.num_feature_blocks;
feature_reg = tracker.eco.feature_reg;
feature_extract_info = tracker.eco.feature_extract_info;
compressed_dim = tracker.eco.compressed_dim;
compressed_dim_cell = tracker.eco.compressed_dim_cell;
filter_sz = tracker.eco.filter_sz;
filter_sz_cell = tracker.eco.filter_sz_cell;
output_sz = tracker.eco.output_sz;
pad_sz = tracker.eco.pad_sz;
kx = tracker.eco.kx;
ky = tracker.eco.ky;
yf = tracker.eco.yf;
cos_window = tracker.eco.cos_window;
interp1_fs = tracker.eco.interp1_fs;
interp2_fs = tracker.eco.interp2_fs;
reg_filter = tracker.eco.reg_filter;
reg_energy = tracker.eco.reg_energy;
nScales = tracker.eco.nScales;
scaleFactors = tracker.eco.scaleFactors;
scale_filter = tracker.eco.scale_filter;
min_scale_factor = tracker.eco.min_scale_factor;
max_scale_factor = tracker.eco.max_scale_factor;
init_CG_opts = tracker.eco.init_CG_opts;
CG_opts = tracker.eco.CG_opts;
rect_position = tracker.eco.rect_position;
prior_weights = tracker.eco.prior_weights;
sample_weights = tracker.eco.sample_weights;
samplesf = tracker.eco.samplesf;
score_matrix = tracker.eco.score_matrix;
latest_ind = tracker.eco.latest_ind;
frames_since_last_train = tracker.eco.frames_since_last_train;
num_training_samples = tracker.eco.num_training_samples;
minimum_sample_weight = tracker.eco.minimum_sample_weight;
res_norms = tracker.eco.res_norms;
is_color_image = tracker.eco.is_color_image;

% variables which are useful when frame == 1
sample_pos = tracker.eco.sample_pos;
sample_scale = tracker.eco.sample_scale;
xl = tracker.eco.xl;
xlf = tracker.eco.xlf;
projection_matrix = tracker.eco.projection_matrix;
shift_samp = tracker.eco.shift_samp;
xlf_proj = tracker.eco.xlf_proj;
hf = tracker.eco.hf;
lf_ind = tracker.eco.lf_ind;
proj_energy = tracker.eco.proj_energy;
sample_energy = tracker.eco.sample_energy;
rhs_samplef = tracker.eco.rhs_samplef;
diag_M = tracker.eco.diag_M;
p = tracker.eco.p;
rho = tracker.eco.rho;
r_old = tracker.eco.r_old;
init_samplef = tracker.eco.init_samplef;
init_samplef_H = tracker.eco.init_samplef_H;
projection_matrix_init = tracker.eco.projection_matrix_init;
init_samplef_proj = tracker.eco.init_samplef_proj;
init_hf = tracker.eco.init_hf;
fyf = tracker.eco.fyf;
res_norms_temp = tracker.eco.res_norms_temp;

% other variables
hf_full = tracker.eco.hf_full;

% narrow the bbox to avoid tracking drift
target_sz(2) = target_sz(2) * 0.5;

% load image
if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Target localization step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Do not estimate translation and scaling on the first frame, since we 
% just want to initialize the tracker there

if frame > 1
    old_pos = inf(size(pos));
    iter = 1;
    
    %translation search
    while iter <= params.refinement_iterations && any(old_pos ~= pos)
        % Extract features at multiple resolutions
        sample_pos = round(pos);
        det_sample_pos = sample_pos;
        sample_scale = currentScaleFactor*scaleFactors;
        xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
                    
        % Project sample
        xt_proj = project_sample(xt, projection_matrix);
        
        % Do windowing of features
        xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
        
        % Compute the fourier series
        xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
        
        % Compute convolution for each feature block in the Fourier domain
        scores_fs_feat = cellfun(@(hf, xf, pad_sz) padarray(sum(bsxfun(@times, hf, xf), 3), pad_sz), hf_full, xtf_proj, pad_sz, 'uniformoutput', false);
        
        % Also sum over all feature blocks.
        % Gives the fourier coefficients of the convolution response.
        scores_fs = permute(sum(cell2mat(scores_fs_feat), 3), [1 2 4 3]);
        
        % Optimize the continuous score function with Newton's method.
        [trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations);
        
        % Compute the translation vector in pixel-coordinates and round
        % to the closest integer pixel.
        translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);
        scale_change_factor = scaleFactors(scale_ind);
        
        % update position
        old_pos = pos;
        pos = sample_pos + translation_vec;
        
        if params.clamp_position
            pos = max([1 1], min([size(im,1) size(im,2)], pos));
        end
        
        % Do scale tracking with the scale filter
        if nScales > 0 && params.use_scale_filter
            scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
        end 
        
        % Update the scale
        currentScaleFactor = currentScaleFactor * scale_change_factor;
        
        % Adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        iter = iter + 1;
    end
end

pre_pos = tracker.eco.pos;
pre_target_sz = tracker.eco.target_sz;
pos_diff = double(sqrt((pos(1)-pre_pos(1)).^2 + (pos(2)-pre_pos(2)).^2));
if pos_diff > 1.2 * double(pre_target_sz(2))
    pos = pre_pos;
    is_drift = 1;
else
    is_drift = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model update step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Extract sample and init projection matrix
if frame == 1
    % Extract image region for training sample
    sample_pos = round(pos);
    sample_scale = currentScaleFactor;
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    
    % Do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    
    % Compute the fourier series
    xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
    
    % Interpolate features to the continuous domain
    xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
    
    % New sample to be added
    xlf = compact_fourier_coeff(xlf);
    
    % Initialize projection matrix
    xl1 = cellfun(@(x) reshape(x, [], size(x,3)), xl, 'uniformoutput', false);
    xl1 = cellfun(@(x) bsxfun(@minus, x, mean(x, 1)), xl1, 'uniformoutput', false);
    
    if strcmpi(params.proj_init_method, 'pca')
        [projection_matrix, ~, ~] = cellfun(@(x) svd(x' * x), xl1, 'uniformoutput', false);
        projection_matrix = cellfun(@(P, dim) single(P(:,1:dim)), projection_matrix, compressed_dim_cell, 'uniformoutput', false);
    elseif strcmpi(params.proj_init_method, 'rand_uni')
        projection_matrix = cellfun(@(x, dim) single(randn(size(x,2), dim)), xl1, compressed_dim_cell, 'uniformoutput', false);
        projection_matrix = cellfun(@(P) bsxfun(@rdivide, P, sqrt(sum(P.^2,1))), projection_matrix, 'uniformoutput', false);
    elseif strcmpi(params.proj_init_method, 'none')
        projection_matrix = [];
    else
        error('Unknown initialization method for the projection matrix: %s', params.proj_init_method);
    end
    clear xl1 xlw
    
    % Shift sample
    shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
    xlf = shift_sample(xlf, shift_samp, kx, ky);
    
    % Project sample
    xlf_proj = project_sample(xlf, projection_matrix);
elseif params.learning_rate > 0
    if ~params.use_detection_sample
        % Extract image region for training sample
        sample_pos = round(pos);
        sample_scale = currentScaleFactor;
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        
        % Project sample
        xl_proj = project_sample(xl, projection_matrix);
        
        % Do windowing of features
        xl_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
        
        % Compute the fourier series
        xlf1_proj = cellfun(@cfft2, xl_proj, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xlf1_proj = interpolate_dft(xlf1_proj, interp1_fs, interp2_fs);
        
        % New sample to be added
        xlf_proj = compact_fourier_coeff(xlf1_proj);
    else        
        % Use the sample that was used for detection
        sample_scale = sample_scale(scale_ind);
        xlf_proj = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,scale_ind), xtf_proj, 'uniformoutput', false);
    end
    
    % Shift the sample so that the target is centered
    shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
    xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);
end

xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
    
if params.use_sample_merge
    % Find the distances with existing samples
    dist_vector = find_cluster_distances(samplesf, xlf_proj_perm, num_feature_blocks, num_training_samples, max_train_samples, params);
    
    [merged_sample, new_cluster, merged_cluster_id, new_cluster_id, score_matrix, prior_weights,num_training_samples] = ...
        merge_clusters(samplesf, xlf_proj_perm, dist_vector, score_matrix, prior_weights,...
                       num_training_samples,num_feature_blocks,max_train_samples,minimum_sample_weight,params);
else
    % Do the traditional adding of a training sample and weight update
    % of C-COT
    [prior_weights, replace_ind] = update_prior_weights(prior_weights, sample_weights, latest_ind, frame, params);
    latest_ind = replace_ind;
    
    merged_cluster_id = 0;
    new_cluster = xlf_proj_perm;
    new_cluster_id = replace_ind;
end

if frame > 1 && params.learning_rate > 0 || frame == 1 && ~params.update_projection_matrix
    % Insert the new training sample
    for k = 1:num_feature_blocks
        if merged_cluster_id > 0
            samplesf{k}(merged_cluster_id,:,:,:) = merged_sample{k};
        end
        
        if new_cluster_id > 0
            samplesf{k}(new_cluster_id,:,:,:) = new_cluster{k};
        end
    end
end

sample_weights = prior_weights;
       
train_tracker = (frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);

if train_tracker && is_drift == 0    
    % Used for preconditioning
    new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
    
    if frame == 1
        if params.update_projection_matrix
            hf = cell(2,1,num_feature_blocks);
            lf_ind = cellfun(@(sz) sz(1) * (sz(2)-1)/2 + 1, filter_sz_cell, 'uniformoutput', false);
            proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'single'), projection_matrix, yf, 'uniformoutput', false);
        else
            hf = cell(1,1,num_feature_blocks);
        end
        % Initialize the filter
        for k = 1:num_feature_blocks
            hf{1,1,k} = complex(zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 compressed_dim(k)], 'single'));
        end
        
        % Initialize Conjugate Gradient parameters
        CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
        init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
        sample_energy = new_sample_energy;
        rhs_samplef = cell(size(hf));
        diag_M = cell(size(hf));
        p = []; rho = []; r_old = [];
    else
        CG_opts.maxit = params.CG_iter;
        
        if params.CG_forgetting_rate == inf || params.learning_rate >= 1
            % CG will be reset
            p = []; rho = []; r_old = [];
        else
            rho = rho / (1-params.learning_rate)^params.CG_forgetting_rate;
        end
        % Update the approximate average sample energy using the learning
        % rate. This is only used to construct the preconditioner.
        sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
    end
    
    % Do training
    if frame == 1 && params.update_projection_matrix
        % Initial Gauss-Newton optimization of the filter and
        % projection matrix.
        
        % Construct stuff for the proj matrix part
        init_samplef = cellfun(@(x) permute(x, [4 3 1 2]), xlf, 'uniformoutput', false);
        init_samplef_H = cellfun(@(X) conj(reshape(X, size(X,2), [])), init_samplef, 'uniformoutput', false);
       
        % Construct preconditioner
        diag_M(1,1,:) = cellfun(@(m, reg_energy) (1-params.precond_reg_param) * bsxfun(@plus, params.precond_data_param * m, (1-params.precond_data_param) * mean(m,3)) + params.precond_reg_param*reg_energy, sample_energy, reg_energy, 'uniformoutput',false);
        diag_M(2,1,:) = cellfun(@(m) params.precond_proj_param * (m + params.projection_reg), proj_energy, 'uniformoutput',false);
        
        projection_matrix_init = projection_matrix;
        
        for iter = 1:params.init_GN_iter
            % Project sample with new matrix
            init_samplef_proj = cellfun(@(x,P) mtimesx(x, P, 'speed'), init_samplef, projection_matrix, 'uniformoutput', false);
            init_hf = cellfun(@(x) permute(x, [3 4 1 2]), hf(1,1,:), 'uniformoutput', false);
            
            % Construct the right hand side vector for the filter part
            rhs_samplef(1,1,:) = cellfun(@(xf, yf) bsxfun(@times, conj(permute(xf, [3 4 2 1])), yf), init_samplef_proj, yf, 'uniformoutput', false);
            
            % Construct the right hand side vector for the projection matrix part
            fyf = cellfun(@(f, yf) reshape(bsxfun(@times, conj(f), yf), [], size(f,3)), hf(1,1,:), yf, 'uniformoutput', false);
            rhs_samplef(2,1,:) = cellfun(@(P, XH, fyf, fi) (2*real(XH * fyf - XH(:,fi:end) * fyf(fi:end,:)) - params.projection_reg * P), ...
                projection_matrix, init_samplef_H, fyf, lf_ind, 'uniformoutput', false);
            
            % Initialize the projection matrix increment to zero
            hf(2,1,:) = cellfun(@(P) zeros(size(P), 'single'), projection_matrix, 'uniformoutput', false);
            
            % do conjugate gradient
            [hf, ~, ~, ~, res_norms_temp] = pcg_ccot(...
                @(x) lhs_operation_joint(x, init_samplef_proj, reg_filter, feature_reg, init_samplef, init_samplef_H, init_hf, params.projection_reg),...
                rhs_samplef, init_CG_opts, ...
                @(x) diag_precond(x, diag_M), ...
                [], hf);
            
            % Make the filter symmetric (avoid roundoff errors)
            hf(1,1,:) = symmetrize_filter(hf(1,1,:));
            
            % Add to the projection matrix
            projection_matrix = cellfun(@plus, projection_matrix, hf(2,1,:), 'uniformoutput', false);
            
            res_norms = [res_norms; res_norms_temp];
        end
        
        % Extract filter
        hf = hf(1,1,:);
        
        % Re-project and insert training sample
        xlf_proj = project_sample(xlf, projection_matrix);
        for k = 1:num_feature_blocks
            samplesf{k}(1,:,:,:) = permute(xlf_proj{k}, [4 3 1 2]);
        end
        
        if debug
            norm_proj_mat_init = sqrt(sum(cellfun(@(P) norm(P(:))^2, projection_matrix_init)));
            norm_proj_mat = sqrt(sum(cellfun(@(P) norm(P(:))^2, projection_matrix)));
            norm_proj_mat_change = sqrt(sum(cellfun(@(P,P2) norm(P(:) - P2(:))^2, projection_matrix_init, projection_matrix)));
            fprintf('Norm init: %f, Norm final: %f, Matrix change: %f\n', norm_proj_mat_init, norm_proj_mat, norm_proj_mat_change / norm_proj_mat_init);
        end
    else
        % Construct the right hand side vector
        rhs_samplef = cellfun(@(xf) permute(mtimesx(sample_weights, 'T', xf, 'speed'), [3 4 2 1]), samplesf, 'uniformoutput', false);
        rhs_samplef = cellfun(@(xf, yf) bsxfun(@times, conj(xf), yf), rhs_samplef, yf, 'uniformoutput', false);
        
        % Construct preconditioner
        diag_M = cellfun(@(m, reg_energy) (1-params.precond_reg_param) * bsxfun(@plus, params.precond_data_param * m, (1-params.precond_data_param) * mean(m,3)) + params.precond_reg_param*reg_energy, sample_energy, reg_energy, 'uniformoutput',false);
        
        % do conjugate gradient
        [hf, ~, ~, ~, res_norms, p, rho, r_old] = pcg_ccot(...
            @(x) lhs_operation(x, samplesf, reg_filter, sample_weights, feature_reg),...
            rhs_samplef, CG_opts, ...
            @(x) diag_precond(x, diag_M), ...
            [], hf, p, rho, r_old);
    end
    
    % Reconstruct the full Fourier series
    hf_full = full_fourier_coeff(hf);
    
    frames_since_last_train = 0;
else
    frames_since_last_train = frames_since_last_train+1;
end

% Update the scale filter
if nScales > 0 && params.use_scale_filter
    scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
end

% Update the target size (only used for computing output box)
target_sz = base_target_sz * currentScaleFactor;

% restore the size of bbox (narrowed before to avoid tracking drift)
target_sz(2) = target_sz(2) * 2.0;

%save position and calculate FPS
rect_position(frame,:) = round([pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])]);

sampled_scores_display = fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz));

pre_pos = tracker.eco.pos;
pre_target_sz = tracker.eco.target_sz;
bbox.x = double(pre_pos(2) - (pre_target_sz(2) - 1)/2);
bbox.y = double(pre_pos(1) - (pre_target_sz(1) - 1)/2);
bbox.w = pre_target_sz(2);
bbox.h = pre_target_sz(1);

if isempty(bboxes_det.fr) == 0
    o = calc_overlap(bbox, 1, bboxes_det, 1:numel(bboxes_det.fr));
    [overlap_box, index] = max(o);
    pre_area = bbox.w * bbox.h;
    cur_area = bboxes_det.w(index) * bboxes_det.h(index);
    if overlap_box > opt.det_overlap_thre && max(bbox.h, bboxes_det.h(index))/min(bbox.h, bboxes_det.h(index)) <= 1.3
        pos_det = single([bboxes_det.y(index) + (bboxes_det.h(index) - 1)/2, bboxes_det.x(index) + (bboxes_det.w(index) - 1)/2]);
        target_sz_det = [bboxes_det.h(index), bboxes_det.w(index)];
        pos = single(overlap_box * pos_det + (1 - overlap_box) * pos);
        target_sz = overlap_box * target_sz_det + (1 - overlap_box) * target_sz;
    end
end

% variables from ECO_initialize.m
tracker.eco.params = params;
tracker.eco.max_train_samples = max_train_samples;
tracker.eco.features = features;
tracker.eco.global_fparams = global_fparams;
tracker.eco.pos = pos;
tracker.eco.target_sz = target_sz;
tracker.eco.currentScaleFactor = currentScaleFactor;
tracker.eco.base_target_sz = base_target_sz;
tracker.eco.img_support_sz = img_support_sz;
tracker.eco.feature_dim = feature_dim;
tracker.eco.num_feature_blocks = num_feature_blocks;
tracker.eco.feature_reg = feature_reg;
tracker.eco.feature_extract_info = feature_extract_info;
tracker.eco.compressed_dim = compressed_dim;
tracker.eco.compressed_dim_cell = compressed_dim_cell;
tracker.eco.filter_sz = filter_sz;
tracker.eco.filter_sz_cell = filter_sz_cell;
tracker.eco.output_sz = output_sz;
tracker.eco.pad_sz = pad_sz;
tracker.eco.kx = kx;
tracker.eco.ky = ky;
tracker.eco.yf = yf;
tracker.eco.cos_window = cos_window;
tracker.eco.interp1_fs = interp1_fs;
tracker.eco.interp2_fs = interp2_fs;
tracker.eco.reg_filter = reg_filter;
tracker.eco.reg_energy = reg_energy;
tracker.eco.nScales = nScales;
tracker.eco.scaleFactors = scaleFactors;
tracker.eco.scale_filter = scale_filter;
tracker.eco.min_scale_factor = min_scale_factor;
tracker.eco.max_scale_factor = max_scale_factor;
tracker.eco.init_CG_opts = tracker.eco.init_CG_opts;
tracker.eco.CG_opts = CG_opts;
tracker.eco.rect_position = rect_position;
tracker.eco.prior_weights = prior_weights;
tracker.eco.sample_weights = sample_weights;
tracker.eco.samplesf = samplesf;
tracker.eco.score_matrix = score_matrix;
tracker.eco.latest_ind = latest_ind;
tracker.eco.frames_since_last_train = frames_since_last_train;
tracker.eco.num_training_samples = num_training_samples;
tracker.eco.minimum_sample_weight = minimum_sample_weight;
tracker.eco.res_norms = res_norms;
tracker.eco.is_color_image = is_color_image;

% variables which are useful when frame == 1
tracker.eco.sample_pos = sample_pos;
tracker.eco.sample_scale = sample_scale;
tracker.eco.xl = xl;
tracker.eco.xlf = xlf;
tracker.eco.projection_matrix = projection_matrix;
tracker.eco.shift_samp = shift_samp;
tracker.eco.xlf_proj = xlf_proj;
tracker.eco.hf = hf;
tracker.eco.lf_ind = lf_ind;
tracker.eco.proj_energy = proj_energy;
tracker.eco.sample_energy = sample_energy;
tracker.eco.rhs_samplef = rhs_samplef;
tracker.eco.diag_M = diag_M;
tracker.eco.p = p;
tracker.eco.rho = rho;
tracker.eco.r_old = r_old;
tracker.eco.init_samplef = init_samplef;
tracker.eco.init_samplef_H = init_samplef_H;
tracker.eco.projection_matrix_init = projection_matrix_init;
tracker.eco.init_samplef_proj = init_samplef_proj;
tracker.eco.init_hf = init_hf;
tracker.eco.fyf = fyf;
tracker.eco.res_norms_temp = res_norms_temp;

% other variables
tracker.eco.hf_full = hf_full;

tracker.eco.bb = double([pos([2,1]) - (target_sz([2,1]) - 1)/2, pos([2,1]) + (target_sz([2,1]) - 1)/2]);
tracker.eco.score = max(max(fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz))));

if tracker.eco.score > opt.tracking_score_thre
    tracker.eco.is_confident = 1;
else
    tracker.eco.is_confident = 0;
end
