function hf_out = my_lhs_operation(hf, samplesf, reg_filter, sample_weights, feature_reg, frame, u)

% This is the left-hand-side operation in Conjugate Gradient

% size of the padding
num_features = length(hf);
output_sz = [size(hf{1},1), 2*size(hf{1},2)-1];

% zhuji{ 
% Compute xf
%u = single(ones(output_sz));
if frame > 1
    xhf_cell = cell(1,1,num_features);
 %{   
    for k = 1:num_features
        xhf_cell{k} = bsxfun(@times, samplesf{k}, permute(hf{k}, [4,3,1,2]));
    end

    xhf_cell_perm = cellfun(@(xhf) permute(xhf, [3,4,2,1]), xhf_cell, 'uniformoutput', false);
    pad_sz = cellfun(@(hf) (output_sz - [size(hf,1), 2*size(hf,2)-1]) / 2, hf, 'uniformoutput', false);
    xhf_full = cellfun(@(xhf) cat(2, xhf, conj(rot90(xhf(:,1:end-1,:,:), 2))), xhf_cell_perm, 'uniformoutput', false);
    xhf_pad = cellfun(@(xhf, pad_sz) padarray(xhf, pad_sz), xhf_full, pad_sz, 'uniformoutput', false);
    xht_sampled = cellfun(@(xhf) sample_fs(xhf), xhf_pad, 'uniformoutput', false);    
    %uxht_sampled = cellfun(@(u,xht) bsxfun(@times, u, xht), u, xht_sampled, 'uniformoutput', false);
    %uxht_sz = cellfun(@(uxht) [size(uxht, 1), size(uxht, 1)], uxht_sampled);
    %uxhf_sampled = cellfun(@(uxht,sz) cfft2(uxht)/prod(sz),  uxht_sampled, uxht_sz, 'uniformoutput', false);
    u2 = u.^2;
    u2xht_sampled = cellfun(@(xht) bsxfun(@times, u2, xht), xht_sampled, 'uniformoutput', false);
    u2xht_sz = cellfun(@(uxht) [size(uxht, 1), size(uxht, 2)], u2xht_sampled, 'uniformoutput', false);
    u2xhf = cellfun(@(uxht,sz) cfft2(uxht)/prod(sz),  u2xht_sampled, u2xht_sz, 'uniformoutput', false);
    u2xhf_compact = cellfun(@(u2xhf) u2xhf(:, 1:(size(u2xhf,2)+1)/2, :, :), u2xhf, 'uniformoutput', false);
    u2xhf_sampled = cellfun(@(u2xhf, pad_sz) u2xhf(1+pad_sz(1):end-pad_sz(1), 1+pad_sz(2):end, :, :), u2xhf_compact, pad_sz, 'uniformoutput', false);
    u2xhf_sampled_perm = cellfun(@(u2xhf) permute(u2xhf, [4,3,1,2]), u2xhf_sampled, 'uniformoutput', false);
    xu2xhf = cellfun(@(samplesf, u2xhf) bsxfun(@times, conj(samplesf), u2xhf), samplesf, u2xhf_sampled_perm, 'uniformoutput', false);
    hf_out = cellfun(@(xu2xhf) permute(mtimesx(sample_weights, 'T', xu2xhf, 'speed'), [3,4,2,1]), xu2xhf, 'uniformoutput', false);
%}
    for k = 1:num_features
        xhf_cell{k} = mtimesx(samplesf{k}, permute(hf{k}, [3 4 1 2]), 'speed');
    end

    % sum over all feature blocks
    xhf = xhf_cell{1};    % assumes the feature with the highest resolution is first
    pad_sz = cell(1,1,num_features);
    for k = 2:num_features
        pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;        
        xhf(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
            xhf(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + xhf_cell{k};
    end
    xhf_perm = permute(xhf, [3,4,1,2]);
    xhf_full = cat(2, xhf_perm, conj(rot90(xhf_perm(:,1:end-1,:), 2)));
    xht_sampled = sample_fs(xhf_full);  
    u2 = u.^2;
    u2xht_sampled = bsxfun(@times, u2, xht_sampled);
    u2xhf = cfft2(u2xht_sampled) / prod(output_sz);
    u2xhf_compact = u2xhf(:, 1:(size(u2xhf,2)+1)/2, :);
    u2xhf_perm = permute(u2xhf_compact, [3,4,1,2]);
    u2xhf_perm = bsxfun(@times,sample_weights, u2xhf_perm);

    % multiply with the transpose
    hf_out = cell(1,1,num_features);
    hf_out{1} = permute(conj(mtimesx(u2xhf_perm, 'C', samplesf{1}, 'speed')), [3 4 2 1]);
    for k = 2:num_features
        hf_out{k} = permute(conj(mtimesx(u2xhf_perm(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf{k}, 'speed')), [3 4 2 1]);
    end
    
else
% zhuji} 
    % Compute the operation corresponding to the data term in the optimization
    % (blockwise matrix multiplications)
    %implements: A' diag(sample_weights) A f

    % sum over all features in each block
    sh_cell = cell(1,1,num_features);
    for k = 1:num_features
        sh_cell{k} = mtimesx(samplesf{k}, permute(hf{k}, [3 4 1 2]), 'speed');
    end

    % sum over all feature blocks
    sh = sh_cell{1};    % assumes the feature with the highest resolution is first
    pad_sz = cell(1,1,num_features);
    for k = 2:num_features
        pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
        
        sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
            sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + sh_cell{k};
    end

    % weight all the samples
    sh = bsxfun(@times,sample_weights,sh);

    % multiply with the transpose
    hf_out = cell(1,1,num_features);
    hf_out{1} = permute(conj(mtimesx(sh, 'C', samplesf{1}, 'speed')), [3 4 2 1]);
    for k = 2:num_features
        hf_out{k} = permute(conj(mtimesx(sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf{k}, 'speed')), [3 4 2 1]);
    end
end
% compute the operation corresponding to the regularization term (convolve
% each feature dimension with the DFT of w, and the tramsposed operation)
% add the regularization part
% hf_conv = cell(1,1,num_features);
for k = 1:num_features
    reg_pad = min(size(reg_filter{k},2)-1, size(hf{k},2)-1);
    
    % add part needed for convolution
    hf_conv = cat(2, hf{k}, conj(rot90(hf{k}(:, end-reg_pad:end-1, :), 2)));
    
    % do first convolution
    hf_conv = convn(hf_conv, reg_filter{k});
    
    % do final convolution and put toghether result
    hf_out{k} = hf_out{k} + convn(hf_conv(:,1:end-reg_pad,:), reg_filter{k}, 'valid');
end

end