% return a subset of the structure s
function s  = sub_bboxes(s, index_sub)

if ~isempty(s),
  field_list = fieldnames(s);
  for i = 1:length(field_list),
    f = field_list{i};
    s.(f) = s.(f)(index_sub,:);
  end
end





