% concatenate two bboxes structs
function bboxes_concat = concatenate_bboxes(bboxes1, bboxes2)

if isempty(bboxes2)
    bboxes_concat = bboxes1;
else
    field_list = fieldnames(bboxes1);
    for i = 1:length(field_list),
        f = field_list{i};
        bboxes_concat.(f) = [bboxes1.(f); bboxes2.(f)];
    end
end