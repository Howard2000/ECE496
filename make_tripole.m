function [tripole_ref_full tripole_short] = make_tripole(recording);

    tripole_ref_full = recording - kron(mean(recording([1:8 49:56],:)),ones(56,1));
    tripole_short = tripole_ref_full(9:48,:);

end
