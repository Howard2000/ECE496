function out = map_rng1_2_rng2(value,rng1a,rng1b,rng2a,rng2b)



    if size(value,1) == 1 && size(value,2) == 1;
        out = rng2a + (rng2b - rng2a)/(rng1b - rng1a)*(value-rng1a);
    else
        out = repmat(rng2a,size(value,1),size(value,2)) + repmat((rng2b - rng2a)/(rng1b - rng1a),size(value,1),size(value,2)).*(value-repmat(rng1a,size(value,1),size(value,2)));
    end
    
end