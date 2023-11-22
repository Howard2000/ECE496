%% normalize training sets to -1 1

function out = convert_data_to_range_neg1_1(input_data,minrange,maxrange,minvalue,maxvalue)
      
    if nargin == 1
        [a,b] = size(input_data);

        out = zeros(a,b);

        for i = 1:b;
            out(:,i) = map_rng1_2_rng2(input_data(:,i),min(input_data(:,i)), ...
                max(input_data(:,i)),-1,1);
        end
    elseif nargin <= 3
        [a,b] = size(input_data);

        out = zeros(a,b);

        for i = 1:b;
            out(:,i) = map_rng1_2_rng2(input_data(:,i),min(input_data(:,i)), ...
                max(input_data(:,i)),minrange,maxrange);
        end
    else
        [a,b] = size(input_data);

        out = zeros(a,b);

        for i = 1:b;
            out(:,i) = map_rng1_2_rng2(input_data(:,i),minvalue, ...
                maxvalue,minrange,maxrange);
        end
    end
end