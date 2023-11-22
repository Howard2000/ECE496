%USING MAKE TRIPOLES and following general structure of provided processing
%code

clearvars -except amplifier_data MPH MPD Threshold Artefact_Threshold notch_on dt dir fc1 fc2 fs n autothreshold mult ratnum RAW_data_pathway Preprocessing_pathway inbet1 inbet2 inbet3;

MPDistance = 90;
autothreshold = 0;
notch_on = 2; 
dt = 1; 
dir = 2; 

read_Intan_RHD2000_file

[b,a] = butter(6,[10/(30000/2) 7500/(30000/2)],'bandpass');
filtered = filtfilt(b,a,amplifier_data);
tripoles = make_tripole(filtered);
ring1 = mean(tripoles(1:8,:));
ring2 = mean(tripoles(9:16,:));
ring3 = mean(tripoles(17:24,:));
ring4 = mean(tripoles(25:32,:));
ring5 = mean(tripoles(33:40,:));
ring6 = mean(tripoles(41:48,:));
ring7 = mean(tripoles(49:56,:));

avarage = [zeros(1,3) ring1(:,1:end-3)] + [zeros(1,2) ring2(:,1:end-2)] + [zeros(1,1) ring3(:,1:end-1) ] + ring4 + [ring5(:,2:end) zeros(1,1)] + [ring6(:,3:end) zeros(1,2)]  + [ring7(:,4:end) zeros(1,3)];

MinPeakHeight = 4*median(abs(avarage))/0.6745;

[peaks locs] = findpeaks(avarage,'MinPeakHeight', MinPeakHeight,'MinPeakDistance', MPDistance);

for i = 1:length(peaks);
    if locs(i) - 100 <= 0;
        remove(i) = 1;
    elseif locs(i) + 100 > length(filtered);
        remove(i) = 1;
    end
end

if exist('remove') ~= 0;
    peaks(find(remove==1)) = [];
    locs(find(remove==1)) = [];
end

count = 1;
remove = [];
for spike = 1:length(peaks);
    section = tripoles(:,locs(spike)-49:locs(spike)+50);
    [section_peaks section_locs] = findpeaks(mean(section(25:32,45:55)));
    if length(section_locs) > 1;
        [value number] = max(section_peaks);
        section_locs = section_locs(number);
        if section_locs == 6;
            spikes(:,:,spike) = tripoles(:,locs(spike)-49:locs(spike)+50);
            locs(spike) = locs(spike);
        elseif section_locs < 6;
            spikes(:,:,spike) = tripoles(:,locs(spike)-49-(6-section_locs):locs(spike)+50-(6-section_locs));
            locs(spike) = locs(spike)-(6-section_locs);
        else
            spikes(:,:,spike) = tripoles(:,locs(spike)-49+(section_locs-6):locs(spike)+50+(section_locs-6));
            locs(spike) = locs(spike)+(section_locs-6);
        end
    elseif isempty(section_locs);
        remove(count) = spike;
        count = count+1;
    else
        if section_locs == 6;
            spikes(:,:,spike) = tripoles(:,locs(spike)-49:locs(spike)+50);
            locs(spike) = locs(spike);
        elseif section_locs < 6;
            spikes(:,:,spike) = tripoles(:,locs(spike)-49-(6-section_locs):locs(spike)+50-(6-section_locs));
            locs(spike) = locs(spike)-(6-section_locs);
        else
            spikes(:,:,spike) = tripoles(:,locs(spike)-49+(section_locs-6):locs(spike)+50+(section_locs-6));
            locs(spike) = locs(spike)+(section_locs-6);
        end
    end
%         plot(mean(spikes(25:32,:,k)))
    clear section section_peaks section_locs
end
if length(remove) > 0;
    locs(remove) = [];
    spikes(:,:,remove) = [];
end
remove = mean(spikes(25:32,50,:)) > 15;
if length(remove) > 0;
    locs(remove) = [];
    spikes(:,:,remove) = [];
end

plot(avarage,'-s', 'MarkerIndices', locs)


location = input("Select sample to plot");

image(spikes(:,:,location),'CDataMapping','scaled')
    






