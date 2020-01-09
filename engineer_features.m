
% Load dataset.
loaded_data = load('./cached-data/proc_seg_data3.mat');
segments = loaded_data.segments;

% Set sampling frequency.
fs = 300;

% Set length of feature vector.
len_feature_vec = 75;

% Initialize final data matrix.
data = zeros(size(segments, 1), len_feature_vec);

% Perform feature engineering for each segment.
for idx = 1:size(segments, 1)
    feature_nxt = extract_features(squeeze(segments(idx, :, :)), fs);
    data(idx, :) = feature_nxt;
end


save('./data/data_fe/data3.mat', 'data');
target = loaded_data.seg_target;
save('./data/data_fe/target3.mat', 'target');


function features = extract_features(segment, fs)
    % Extract features from segment. The parameter segment represents a mxn
    % segment containing the time series data in columns. The parameter fs
    % represents the sampling frequency.
    
    % Set number of feature that will be extracted and allocate feature
    % vecotr.
    num_features = 25;
    features = zeros(1, num_features*size(segment, 2));
    
    % Go over signals.
    idx_features = 1;
    for signal_idx = 1:size(segment, 2)
       signal_nxt = segment(:, signal_idx)';
       
       % Get features.
       feature_mu = mean(signal_nxt);
       feature_sig = std(signal_nxt);
       feature_var = var(signal_nxt);
       feature_med = median(signal_nxt);
       feature_max_t = max(signal_nxt);
       feature_min_t = min(signal_nxt);
       feature_range = feature_max_t - feature_min_t;
       feature_rms = rms(signal_nxt);
       [~, feature_argmin] = min(signal_nxt);
       [~, feature_argmax] = max(signal_nxt);
       feature_energy_t = sum(abs(signal_nxt).^2);
       feature_ent = entropy(signal_nxt/max(abs(signal_nxt)));
       feature_skw = skewness(signal_nxt);
       feature_krt = kurtosis(signal_nxt);
       feature_iqr = iqr(signal_nxt);
       feature_mad = mad(signal_nxt);
       [feature_wd, feature_lo, feature_hi, feature_power] = obw(signal_nxt, fs);
       
       signal_nxt_f = fft(signal_nxt)/length(signal_nxt);
       feature_bndpwr = bandpower(signal_nxt_f);
       feature_energy_f = sum(abs(signal_nxt_f));
       feature_mean_f = mean(abs(signal_nxt_f));
       feature_max_f = max(abs(signal_nxt_f));
       feature_min_f = min(abs(signal_nxt_f));
       % feature_normc = fftshift(abs(signal_nxt_f)); 
       
       % Stack into
       features_vec_nxt = [
                                         feature_mu 
                                         feature_sig 
                                         feature_var 
                                         feature_med 
                                         feature_max_t
                                         feature_min_t
                                         feature_range
                                         feature_rms
                                         feature_argmin
                                         feature_argmax
                                         feature_wd
                                         feature_lo
                                         feature_hi
                                         feature_power
                                         feature_energy_t
                                         feature_ent
                                         feature_skw
                                         feature_krt
                                         feature_iqr
                                         feature_mad
                                         feature_bndpwr
                                         feature_energy_f
                                         feature_mean_f
                                         feature_max_f
                                         feature_min_f
                                         % feature_normc(1:10)
                                         ];
       % features_vec_nxt = cat(2, features_vec_nxt', feature_normc);
       features(idx_features:idx_features+num_features-1) = features_vec_nxt;
       idx_features = idx_features + num_features;
    end
end

