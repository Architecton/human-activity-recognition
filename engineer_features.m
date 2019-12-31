data = load('proc_seg_data.mat');
segments = data.segments;
seg_target = data.target_encoded;

fs = 1200;

features = extract_features(squeeze(segments(1, :, :)), fs);

function features = extract_features(segment, fs)
    num_features = 15;
    features = zeros(1, num_features*size(segment, 2));
    idx_features = 1;
    for signal_idx = 1:size(segment, 2)
       signal_nxt = segment(:, signal_idx)';
       feature_mu = mean(signal_nxt);
       feature_sig = std(signal_nxt);
       feature_var = var(signal_nxt);
       feature_med = median(signal_nxt);
       feature_max = max(signal_nxt);
       feature_min = min(signal_nxt);
       feature_range = feature_max - feature_min;
       feature_rms = rms(signal_nxt);
       [~, feature_argmin] = min(signal_nxt);
       [~, feature_argmax] = max(signal_nxt);
       [feature_wd, feature_lo, feature_hi, feature_power] = obw(s, fs);
       feature_energy = sum(abs(signal_nxt).^2);
       feature_ent = entropy(signal_nxt/max(abs(signal_nxt)));
       feature_skw = skewness(signal_nxt);
       feature_krt = kurtosis(signal_nxt);
       feature_iqr = iqr(signal_nxt);
       feature_mad = mad(signal_nxt);
       
       features(idx_features:idx_features+num_features-1) = [
                                         feature_mu 
                                         feature_sig 
                                         feature_var 
                                         feature_med 
                                         feature_max 
                                         feature_min 
                                         feature_range
                                         feature_rms
                                         feature_argmin
                                         feature_argmax
                                         feature_wd
                                         feature_lo
                                         feature_hi
                                         feature_power
                                         feature_energy
                                         feature_ent
                                         feature_skw
                                         feature_krt
                                         feature_iqr
                                         feature_mad
                                         ];
       idx_features = idx_features + num_features;
    end
    
    % TODO
    feature_sma = 0;
end

