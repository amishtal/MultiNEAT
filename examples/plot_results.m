function [] = plot_results(fname, smooth_factor)
    %smooth_factor = 1;
    n_generations = -1;
    try
        n_generations = h5read(fname, '/n_generations');
    catch err
        n_generations = length(h5read(fname, '/coop/max'));
    end
    disp(n_generations);

    coop_max = h5read(fname, '/coop/max');
    coop_min = h5read(fname, '/coop/min');
    coop_mean= h5read(fname, '/coop/mean');
    coop_std = h5read(fname, '/coop/std');
    plot_set('Cooperation', 1, smooth_factor, coop_max(1:n_generations), coop_min(1:n_generations), coop_mean(1:n_generations), coop_std(1:n_generations));

    fit_max = h5read(fname, '/fit/max');
    fit_min = h5read(fname, '/fit/min');
    fit_mean= h5read(fname, '/fit/mean');
    fit_std = h5read(fname, '/fit/std');
    plot_set('Fitness', 2, smooth_factor, fit_max(1:n_generations), fit_min(1:n_generations), fit_mean(1:n_generations), fit_std(1:n_generations));

    int_max = h5read(fname, '/intel/max');
    int_min = h5read(fname, '/intel/min');
    int_mean= h5read(fname, '/intel/mean');
    int_std = h5read(fname, '/intel/std');
    plot_set('Intelligence', 3, smooth_factor, int_max(1:n_generations), int_min(1:n_generations), int_mean(1:n_generations), int_std(1:n_generations));

    figure(4);
    clf;
    hold on;
    coop_fit_corr   = h5read(fname, '/corr/coop_fit');
    coop_fit_corr(isnan(coop_fit_corr)) = 0;
    intel_coop_corr = h5read(fname, '/corr/intel_coop');
    intel_coop_corr(isnan(intel_coop_corr)) = 0;
    intel_fit_corr  = h5read(fname, '/corr/intel_fit');
    intel_fit_corr(isnan(intel_fit_corr)) = 0;
    plot(smooth_(coop_fit_corr(1:n_generations), smooth_factor), 'r');
    plot(smooth_(intel_coop_corr(1:n_generations), smooth_factor), 'b');
    plot(smooth_(intel_fit_corr(1:n_generations), smooth_factor), 'k');
    legend({'Coop/Fit', 'Intel/Coop', 'Intel/Fit'});
    title('Correlations');
    xlabel('Time (Generations)');
    ylabel('Pearsons CC');

    figure(5);
    clf;
    hold on;
    ac = h5read(fname, '/strategy/ac');
    ad = h5read(fname, '/strategy/ad');
    tft = h5read(fname, '/strategy/tft');
    tftt = h5read(fname, '/strategy/tftt');
    pavlov = h5read(fname, '/strategy/pavlov');
    plot(smooth_(ac(1:n_generations), smooth_factor), 'r');
    plot(smooth_(ad(1:n_generations), smooth_factor), 'b');
    plot(smooth_(tft(1:n_generations), smooth_factor), 'g');
    plot(smooth_(tftt(1:n_generations), smooth_factor), 'c');
    plot(smooth_(pavlov(1:n_generations), smooth_factor), 'm');
    legend({'ac', 'ad', 'tft', 'tftt', 'pavlov'});
    title('Strategy Distributions');
    xlabel('Time (Generations)');
    ylabel('Percentage of Individuals');

    figure(6);
    clf;
    hold on;
    sel_for_intel = h5read(fname, '/sel_for_intel');
    plot(smooth_(sel_for_intel(1:n_generations), smooth_factor), 'b');
    legend({'sel for intel'});
    title('Selection for Intelligence');
    xlabel('Time (Generations)');

end

function [] = plot_set(dname, fig, smooth_factor, dmax, dmin, dmean, dstd)
    figure(fig);
    clf;
    hold on;

    [xs, ys] = get_std_area(dmean, dstd, smooth_factor);
    plot(xs, ys, 'g.', 'MarkerSize', 3.0);
    plot(smooth_(dmean, smooth_factor), 'r', 'LineWidth', 1);
    plot(smooth_(dmax, smooth_factor), 'b', 'LineWidth', 1);
    plot(smooth_(dmin, smooth_factor), 'k', 'LineWidth', 1);

    title(dname);
    xlabel('Time (Generations)');
    ylabel(dname);
    legend({'mean+/-std', 'mean', 'max', 'min'}, 'Location', 'Best');
end

function [xs, ys] = get_std_area(dmean, dstd, smooth_factor)
    n = 9.0;
    xs = repmat(1:length(dmean), n, 1);
    a = smooth_(dmean + dstd, smooth_factor);
    b = smooth_(dmean - dstd, smooth_factor);
    l = length(dmean);
    ys = zeros(n, l);
    for i = 1:n
        ys(i, :) = ((1 - (i-1)/(n-1)) * a) + (((i-1)/(n-1)) * b);
        %plot(((1 - (i-1)/1000) * a) + (((i-1)/1000) * b));
    end
    xs = reshape(xs, n*l, 1);
    ys = reshape(ys, n*l, 1);
end

function [y] = smooth_(x, n)
    y = smooth(x, n);
    %y = smooth(x, n, 'lowess');
    %y = smooth(x, n, 'loess');
    %y = smooth(x, n, 'rlowess');
    %y = smooth(x, n, 'rloess');
    %y = smooth(x, n, 'sgolay');
end

