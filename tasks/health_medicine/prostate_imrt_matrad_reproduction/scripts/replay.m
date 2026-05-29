function replay(replay_state_path, out_dir)
%REPLAY  Load a clean replay state and recompute dose. Avoids classdef/load issues.

fprintf('\n=== replay.m ===\n');
fprintf('Loading %s\n', replay_state_path);

data = load(replay_state_path);

w = data.w_replay;
D = data.dij_physicalDose;
dims = data.dij_dimensions;
nfx = data.nFractions;

fprintf('Reconstructing dose: D (%d x %d) * w (%d) -> dims %dx%dx%d\n', ...
  size(D, 1), size(D, 2), numel(w), dims);

% Per-fraction dose
frac_dose = reshape(D * w, dims);
total = frac_dose * nfx;

out_bin = fullfile(out_dir, 'replayed_dose.bin');
out_shape = fullfile(out_dir, 'replayed_shape.txt');
fid = fopen(out_bin, 'wb');
fwrite(fid, double(total(:)), 'double');
fclose(fid);
fid = fopen(out_shape, 'w');
fprintf(fid, '%d %d %d\n', size(total));
fclose(fid);

fprintf('Replayed dose: max=%.2f Gy, mean=%.4f Gy\n', max(total(:)), mean(total(:)));
fprintf('=== replay done ===\n');
