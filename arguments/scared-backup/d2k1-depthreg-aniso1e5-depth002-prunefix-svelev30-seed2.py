# RESULT: shifts the useful window rather than widening it. 3-seed mean gains 2.15dB at
# elev45 (15.13 vs 12.98) but gives back 1.7dB at elev20, landing at baseline level. Only
# one clean win over baseline (elev5) vs four for sideview_elev=20. Keep 20.

# Best scared config found so far. Warpback PSNR vs d2k1 baseline, mean of 3 seeds, offset views:
#   elev      5     10     15     20     30     45
#   baseline  26.71  24.64  23.02  21.81  19.58  16.02
#   this      28.54  27.11  25.29  23.30  18.49  12.98
# Clean wins (no seed overlap) at elev5-20; clean loss at elev45. See arguments/scared-backup/ for
# variants that did NOT help: -thresh5 (anisotropy hinge 5, loses the elev15/20 wins),
# -depth001/-depth0005/-depth004/-depth008 (depth weight sweep, unresolvable at 3 seeds),
# -oreset1200 (fixes the wide-angle collapse but costs the elev10-20 wins),
# -oreset1500-it3000 (unstable), -dens2500 (worse than baseline everywhere).

ModelParams = dict(
    extra_mark = 'scared',
    no_fine=False,
    init_pts=20_000
)

OptimizationParams = dict(
    seed = 2,
    coarse_iterations = 1000,
    iterations = 2000,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 2000,

    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,

    pruning_interval = 100,              # was 2000, which with iterations=2000 pruned about once
    prune_scale_extent_ratio = 10.0,     # was -1 (off) -- scale.max() > 10 * extent(16.13) = 161 pruned as floater
    percent_dense = 0.01,
    opacity_reset_interval = 3000,

    # depth reg + anisotropy reg + endonerf-style floater pruning, matching endonerf/cutting-depthreg-aniso1e5-depth002. The whole
    # block is copied rather than just the two weights that differ from the endonerf baseline,
    # since sideview_reg_interval defaults to -1 (off) and the shape params default differently
    sideview_smooth_weight = 0,          # color TV loss off, as on cutting
    sideview_depth_weight = 0.02,
    sideview_depth_huber_beta = 1.0,
    sideview_reg_interval = 5,
    sideview_elev = 30,                  # was 20 -- the clean wins stop at elev20, which is exactly where the regularizer is applied
    sideview_azims = -1,                 # sample azimuth uniformly from [0, 360)
    anisotropy_weight = 1e-5,
    anisotropy_ratio_power = 2.0,
    anisotropy_ratio_threshold = 10.0,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 100]
    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
    no_dx=False,
    no_ds=True,
    no_dr=True,
    no_do=False
)
