# BEST scared config. Pruning fix + depth reg, no anisotropy. Warpback PSNR, offset views,
# mean of 3 seeds:
#   elev              5      10      15      20      30      45
#   baseline      26.71   24.64   23.02   21.81   19.58   16.02
#   prunefix only 27.12   25.35   23.85   22.13   18.58   13.01
#   + aniso+depth 28.54   27.11   25.29   23.30   18.49   12.98
#   this          28.62   27.29   25.64   23.66   19.76   14.90
# Clean wins over baseline (no seed overlap) at elev5-20, and unlike the +aniso version it does
# not lose at elev45 -- ranges overlap there instead. It beats +aniso cleanly at elev45
# (14.54-15.28 vs 10.67-14.31), so the anisotropy term cost ~2dB at wide angles and bought
# nothing elsewhere. Also the most repeatable: elev5 spans 0.10dB across seeds, elev10 0.37dB.
# Both terms are needed though -- prunefix alone overlaps baseline everywhere (one seed collapsed
# to 25.65 at elev5), so the depth term is doing real work.
#
# prune_scale_extent_ratio=10.0 fires on nothing here (needs scale >10*extent(16.13)=161, largest
# gaussian is ~134). The pruning gain is opacity-only, and it is "prune at all" rather than "prune
# more often": at the baseline's interval of 2000, iteration % 2000 == 0 only at 2000, and the
# save (train_eval.py:374) runs before the prune block (:486), so the baseline never prunes its
# saved model. Only ~365 of ~56k gaussians are removed, so the size of the effect is unexplained.

ModelParams = dict(
    extra_mark = 'scared',
    no_fine=False,
    init_pts=20_000
)

OptimizationParams = dict(
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
    sideview_elev = 20,
    sideview_azims = -1,                 # sample azimuth uniformly from [0, 360)
    anisotropy_weight = 0,               # prunefix + depth reg only
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
