"""Preserved experiment registry from before streamlining (not used by default)."""

EXPERIMENT_GROUPS_LEGACY = {

    # =========================================================================
    # NEW: Incremental R-value sweep experiments
    # =========================================================================

    'vanilla_gnn_node_repaired': {
        'experiment_name': 'vanilla_gnn_node_repaired',
        'variants': [
            {
                'variant_id': 'no_attention',
                'gsat_overrides': {
                    'tuning_id': 'no_attention',
                    'no_attention': True,
                    'fix_r': 1.0,
                    'info_loss_coef': 0,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'vanilla_gnn_clean': {
        'experiment_name': 'vanilla_gnn_clean',
        'variants': [
            {
                'variant_id': 'clean',
                'gsat_overrides': {
                    'tuning_id': 'clean',
                    'fix_r': 1.0,
                },
                'learn_edge_att': False,
                'vanilla_clean': True,
            },
        ],
    },

    'base_gsat_fix_r_node_repaired': {
        'experiment_name': 'base_gsat_fix_r_node_repaired',
        'variants': [
            {
                'variant_id': f'fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        ],
    },

    'base_gsat_decay_r_node_repaired': {
        'experiment_name': 'base_gsat_decay_r_node_repaired',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        ],
    },

    'motif_readout_fix_r_repaired': {
        'experiment_name': 'motif_readout_fix_r_repaired',
        'variants': [
            {
                'variant_id': f'readout_fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'readout_fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': 'readout',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [1.0, 0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_fix_r_mean': {
        'experiment_name': 'motif_readout_fix_r_mean',
        'variants': [
            {
                'variant_id': f'readout_mean_fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_fix_r_sum': {
        'experiment_name': 'motif_readout_fix_r_sum',
        'variants': [
            {
                'variant_id': f'readout_sum_fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'readout_sum_fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'sum',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean': {
        'experiment_name': 'motif_readout_decay_r_mean',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_sum': {
        'experiment_name': 'motif_readout_decay_r_sum',
        'variants': [
            {
                'variant_id': f'readout_sum_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_sum_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'sum',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.9, 0.8, 0.7]
        ],
    },

    # =========================================================================
    # Explainer analysis experiments (score-vs-impact with r highlight)
    # New names to avoid overwriting previous results.
    # =========================================================================

    'base_gsat_decay_r_explainer': {
        'experiment_name': 'base_gsat_decay_r_explainer',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_explainer': {
        'experiment_name': 'motif_readout_decay_r_mean_explainer',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': False,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_sampling_explainer': {
        'experiment_name': 'motif_readout_decay_r_mean_sampling_explainer',
        'variants': [
            {
                'variant_id': f'readout_mean_sampling_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_sampling_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Info loss coefficient sweep for motif readout + sampling
    # Base: motif_readout_decay_r_mean_sampling_explainer (info_loss_coef=1)
    # Sweep: 0.1, 0.25, 0.5, 0.75, 1.0 (baseline), 1.5, 2.0 × final_r {0.8, 0.7}
    # =========================================================================

    'motif_readout_sampling_info_coef_sweep': {
        'experiment_name': 'motif_readout_sampling_info_coef_sweep',
        'variants': [
            {
                'variant_id': f'info{c}_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'info{c}_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'info_loss_coef': c,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                },
                'learn_edge_att': False,
            }
            for c in [0.0, 0.25, 0.5, 1.0, 2.0]
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Extractor MLP capacity sweep (motif readout + sampling)
    # Base: motif_readout_decay_r_mean_sampling_explainer (extractor_hidden_mult=1)
    # Sweep: mult 1 (baseline), 2, 4 × final_r {0.8, 0.7}
    # =========================================================================

    'motif_readout_sampling_extractor_sweep': {
        'experiment_name': 'motif_readout_sampling_extractor_sweep',
        'variants': [
            {
                'variant_id': f'ext{m}_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'ext{m}_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                },
                'learn_edge_att': False,
                'shared_overrides': {'extractor_hidden_mult': m},
            }
            for m in [1, 2, 4]
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Rich motif readout: concatenate mean+max+sum pooling (3× embedding width)
    # Tests whether richer motif representations improve attention quality
    # =========================================================================

    'motif_readout_sampling_rich_pool': {
        'experiment_name': 'motif_readout_sampling_rich_pool',
        'variants': [
            {
                'variant_id': f'rich_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'rich_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'multi',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Motif-level info loss variants (compare with the originals above)
    # Same as base/readout/sampling explainer but with motif_level_info_loss=True
    # =========================================================================

    'base_gsat_decay_r_explainer_motif_info': {
        'experiment_name': 'base_gsat_decay_r_explainer_motif_info',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'motif_level_info_loss': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_explainer_motif_info': {
        'experiment_name': 'motif_readout_decay_r_mean_explainer_motif_info',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': False,
                    'motif_level_info_loss': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_sampling_explainer_motif_info': {
        'experiment_name': 'motif_readout_decay_r_mean_sampling_explainer_motif_info',
        'variants': [
            {
                'variant_id': f'readout_mean_sampling_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_sampling_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                    'motif_level_info_loss': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Warmup experiments: prediction-only warmup before info loss kicks in
    # Conservative schedule: higher init_r, slower decay, lower info_loss_coef
    # =========================================================================

    'base_gsat_decay_r_explainer_warmup': {
        'experiment_name': 'base_gsat_decay_r_explainer_warmup',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_interval': 20,
                    'decay_r': 0.05,
                    'info_loss_coef': 0.5,
                    'info_warmup_epochs': 30,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_explainer_warmup': {
        'experiment_name': 'motif_readout_decay_r_mean_explainer_warmup',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_interval': 20,
                    'decay_r': 0.05,
                    'info_loss_coef': 0.5,
                    'info_warmup_epochs': 30,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': False,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_sampling_explainer_warmup': {
        'experiment_name': 'motif_readout_decay_r_mean_sampling_explainer_warmup',
        'variants': [
            {
                'variant_id': f'readout_mean_sampling_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_sampling_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_interval': 20,
                    'decay_r': 0.05,
                    'info_loss_coef': 0.5,
                    'info_warmup_epochs': 30,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Injection point ablation with motif-level sampling
    # All use motif readout + mean pooling + motif_level_sampling=True
    # Varying: w_feat, w_message, w_readout, learn_edge_att
    # =========================================================================

    'motif_injection_node': {
        'experiment_name': 'motif_injection_node',
        'variants': [
            {
                'variant_id': f'node_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'node_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                    'w_feat': False,
                    'w_message': True,
                    'w_readout': False,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_injection_node_readout': {
        'experiment_name': 'motif_injection_node_readout',
        'variants': [
            {
                'variant_id': f'node_readout_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'node_readout_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                    'w_feat': False,
                    'w_message': True,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_injection_readout_only': {
        'experiment_name': 'motif_injection_readout_only',
        'variants': [
            {
                'variant_id': f'readout_only_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_only_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                    'w_feat': False,
                    'w_message': False,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_injection_edge_readout': {
        'experiment_name': 'motif_injection_edge_readout',
        'variants': [
            {
                'variant_id': f'edge_readout_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'edge_readout_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Legacy experiments (kept for reference, still functional)
    # =========================================================================

    'r_impact_node': {
        'experiment_name': 'r_impact_node',
        'variants': [
            {
                'variant_id': f'node_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'node_r{r}',
                    'final_r': r,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [0.4, 0.5, 0.6]
        ],
    },

    'r_impact_edge': {
        'experiment_name': 'r_impact_edge',
        'variants': [
            {
                'variant_id': f'edge_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'edge_r{r}',
                    'final_r': r,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': True,
            }
            for r in [0.4, 0.5, 0.6]
        ],
    },

    'within_motif_consistency_impact': {
        'experiment_name': 'within_motif_consistency_impact',
        'variants': [
            {
                'variant_id': 'within_w1',
                'gsat_overrides': {
                    'tuning_id': 'within_w1',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'within_w2',
                'gsat_overrides': {
                    'tuning_id': 'within_w2',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 2.0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'between_motif_consistency_impact': {
        'experiment_name': 'between_motif_consistency_impact',
        'variants': [
            {
                'variant_id': 'fisher_w1_b1',
                'gsat_overrides': {
                    'tuning_id': 'fisher_w1_b1',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 1.0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'fisher_w1_b2',
                'gsat_overrides': {
                    'tuning_id': 'fisher_w1_b2',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 2.0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'motif_readout_info_loss': {
        'experiment_name': 'motif_readout_info_loss',
        'variants': [
            {
                'variant_id': 'readout_motif_info_r0.5',
                'gsat_overrides': {
                    'tuning_id': 'readout_motif_info_r0.5',
                    'final_r': 0.5,
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'motif_readout_adaptive_r': {
        'experiment_name': 'motif_readout_adaptive_r',
        'variants': [
            {
                'variant_id': 'readout_targetk1',
                'gsat_overrides': {
                    'tuning_id': 'readout_targetk1',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'target_k': 1.0,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'readout_targetk2',
                'gsat_overrides': {
                    'tuning_id': 'readout_targetk2',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'target_k': 2.0,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'att_injection_point': {
        'experiment_name': 'att_injection_point',
        'variants': [
            {
                'variant_id': 'w_feat_only',
                'gsat_overrides': {
                    'tuning_id': 'w_feat_only',
                    'w_feat': True,
                    'w_message': False,
                    'w_readout': False,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'w_message_only',
                'gsat_overrides': {
                    'tuning_id': 'w_message_only',
                    'w_feat': False,
                    'w_message': True,
                    'w_readout': False,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'w_readout_only',
                'gsat_overrides': {
                    'tuning_id': 'w_readout_only',
                    'w_feat': False,
                    'w_message': False,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'w_feat_readout',
                'gsat_overrides': {
                    'tuning_id': 'w_feat_readout',
                    'w_feat': True,
                    'w_message': False,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            },
        ],
    },
}
