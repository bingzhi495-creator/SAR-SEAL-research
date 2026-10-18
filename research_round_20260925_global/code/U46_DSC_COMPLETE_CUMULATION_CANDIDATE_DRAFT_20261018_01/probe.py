"""Candidate coordinate-only probe; Torch and the candidate DSC are injected.

No scientific import, main, runner, repaired operator, feature interpolation,
Conv/BN/network construction, data, optimizer or qualification is included.
"""


def run_diagnostic(torch, DSC, report):
    """Record candidate AD versus central FD and complete outward accumulation.

    The actual owner must close the exact candidate DSC source before injection.
    Returning a diagnostic, including expected numerical RED evidence, does not
    repair the operator or qualify a full model. Execution status belongs to the separate actual summary.
    """
    if type(report) is not dict or report:
        raise ValueError('Caller retains a fresh report, including on failure')
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only probe requires CUDA uninitialized')
    original_dtype = torch.get_default_dtype()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_grad_enabled = torch.is_grad_enabled()
    expected_fd = [1, 2, 3, 4, 0, 4, 3, 2, 1]
    expected_ad = {0: list(expected_fd), 1: list(expected_fd)}
    values = [(index-8.5)/32.0 for index in range(18)]
    step, tolerance = 1.0/4096.0, 1e-9
    report.update(status='STARTED', scope='candidate coordinate map only; not a full DSConv/network gradient test',
        expected_candidate_source=dict(bytes=15206, sha256='321e9f23681d1da17ad2ecec008e4a58f64b66df4ad2bf78b68309a7fae820ea'),
        actual_source_identity_verified_by_probe=False,
        B=1, C=1, H=1, W=1, kernel_size=9, offset_channels=18,
        dtype='torch.float64', device='cpu', extend_scope=1.0, if_offset=True,
        offset_values=values, finite_difference='central', finite_difference_step=step,
        absolute_tolerance=tolerance, relative_tolerance=0.0,
        coordinate_map_calls=0, autograd_calls=0, cases={},
        Conv_or_BN_constructed=False, feature_interpolation_called=False,
        model_forward=False, data_read=False, optimizer_created=False, new_training=False, AP=False,
        CUDA_initialized_before=False, numerical_execution_completed=False)

    def close(left, right):
        return abs(left-right) <= tolerance

    def coordinates(dsc, offset, morph):
        y, x = dsc._coordinate_map_3D(offset, True)
        report['coordinate_map_calls'] += 1
        shape = (1, 9, 1) if morph == 0 else (1, 1, 9)
        if any(tuple(value.shape) != shape or value.device.type != 'cpu'
               or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()) for value in (y, x)):
            raise ValueError('Original coordinate API must return two finite CPU-double maps of the original shape')
        return y, x

    try:
        # DSC internally creates linspace grids without a dtype argument.
        # Its arange(...).float() zero centers promote against these double grids.
        torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(True)
        for morph in (0, 1):
            dsc = DSC((1, 1, 1, 1), 9, 1.0, morph)
            case = report['cases'].setdefault(str(morph), dict(
                morph=morph, objective='sum(y)' if morph == 0 else 'sum(x)', channels=[]))
            with torch.enable_grad():
                offsets = torch.tensor(values, dtype=torch.float64, device='cpu').reshape(1, 18, 1, 1)
                offsets.requires_grad_(True)
                y, x = coordinates(dsc, offsets, morph)
                objective = (y if morph == 0 else x).sum()
                derivative = torch.autograd.grad(objective, offsets, create_graph=False, retain_graph=False)[0]
                report['autograd_calls'] += 1
                if not bool(torch.isfinite(derivative).all()):
                    raise ValueError('Original AD returned nonfinite entries')
                case.update(base_objective=float(objective.detach().item()),
                    base_y=y.detach().reshape(-1).tolist(), base_x=x.detach().reshape(-1).tolist(),
                    AD=derivative.detach().reshape(-1).tolist())
                del offsets, y, x, objective, derivative
            with torch.no_grad():
                for channel in range(18):
                    plus = torch.tensor(values, dtype=torch.float64, device='cpu').reshape(1, 18, 1, 1)
                    minus = plus.clone()
                    plus[0, channel, 0, 0] += step
                    minus[0, channel, 0, 0] -= step
                    py, px = coordinates(dsc, plus, morph)
                    my, mx = coordinates(dsc, minus, morph)
                    f_plus = float((py if morph == 0 else px).sum().item())
                    f_minus = float((my if morph == 0 else mx).sum().item())
                    fd = (f_plus-f_minus)/(2.0*step)
                    active = channel < 9 if morph == 0 else channel >= 9
                    knot = channel % 9
                    ad_expected = expected_ad[morph][knot] if active else 0
                    fd_expected = expected_fd[knot] if active else 0
                    ad = case['AD'][channel]
                    case['channels'].append(dict(channel=channel, offset_axis='y' if channel < 9 else 'x',
                        knot_index=knot, AD=ad, FD=fd, objective_plus=f_plus, objective_minus=f_minus,
                        expected_candidate_AD=ad_expected, expected_candidate_FD=fd_expected,
                        AD_matches_candidate_expected=close(ad, ad_expected),
                        FD_matches_candidate_expected=close(fd, fd_expected),
                        AD_matches_FD=close(ad, fd), absolute_AD_FD_difference=abs(ad-fd)))
                    del plus, minus, py, px, my, mx
            case['FD'] = [row['FD'] for row in case['channels']]
            active_slice = slice(0, 9) if morph == 0 else slice(9, 18)
            case.update(active_AD=case['AD'][active_slice], active_FD=case['FD'][active_slice],
                candidate_expected_active_AD=list(expected_ad[morph]), candidate_expected_active_FD=list(expected_fd),
                AD_FD_disagreement_channels=[row['channel'] for row in case['channels'] if not row['AD_matches_FD']],
                candidate_expected_pattern_observed=all(row['AD_matches_candidate_expected'] and row['FD_matches_candidate_expected']
                                                       for row in case['channels']))
            del dsc
        zero, one = report['cases']['0'], report['cases']['1']
        report.update(active_AD_morph0_minus_morph1=[a-b for a, b in zip(zero['active_AD'], one['active_AD'])],
            active_FD_morph0_minus_morph1=[a-b for a, b in zip(zero['active_FD'], one['active_FD'])],
            candidate_expected_pattern_observed=all(case['candidate_expected_pattern_observed'] for case in report['cases'].values()),
            coordinate_AD_FD_mismatch_observed=any(case['AD_FD_disagreement_channels'] for case in report['cases'].values()),
            numerical_execution_completed=True)
        report['status'] = ('COMPLETED_ORIGINAL_AD_FD_MISMATCH_OBSERVED'
                            if report['coordinate_AD_FD_mismatch_observed'] else 'COMPLETED_NO_AD_FD_MISMATCH_OBSERVED')
    except BaseException as error:
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        torch.set_default_dtype(original_dtype)
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)
        report['CUDA_initialized_after'] = bool(torch.cuda.is_initialized())
        report['caller_controls_restored'] = (torch.get_default_dtype() == original_dtype
            and torch.are_deterministic_algorithms_enabled() == original_deterministic
            and torch.is_deterministic_algorithms_warn_only_enabled() == original_warn_only
            and torch.is_grad_enabled() == original_grad_enabled)
        if report['CUDA_initialized_after'] or not report['caller_controls_restored']:
            report['status'] = 'FAILED'
            raise RuntimeError('CPU-only/caller-control contract was not preserved')
    return report
