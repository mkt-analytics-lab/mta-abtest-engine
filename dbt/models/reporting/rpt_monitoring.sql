-- Model health page (MLOps): drift, calibration and the recommended action.
select
    monitor_date,
    model_version,
    n_scored,
    share_drifted_features,
    psi_propensity,
    channel_mix_tvd,
    channel_max_change,
    matured_positives,
    calibration_ratio,
    action,
    cast(alert_feature_drift as int64) as alert_feature_drift,
    cast(alert_prediction_drift as int64) as alert_prediction_drift,
    cast(alert_calibration as int64) as alert_calibration,
    cast(alert_channel_mix as int64) as alert_channel_mix
from {{ source('results', 'monitoring_daily') }}
