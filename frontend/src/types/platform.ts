export type IncidentCategory =
  | 'collision_multi_vehicle'
  | 'collision_single_vehicle'
  | 'road_obstruction_debris'
  | 'disabled_stalled_vehicle'
  | 'non_incident_congestion';

export type SeverityLevel = 'low' | 'moderate' | 'high' | 'critical';

export interface BoundingBox {
  bbox: [number, number, number, number];
  confidence: number;
  label: string;
  area_norm: number;
}

export interface RecoveryPredictionResponse {
  prediction_id: string;
  incident_id: string;
  multimodal_classification: {
    fusion_method: string;
    visual_modality_present: boolean;
    missing_modalities: string[];
    incident_category: {
      predicted_class: IncidentCategory;
      confidence: number;
      normalized_entropy_uncertainty: number;
      probabilities: Record<IncidentCategory, number>;
    };
    severity: {
      predicted_severity: SeverityLevel;
      confidence: number;
      normalized_entropy_uncertainty: number;
      probabilities: Record<SeverityLevel, number>;
      limitation_notice: string;
    };
  };
  recovery_prediction: {
    model_version: string;
    target_distinction_notice: string;
    incident_clearance_prediction: {
      predicted_minutes: number;
      prediction_interval_80_pct: [number, number];
      category_median_baseline_minutes: number;
      estimated_clearance_timestamp: string | null;
    };
    road_network_recovery_prediction: {
      predicted_minutes: number;
      queue_dissipation_lag_minutes: number;
      prediction_interval_80_pct: [number, number];
      category_median_baseline_minutes: number;
      estimated_restoration_timestamp: string | null;
    };
    top_contributing_factors: Array<{
      factor: string;
      value: string;
      impact_direction: string;
      estimated_minute_contribution: number;
    }>;
  };
  road_network_impact: Record<string, unknown>;
}
