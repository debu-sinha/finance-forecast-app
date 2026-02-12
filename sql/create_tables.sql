-- Lakebase PostgreSQL Schema for Finance Forecasting Platform
-- Multi-user architecture with session management, job tracking, and reproducibility
--
-- Lakebase: Databricks' fully-managed PostgreSQL (based on Neon)
-- Features: <10ms latency, >10K QPS, scale-to-zero, pgvector support

-- Create schema
CREATE SCHEMA IF NOT EXISTS forecast;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- 1. SESSIONS TABLE - User Session State
-- =============================================================================
-- Tracks active user sessions with preferences and activity
-- Optimized for high-frequency reads (session validation) and periodic updates

CREATE TABLE IF NOT EXISTS forecast.sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    user_email VARCHAR(255),

    -- Session lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours'),
    is_active BOOLEAN DEFAULT TRUE,

    -- Session configuration (UI preferences, last used settings)
    session_config JSONB DEFAULT '{}',

    -- Request tracking
    request_count INTEGER DEFAULT 0,
    last_request_id UUID,

    -- Constraints
    CONSTRAINT sessions_user_id_not_empty CHECK (user_id <> '')
);

-- Indexes for session lookups
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON forecast.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON forecast.sessions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON forecast.sessions(expires_at) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON forecast.sessions(last_active_at DESC);

-- =============================================================================
-- 2. EXECUTION HISTORY TABLE - Job Tracking & Reproducibility
-- =============================================================================
-- Full audit trail for every training job with complete parameter capture
-- Enables exact reproduction of any previous forecast

CREATE TYPE forecast.job_status AS ENUM (
    'PENDING',
    'QUEUED',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TABLE IF NOT EXISTS forecast.execution_history (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES forecast.sessions(session_id),
    user_id VARCHAR(255) NOT NULL,

    -- Databricks Job Reference
    databricks_run_id BIGINT,
    databricks_job_id BIGINT,
    cluster_id VARCHAR(255),

    -- Request Parameters (Full Reproducibility)
    request_params JSONB NOT NULL,  -- Complete TrainRequest serialized
    time_col VARCHAR(255) NOT NULL,
    target_col VARCHAR(255) NOT NULL,
    horizon INTEGER NOT NULL,
    frequency VARCHAR(50) NOT NULL,
    models TEXT[] NOT NULL,
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    random_seed INTEGER DEFAULT 42,
    hyperparameter_filters JSONB DEFAULT '{}',

    -- Data Reference
    data_upload_id UUID,
    data_row_count INTEGER,
    data_hash VARCHAR(64),  -- SHA-256 for data versioning
    data_columns TEXT[],
    data_date_range_start TIMESTAMP WITH TIME ZONE,
    data_date_range_end TIMESTAMP WITH TIME ZONE,

    -- Execution State
    status forecast.job_status DEFAULT 'PENDING',
    progress_percent INTEGER DEFAULT 0,
    current_step VARCHAR(255),
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    queued_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,

    -- Error Handling
    error_message TEXT,
    error_traceback TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Results Reference
    mlflow_experiment_id VARCHAR(255),
    mlflow_run_id VARCHAR(255),
    mlflow_experiment_url TEXT,
    mlflow_run_url TEXT,

    -- Best Model Summary
    best_model VARCHAR(100),
    best_mape DOUBLE PRECISION,
    best_rmse DOUBLE PRECISION,
    models_trained INTEGER,
    models_failed INTEGER,

    -- Metadata
    app_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT execution_valid_horizon CHECK (horizon > 0),
    CONSTRAINT execution_valid_confidence CHECK (confidence_level > 0 AND confidence_level < 1),
    CONSTRAINT execution_valid_progress CHECK (progress_percent >= 0 AND progress_percent <= 100)
);

-- Indexes for execution history queries
CREATE INDEX IF NOT EXISTS idx_execution_user_id ON forecast.execution_history(user_id);
CREATE INDEX IF NOT EXISTS idx_execution_session_id ON forecast.execution_history(session_id);
CREATE INDEX IF NOT EXISTS idx_execution_status ON forecast.execution_history(status);
CREATE INDEX IF NOT EXISTS idx_execution_submitted ON forecast.execution_history(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_databricks_run ON forecast.execution_history(databricks_run_id) WHERE databricks_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_execution_mlflow_run ON forecast.execution_history(mlflow_run_id) WHERE mlflow_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_execution_data_hash ON forecast.execution_history(data_hash);

-- Partial index for active jobs (common query pattern)
CREATE INDEX IF NOT EXISTS idx_execution_active_jobs ON forecast.execution_history(user_id, status)
    WHERE status IN ('PENDING', 'QUEUED', 'RUNNING');

-- =============================================================================
-- 3. FORECAST RESULTS TABLE - Model Predictions Storage
-- =============================================================================
-- Stores forecast outputs for each model in a job
-- Enables comparison and ensemble creation

CREATE TABLE IF NOT EXISTS forecast.forecast_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES forecast.execution_history(job_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,

    -- Forecast Data (stored as JSONB arrays for flexibility)
    forecast_dates JSONB NOT NULL,  -- Array of ISO timestamps
    predictions JSONB NOT NULL,      -- Array of predicted values
    lower_bounds JSONB,              -- Array of lower CI bounds
    upper_bounds JSONB,              -- Array of upper CI bounds

    -- Validation Data
    validation_dates JSONB,
    validation_actuals JSONB,
    validation_predictions JSONB,

    -- Metrics
    mape DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    cv_mape DOUBLE PRECISION,        -- Cross-validation MAPE
    cv_mape_std DOUBLE PRECISION,    -- CV standard deviation

    -- Model Details
    model_params JSONB,              -- Fitted hyperparameters
    feature_importance JSONB,        -- For models that support it
    training_time_seconds DOUBLE PRECISION,

    -- MLflow Reference
    mlflow_run_id VARCHAR(255),
    mlflow_model_uri TEXT,

    -- Ranking
    rank_by_mape INTEGER,
    is_best_model BOOLEAN DEFAULT FALSE,
    is_ensemble_member BOOLEAN DEFAULT FALSE,
    ensemble_weight DOUBLE PRECISION,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT results_unique_model_per_job UNIQUE (job_id, model_name),
    CONSTRAINT results_valid_mape CHECK (mape IS NULL OR mape >= 0),
    CONSTRAINT results_valid_ensemble_weight CHECK (ensemble_weight IS NULL OR (ensemble_weight >= 0 AND ensemble_weight <= 1))
);

-- Indexes for forecast results queries
CREATE INDEX IF NOT EXISTS idx_results_job_id ON forecast.forecast_results(job_id);
CREATE INDEX IF NOT EXISTS idx_results_model_name ON forecast.forecast_results(model_name);
CREATE INDEX IF NOT EXISTS idx_results_mape ON forecast.forecast_results(mape) WHERE mape IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_results_best_model ON forecast.forecast_results(job_id) WHERE is_best_model = TRUE;

-- =============================================================================
-- 4. USER UPLOADS TABLE - Data Versioning
-- =============================================================================
-- Tracks uploaded datasets for reproducibility and data lineage

CREATE TABLE IF NOT EXISTS forecast.user_uploads (
    upload_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_id UUID REFERENCES forecast.sessions(session_id),

    -- Data Location (Unity Catalog Volume or temporary storage)
    storage_path TEXT,               -- /Volumes/forecast/uploads/{upload_id}/
    file_name VARCHAR(255),
    file_size_bytes BIGINT,
    content_type VARCHAR(100),

    -- Schema Info
    columns JSONB,                   -- [{name, dtype, nullable}]
    row_count INTEGER,
    date_range_start TIMESTAMP WITH TIME ZONE,
    date_range_end TIMESTAMP WITH TIME ZONE,

    -- Detected Metadata
    detected_time_col VARCHAR(255),
    detected_target_col VARCHAR(255),
    detected_frequency VARCHAR(50),

    -- Data Quality Profile
    profile_json JSONB,              -- DataProfile serialized
    quality_score DOUBLE PRECISION,
    warnings JSONB,                  -- Array of warning messages

    -- Hash for Deduplication
    data_hash VARCHAR(64),           -- SHA-256 of content

    -- Metadata
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
    is_deleted BOOLEAN DEFAULT FALSE,

    -- Constraints
    CONSTRAINT uploads_valid_size CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
    CONSTRAINT uploads_valid_rows CHECK (row_count IS NULL OR row_count >= 0)
);

-- Indexes for user uploads queries
CREATE INDEX IF NOT EXISTS idx_uploads_user_id ON forecast.user_uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_uploads_session_id ON forecast.user_uploads(session_id);
CREATE INDEX IF NOT EXISTS idx_uploads_hash ON forecast.user_uploads(data_hash);
CREATE INDEX IF NOT EXISTS idx_uploads_expires ON forecast.user_uploads(expires_at) WHERE is_deleted = FALSE;

-- =============================================================================
-- 5. JOB QUEUE TABLE - Distributed Job Management
-- =============================================================================
-- Manages job queue for fair scheduling across users

CREATE TYPE forecast.queue_priority AS ENUM ('LOW', 'NORMAL', 'HIGH', 'URGENT');

CREATE TABLE IF NOT EXISTS forecast.job_queue (
    queue_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES forecast.execution_history(job_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,

    -- Queue Management
    priority forecast.queue_priority DEFAULT 'NORMAL',
    position INTEGER,                -- Queue position (computed)

    -- Timing
    enqueued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    dequeued_at TIMESTAMP WITH TIME ZONE,
    estimated_start_at TIMESTAMP WITH TIME ZONE,

    -- Resource Requirements
    estimated_duration_seconds INTEGER,
    required_memory_gb INTEGER DEFAULT 64,
    required_cpus INTEGER DEFAULT 16,

    -- State
    is_active BOOLEAN DEFAULT TRUE,

    -- Constraints
    CONSTRAINT queue_unique_job UNIQUE (job_id)
);

-- Indexes for job queue
CREATE INDEX IF NOT EXISTS idx_queue_active ON forecast.job_queue(priority DESC, enqueued_at ASC) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_queue_user ON forecast.job_queue(user_id) WHERE is_active = TRUE;

-- =============================================================================
-- 6. AUDIT LOG TABLE - Compliance and Debugging
-- =============================================================================
-- Immutable audit trail for all significant operations

CREATE TABLE IF NOT EXISTS forecast.audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Actor
    user_id VARCHAR(255),
    session_id UUID,

    -- Action
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,

    -- Details
    request_data JSONB,
    response_data JSONB,
    error_data JSONB,

    -- Context
    ip_address INET,
    user_agent TEXT,
    app_version VARCHAR(50),

    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    duration_ms INTEGER
);

-- Indexes for audit log (time-series optimized)
CREATE INDEX IF NOT EXISTS idx_audit_created ON forecast.audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user ON forecast.audit_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON forecast.audit_log(action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON forecast.audit_log(resource_type, resource_id);

-- =============================================================================
-- FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION forecast.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for execution_history updated_at
DROP TRIGGER IF EXISTS execution_history_updated_at ON forecast.execution_history;
CREATE TRIGGER execution_history_updated_at
    BEFORE UPDATE ON forecast.execution_history
    FOR EACH ROW
    EXECUTE FUNCTION forecast.update_updated_at();

-- Function to calculate job duration
CREATE OR REPLACE FUNCTION forecast.calculate_job_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('COMPLETED', 'FAILED', 'CANCELLED') AND NEW.started_at IS NOT NULL THEN
        NEW.duration_seconds = EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at))::INTEGER;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for job duration calculation
DROP TRIGGER IF EXISTS execution_calculate_duration ON forecast.execution_history;
CREATE TRIGGER execution_calculate_duration
    BEFORE UPDATE ON forecast.execution_history
    FOR EACH ROW
    WHEN (NEW.status IN ('COMPLETED', 'FAILED', 'CANCELLED'))
    EXECUTE FUNCTION forecast.calculate_job_duration();

-- Function to rank models by MAPE within a job
CREATE OR REPLACE FUNCTION forecast.update_model_rankings(p_job_id UUID)
RETURNS VOID AS $$
BEGIN
    -- Update rankings
    WITH ranked AS (
        SELECT
            result_id,
            RANK() OVER (ORDER BY mape ASC NULLS LAST) as rank_num,
            RANK() OVER (ORDER BY mape ASC NULLS LAST) = 1 as is_best
        FROM forecast.forecast_results
        WHERE job_id = p_job_id AND mape IS NOT NULL
    )
    UPDATE forecast.forecast_results fr
    SET
        rank_by_mape = r.rank_num,
        is_best_model = r.is_best
    FROM ranked r
    WHERE fr.result_id = r.result_id;

    -- Update execution_history with best model info
    UPDATE forecast.execution_history eh
    SET
        best_model = fr.model_name,
        best_mape = fr.mape,
        best_rmse = fr.rmse,
        models_trained = (SELECT COUNT(*) FROM forecast.forecast_results WHERE job_id = p_job_id)
    FROM forecast.forecast_results fr
    WHERE eh.job_id = p_job_id
      AND fr.job_id = p_job_id
      AND fr.is_best_model = TRUE;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View: Active jobs with queue position
CREATE OR REPLACE VIEW forecast.v_active_jobs AS
SELECT
    eh.job_id,
    eh.user_id,
    eh.status,
    eh.progress_percent,
    eh.current_step,
    eh.submitted_at,
    eh.started_at,
    eh.models,
    eh.horizon,
    eh.frequency,
    jq.priority,
    jq.position AS queue_position,
    jq.estimated_start_at
FROM forecast.execution_history eh
LEFT JOIN forecast.job_queue jq ON eh.job_id = jq.job_id
WHERE eh.status IN ('PENDING', 'QUEUED', 'RUNNING')
ORDER BY
    CASE eh.status
        WHEN 'RUNNING' THEN 1
        WHEN 'QUEUED' THEN 2
        ELSE 3
    END,
    jq.priority DESC,
    eh.submitted_at ASC;

-- View: User execution summary
CREATE OR REPLACE VIEW forecast.v_user_summary AS
SELECT
    user_id,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed_jobs,
    COUNT(*) FILTER (WHERE status = 'FAILED') as failed_jobs,
    AVG(best_mape) FILTER (WHERE status = 'COMPLETED') as avg_mape,
    MIN(best_mape) FILTER (WHERE status = 'COMPLETED') as best_mape,
    SUM(duration_seconds) FILTER (WHERE status = 'COMPLETED') as total_compute_seconds,
    MAX(submitted_at) as last_job_at
FROM forecast.execution_history
GROUP BY user_id;

-- View: Recent results with model details
CREATE OR REPLACE VIEW forecast.v_recent_results AS
SELECT
    eh.job_id,
    eh.user_id,
    eh.submitted_at,
    eh.completed_at,
    eh.horizon,
    eh.frequency,
    fr.model_name,
    fr.mape,
    fr.rmse,
    fr.cv_mape,
    fr.is_best_model,
    fr.training_time_seconds,
    eh.mlflow_run_url
FROM forecast.execution_history eh
JOIN forecast.forecast_results fr ON eh.job_id = fr.job_id
WHERE eh.status = 'COMPLETED'
ORDER BY eh.completed_at DESC, fr.rank_by_mape ASC;

-- =============================================================================
-- GRANTS (adjust based on your Lakebase user setup)
-- =============================================================================

-- Grant usage on schema
-- GRANT USAGE ON SCHEMA forecast TO forecast_app_user;

-- Grant table permissions
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA forecast TO forecast_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA forecast TO forecast_app_user;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON SCHEMA forecast IS 'Finance Forecasting Platform - Multi-user state management';
COMMENT ON TABLE forecast.sessions IS 'User session tracking with activity and preferences';
COMMENT ON TABLE forecast.execution_history IS 'Complete audit trail for training jobs with full reproducibility';
COMMENT ON TABLE forecast.forecast_results IS 'Model predictions and metrics for each training job';
COMMENT ON TABLE forecast.user_uploads IS 'Uploaded data files with schema detection and profiling';
COMMENT ON TABLE forecast.job_queue IS 'Distributed job queue for fair scheduling';
COMMENT ON TABLE forecast.audit_log IS 'Immutable audit trail for compliance and debugging';
