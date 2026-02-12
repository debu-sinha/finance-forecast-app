
import { DataRow, ForecastResult, ModelRunResult, BatchTrainingResult, BatchTrainingSummary, DataAnalysisResult } from "../types";
import { logFunctionIO, logSyncFunctionIO, logger } from "../utils/logger";

// When running as a Databricks App, the backend is on the same host/port usually,
// or proxied. For local dev, you might need a proxy setup.
const API_BASE = "/api";

/**
 * Analyze training data and get intelligent model/hyperparameter recommendations
 */
const _analyzeTrainingData = async (
    data: DataRow[],
    timeCol: string,
    targetCol: string,
    frequency: string = 'auto'
): Promise<DataAnalysisResult> => {
    const payload = {
        data,
        time_col: timeCol,
        target_col: targetCol,
        frequency
    };

    const response = await fetch(`${API_BASE}/analyze-data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = "Data analysis failed";
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch (e) {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }

    return response.json();
};
export const analyzeTrainingData = logFunctionIO('analyzeTrainingData', _analyzeTrainingData);

const _trainModelOnBackend = async (
    data: DataRow[],
    timeCol: string,
    targetCol: string,
    covariates: string[],
    horizon: number,
    frequency: string,
    seasonalityMode: string,
    regressorMethod: string = 'mean',
    models: string[] = ['prophet'],
    catalogName: string = 'main',
    schemaName: string = 'default',
    modelName: string = 'finance_forecast_model',
    country: string = 'US',
    filters?: Record<string, any>,
    fromDate?: string,
    toDate?: string,
    randomSeed?: number,
    futureFeatures?: DataRow[],
    hyperparameterFilters?: Record<string, Record<string, any>>
): Promise<any> => {

    const payload = {
        data,
        time_col: timeCol,
        target_col: targetCol,
        covariates,
        horizon,
        frequency,
        seasonality_mode: seasonalityMode,
        regressor_method: regressorMethod,
        models: models,
        catalog_name: catalogName,
        schema_name: schemaName,
        model_name: modelName,
        country: country,
        filters: filters || null,
        from_date: fromDate || null,
        to_date: toDate || null,
        random_seed: randomSeed || 42,
        future_features: futureFeatures || null,
        hyperparameter_filters: hyperparameterFilters || null
    };

    const response = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = "Training failed";
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch (e) {
            // If not JSON, use the text (likely upstream timeout/error)
            errDetail = text || `Error ${response.status}: ${response.statusText}`;
            if (errDetail.includes("upstream") || response.status === 504 || response.status === 502) {
                errDetail = "Training timed out (Server/Proxy limit exceeded). Try reducing the number of models or data size.";
            }
        }
        throw new Error(errDetail);
    }

    return await response.json();
};
export const trainModelOnBackend = logFunctionIO('trainModelOnBackend', _trainModelOnBackend);

const _deployModel = async (modelName: string, version: string | null, endpointName: string, runId?: string) => {
    const response = await fetch(`${API_BASE}/deploy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            model_name: modelName,
            model_version: (version === "latest" || !version) ? null : version,
            run_id: runId,
            endpoint_name: endpointName
        })
    });
    return await response.json();
};
export const deployModel = logFunctionIO('deployModel', _deployModel);

export interface ActualsComparisonSummary {
    overallMAPE: number;
    overallRMSE: number;
    overallBias: number;
    excellentCount: number;
    goodCount: number;
    acceptableCount: number;
    reviewCount: number;
    deviationCount: number;
    totalPeriods: number;
    worstPeriods: Array<{ date: string; mape: number; error: number; predicted: number; actual: number }>;
}

const _generateExecutiveSummary = async (
    bestModelName: string,
    bestModelMetrics: { rmse: number; mape: number; r2: number },
    allModels: Array<{ modelName: string; metrics: { rmse: string; mape: string; r2: string } }>,
    targetCol: string,
    timeCol: string,
    covariates: string[],
    forecastHorizon: number,
    frequency: string,
    actualsComparison?: ActualsComparisonSummary
): Promise<string> => {
    const response = await fetch(`${API_BASE}/executive-summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            bestModelName,
            bestModelMetrics,
            allModels,
            targetCol,
            timeCol,
            covariates,
            forecastHorizon,
            frequency,
            actualsComparison
        })
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = "Executive summary generation failed";
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch (e) {
            errDetail = text || `Error ${response.status}`;
        }
        throw new Error(errDetail);
    }

    const data = await response.json();
    return data.summary || "";
};
export const generateExecutiveSummary = logFunctionIO('generateExecutiveSummary', _generateExecutiveSummary);

// ==========================================
// BATCH TRAINING API
// ==========================================

export interface BatchTrainRequest {
    data: DataRow[];
    timeCol: string;
    targetCol: string;
    covariates: string[];
    horizon: number;
    frequency: string;
    seasonalityMode: string;
    regressorMethod: string;
    models: string[];
    filters: Record<string, string | number>;
    catalogName?: string;
    schemaName?: string;
    modelName?: string;
    country?: string;
    randomSeed?: number;
    // Batch context for MLflow tracking
    batchId?: string;
    batchSegmentId?: string;
    batchSegmentIndex?: number;
    batchTotalSegments?: number;
}

const _trainBatchOnBackend = async (
    requests: BatchTrainRequest[],
    maxWorkers: number = 4,
    onProgress?: (completed: number, total: number, latestResult?: any) => void,
    signal?: AbortSignal
): Promise<BatchTrainingSummary> => {
    const startTime = Date.now();

    // Generate a unique batch ID for MLflow tracking
    const batchId = `${new Date().toISOString().split('T')[0]}_${Date.now().toString(36)}`;
    const totalSegments = requests.length;

    logger.debug(`Starting batch training: ${totalSegments} segments, batch_id=${batchId}, max_workers=${maxWorkers}`);

    // Prepare the payload for the parallel backend endpoint
    const trainRequests = requests.map((req, i) => {
        const segmentId = Object.entries(req.filters)
            .map(([k, v]) => `${k}=${v}`)
            .join(' | ');

        return {
            data: req.data,
            time_col: req.timeCol,
            target_col: req.targetCol,
            covariates: req.covariates,
            horizon: req.horizon,
            frequency: req.frequency,
            seasonality_mode: req.seasonalityMode,
            regressor_method: req.regressorMethod,
            models: req.models,
            catalog_name: req.catalogName || 'main',
            schema_name: req.schemaName || 'default',
            model_name: req.modelName || 'finance_forecast_model',
            country: req.country || 'US',
            filters: req.filters,
            random_seed: req.randomSeed || 42,
            // Batch context for MLflow tracking
            batch_id: batchId,
            batch_segment_id: segmentId,
            batch_segment_index: i + 1,
            batch_total_segments: totalSegments
        };
    });

    const payload = {
        requests: trainRequests,
        max_workers: maxWorkers
    };

    try {
        const response = await fetch(`${API_BASE}/train-batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: signal
        });

        if (!response.ok) {
            const text = await response.text();
            let errDetail = "Batch training failed";
            try {
                const json = JSON.parse(text);
                errDetail = json.detail || errDetail;
            } catch (e) {
                errDetail = text || `Error ${response.status}`;
            }
            throw new Error(errDetail);
        }

        const backendResponse = await response.json();

        // Transform backend results to our internal format
        const results: BatchTrainingResult[] = backendResponse.results.map((item: any) => {
            if (item.status === 'error') {
                return {
                    segmentId: item.segment_id || 'unknown',
                    filters: item.filters || {},
                    status: 'error',
                    error: item.error || 'Unknown error'
                };
            }

            // Success case
            const trainRes = item.result;
            const bestModel = trainRes.models?.find((m: any) => m.is_best) || trainRes.models?.[0];

            // Transform to ForecastResult format
            const forecastResult: any = {
                history: trainRes.history || [],  // Use history from backend response
                results: trainRes.models?.map((m: any) => ({
                    modelType: m.model_type,
                    modelName: m.model_name,
                    isBest: m.is_best,
                    metrics: m.metrics,
                    hyperparameters: m.hyperparameters || {},
                    validation: m.validation || [],
                    forecast: m.forecast || [],
                    experimentUrl: m.experiment_url,
                    runUrl: m.run_url
                })) || [],
                explanation: '',
                pythonCode: ''
            };

            return {
                segmentId: item.segment_id,
                filters: item.filters,
                status: 'success',
                result: forecastResult,
                bestModel: bestModel?.model_name || trainRes.best_model,
                metrics: bestModel ? {
                    mape: bestModel.metrics.mape,
                    rmse: bestModel.metrics.rmse,
                    r2: bestModel.metrics.r2,
                    cv_mape: bestModel.metrics.cv_mape
                } : undefined,
                runId: bestModel?.run_id,
                experimentUrl: bestModel?.experiment_url
            };
        });

        // Calculate MAPE stats
        const mapes = results
            .filter(r => r.status === 'success' && r.metrics?.mape)
            .map(r => parseFloat(r.metrics!.mape));

        let mapeStats = undefined;
        if (mapes.length > 0) {
            const sorted = [...mapes].sort((a, b) => a - b);
            mapeStats = {
                min: Math.min(...mapes),
                max: Math.max(...mapes),
                mean: mapes.reduce((a, b) => a + b, 0) / mapes.length,
                median: sorted[Math.floor(sorted.length / 2)]
            };
        }

        logger.debug(`Batch training complete: ${backendResponse.successful}/${backendResponse.total_requests} successful`);

        return {
            totalSegments: backendResponse.total_requests,
            successful: backendResponse.successful,
            failed: backendResponse.failed,
            results,
            startTime,
            endTime: Date.now(),
            mapeStats,
            batchId,
            experimentName: `finance-forecasting-batch-${batchId}`
        };

    } catch (error: any) {
        if (error.name === 'AbortError') {
            logger.debug('Batch training cancelled by user');
            throw error;
        }
        logger.error('Batch training error:', error);
        throw error;
    }
};
export const trainBatchOnBackend = logFunctionIO('trainBatchOnBackend', _trainBatchOnBackend);

// ==========================================
// BATCH DEPLOYMENT API
// ==========================================

export interface BatchDeployRequest {
    segments: Array<{
        segmentId: string;
        filters: Record<string, any>;
        modelVersion: string;
        runId?: string;
    }>;
    endpointName: string;
    catalogName: string;
    schemaName: string;
    modelName: string;
}

export interface BatchDeployResponse {
    status: 'success' | 'error';
    message: string;
    endpointName?: string;
    endpointUrl?: string;
    deployedSegments?: number;
    routerModelVersion?: string;
}

const _deployBatchModels = async (
    batchResults: BatchTrainingSummary,
    endpointName: string,
    catalogName: string = 'main',
    schemaName: string = 'default',
    modelName: string = 'finance_forecast_model'
): Promise<BatchDeployResponse> => {
    // Collect all successful segments with their model versions
    const segments = batchResults.results
        .filter(r => r.status === 'success' && r.result)
        .map(r => {
            // Find the best model's registered version
            const bestModel = r.result?.results?.find(m => m.isBest) || r.result?.results?.[0];
            return {
                segmentId: r.segmentId,
                filters: r.filters,
                modelVersion: (bestModel as any)?.registeredVersion || 'latest',
                runId: r.runId
            };
        });

    if (segments.length === 0) {
        return {
            status: 'error',
            message: 'No successful models to deploy'
        };
    }

    // Convert to snake_case for backend API
    const payload = {
        segments: segments.map(s => ({
            segment_id: s.segmentId,
            filters: s.filters,
            model_version: s.modelVersion,
            run_id: s.runId
        })),
        endpoint_name: endpointName,
        catalog_name: catalogName,
        schema_name: schemaName,
        model_name: modelName
    };

    const response = await fetch(`${API_BASE}/deploy-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = "Batch deployment failed";
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch (e) {
            errDetail = text || `Error ${response.status}`;
        }
        return {
            status: 'error',
            message: errDetail
        };
    }

    return await response.json();
};
export const deployBatchModels = logFunctionIO('deployBatchModels', _deployBatchModels);

// ==========================================
// SIMPLE MODE API (Autopilot for Finance Users)
// ==========================================

export interface SimpleProfileResponse {
    success: boolean;
    profile: {
        frequency: string;
        date_column: string;
        target_column: string;
        date_range: [string, string];
        total_periods: number;
        history_months: number;
        data_quality_score: number;
        holiday_coverage_score: number;
        has_trend: boolean;
        has_seasonality: boolean;
        covariate_columns: string[];
        recommended_models: string[];
        recommended_horizon: number;
        data_hash: string;
        row_count: number;
    };
    warnings: Array<{ level: string; message: string; recommendation: string }>;
    config_preview: any;
}

export interface SimpleForecastResponse {
    success: boolean;
    mode: string;
    run_id: string;
    summary: string;
    forecast: number[];
    dates: string[];
    lower_bounds: number[];
    upper_bounds: number[];
    components: {
        formula: string;
        totals: { base: number; trend: number; seasonal: number; holiday: number };
        periods: Array<{
            date: string;
            forecast: number;
            lower: number;
            upper: number;
            base: number;
            trend: number;
            seasonal: number;
            holiday: number;
            explanation: string;
        }>;
    };
    confidence: {
        level: string;
        score: number;
        mape: number;
        factors: Array<{ factor: string; score: number; note: string }>;
        explanation: string;
    };
    warnings: Array<{ level: string; message: string; recommendation: string }>;
    caveats: string[];
    audit: {
        run_id: string;
        timestamp: string;
        data_hash: string;
        config_hash: string;
        model: string;
        reproducibility_token: string;
    };
    excel_download_url: string;
}

const _profileDataForSimpleMode = async (file: File): Promise<SimpleProfileResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/simple/profile`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to profile data';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch (e) {
            errDetail = text || `Error ${response.status}`;
        }
        throw new Error(errDetail);
    }

    return await response.json();
};
export const profileDataForSimpleMode = logFunctionIO('profileDataForSimpleMode', _profileDataForSimpleMode);

const _runSimpleForecast = async (
    file: File,
    horizon?: number
): Promise<SimpleForecastResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (horizon) {
        formData.append('horizon', String(horizon));
    }

    const response = await fetch(`${API_BASE}/simple/forecast`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Forecast failed';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch (e) {
            errDetail = text || `Error ${response.status}`;
        }
        throw new Error(errDetail);
    }

    return await response.json();
};
export const runSimpleForecast = logFunctionIO('runSimpleForecast', _runSimpleForecast);

const _downloadSimpleForecastExcel = async (runId: string): Promise<Blob> => {
    const response = await fetch(`${API_BASE}/simple/export/${runId}/excel`);

    if (!response.ok) {
        throw new Error('Failed to download Excel file');
    }

    return await response.blob();
};
export const downloadSimpleForecastExcel = logFunctionIO('downloadSimpleForecastExcel', _downloadSimpleForecastExcel);

const _downloadSimpleForecastCSV = async (runId: string): Promise<Blob> => {
    const response = await fetch(`${API_BASE}/simple/export/${runId}/csv`);

    if (!response.ok) {
        throw new Error('Failed to download CSV file');
    }

    return await response.blob();
};
export const downloadSimpleForecastCSV = logFunctionIO('downloadSimpleForecastCSV', _downloadSimpleForecastCSV);

// Export batch results to CSV
const _exportBatchResultsToCSV = (summary: BatchTrainingSummary, segmentCols: string[]): string => {
    const headers = [
        ...segmentCols,
        'status',
        'best_model',
        'mape',
        'rmse',
        'r2',
        'cv_mape',
        'run_id',
        'error'
    ];

    const rows = summary.results.map(r => {
        const segmentValues = segmentCols.map(col => r.filters[col] || '');
        return [
            ...segmentValues,
            r.status,
            r.bestModel || '',
            r.metrics?.mape || '',
            r.metrics?.rmse || '',
            r.metrics?.r2 || '',
            r.metrics?.cv_mape || '',
            r.runId || '',
            r.error || ''
        ].map(v => `"${v}"`).join(',');
    });

    return [headers.join(','), ...rows].join('\n');
};
export const exportBatchResultsToCSV = logSyncFunctionIO('exportBatchResultsToCSV', _exportBatchResultsToCSV);
