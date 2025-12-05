
import { DataRow, ForecastResult, ModelRunResult, BatchTrainingResult, BatchTrainingSummary } from "../types";

// When running as a Databricks App, the backend is on the same host/port usually, 
// or proxied. For local dev, you might need a proxy setup.
const API_BASE = "/api";

export const trainModelOnBackend = async (
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
    futureFeatures?: DataRow[]
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
        future_features: futureFeatures || null
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

export const deployModel = async (modelName: string, version: string | null, endpointName: string, runId?: string) => {
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

export const generateExecutiveSummary = async (
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

export const trainBatchOnBackend = async (
    requests: BatchTrainRequest[],
    maxWorkers: number = 2,
    onProgress?: (completed: number, total: number, latestResult?: any) => void
): Promise<BatchTrainingSummary> => {
    const startTime = Date.now();
    const results: BatchTrainingResult[] = [];
    let successful = 0;
    let failed = 0;

    // Generate a unique batch ID for MLflow tracking
    // This ensures all segments from this batch are grouped in the same experiment
    const batchId = `${new Date().toISOString().split('T')[0]}_${Date.now().toString(36)}`;
    const totalSegments = requests.length;

    console.log(`Starting batch training: ${totalSegments} segments, batch_id=${batchId}`);

    // Process segments sequentially in the frontend to show progress
    // The backend /api/train endpoint handles each segment
    for (let i = 0; i < requests.length; i++) {
        const req = requests[i];
        const segmentId = Object.entries(req.filters)
            .map(([k, v]) => `${k}=${v}`)
            .join(' | ');

        try {
            const payload = {
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
                    errDetail = text || `Error ${response.status}`;
                }
                throw new Error(errDetail);
            }

            const backendResponse = await response.json();

            // Find best model from response
            const bestModel = backendResponse.models?.find((m: any) => m.is_best) || backendResponse.models?.[0];

            results.push({
                segmentId,
                filters: req.filters,
                status: 'success',
                bestModel: bestModel?.model_name || backendResponse.best_model,
                metrics: bestModel ? {
                    mape: bestModel.metrics.mape,
                    rmse: bestModel.metrics.rmse,
                    r2: bestModel.metrics.r2,
                    cv_mape: bestModel.metrics.cv_mape
                } : undefined,
                runId: bestModel?.run_id
            });
            successful++;

            if (onProgress) {
                onProgress(i + 1, requests.length, results[results.length - 1]);
            }

        } catch (error: any) {
            results.push({
                segmentId,
                filters: req.filters,
                status: 'error',
                error: error.message || 'Unknown error'
            });
            failed++;

            if (onProgress) {
                onProgress(i + 1, requests.length, results[results.length - 1]);
            }
        }
    }

    // Calculate MAPE statistics
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

    console.log(`Batch training complete: ${successful}/${totalSegments} successful, batch_id=${batchId}`);

    return {
        totalSegments: requests.length,
        successful,
        failed,
        results,
        startTime,
        endTime: Date.now(),
        mapeStats,
        batchId,
        experimentName: `finance-forecasting-batch-${batchId}`
    };
};

// Export batch results to CSV
export const exportBatchResultsToCSV = (summary: BatchTrainingSummary, segmentCols: string[]): string => {
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
