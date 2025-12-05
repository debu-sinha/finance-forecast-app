import { DataRow, DatasetAnalysis, ForecastResult, ModelType, FutureRegressorMethod } from "../types";

// Use backend proxy for analysis calls (avoids CORS issues)
const API_BASE = "/api";

/**
 * Call backend analysis endpoint (avoids CORS issues)
 */
async function callBackendAnalysis(
  sampleData: DataRow[],
  columns: string[]
): Promise<DatasetAnalysis> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      sample_data: sampleData,
      columns: columns
    })
  });

  if (!response.ok) {
    throw new Error(`Backend analysis failed: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Call backend insights endpoint
 */
async function callBackendInsights(
  dataSummary: string,
  targetCol: string,
  timeCol: string,
  covariates: string[],
  filters: Record<string, string>,
  seasonalityMode: string,
  winningModel: string,
  frequency: string
): Promise<Partial<ForecastResult>> {
  const response = await fetch(`${API_BASE}/insights`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      data_summary: dataSummary,
      target_col: targetCol,
      time_col: timeCol,
      covariates: covariates,
      filters: filters,
      seasonality_mode: seasonalityMode,
      winning_model: winningModel,
      frequency: frequency
    })
  });

  if (!response.ok) {
    throw new Error(`Backend insights failed: ${response.statusText}`);
  }

  return await response.json();
}

export const analyzeDataset = async (sampleData: DataRow[], columns: string[]): Promise<DatasetAnalysis> => {
  try {
    console.log('Calling backend for dataset analysis');
    const result = await callBackendAnalysis(sampleData, columns);
    console.log('Analysis complete:', result);
    return result;
  } catch (error: any) {
    console.error("Backend Analysis Error:", error);
    return {
      summary: "Could not analyze dataset. Please configure columns manually.",
      suggestedCovariates: [],
      suggestedGroupColumns: []
    };
  }
};

interface SeasonalityConfig {
  mode: 'additive' | 'multiplicative';
  yearly: boolean;
  weekly: boolean;
}

export const generateForecastInsights = async (
  dataSummary: string,
  targetCol: string,
  timeCol: string,
  covariates: string[],
  filters: Record<string, string>,
  startDate: string | undefined,
  seasonality: SeasonalityConfig,
  selectedModels: ModelType[],
  winningModel: string,
  frequency: 'daily' | 'weekly' | 'monthly',
  regressorMethod: FutureRegressorMethod
): Promise<Partial<ForecastResult>> => {
  try {
    console.log('Calling backend for forecast insights');
    const result = await callBackendInsights(
      dataSummary,
      targetCol,
      timeCol,
      covariates,
      filters,
      seasonality.mode,
      winningModel,
      frequency
    );
    console.log('Insights generated');
    return result;
  } catch (error) {
    console.error("Backend Insights Error:", error);
    return {
      explanation: `Model trained successfully using ${winningModel}. The model used ${seasonality.mode} seasonality and incorporated ${covariates.length} covariates.`,
      pythonCode: `# Prophet model training example\nfrom prophet import Prophet\n\nmodel = Prophet(seasonality_mode='${seasonality.mode}')\nmodel.fit(df)\nforecast = model.predict(future)`
    };
  }
};

