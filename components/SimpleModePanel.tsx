import React, { useState, useRef, useMemo, useEffect } from 'react';
import {
  Upload,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Download,
  FileSpreadsheet,
  Info,
  TrendingUp,
  Calendar,
  Target,
  Shield,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  HelpCircle,
  FileText,
  Table,
  Layers,
  XCircle,
  Check,
  Sparkles,
  GitBranch,
  BarChart3,
  Lightbulb,
  ArrowRight,
  PieChart,
  Users,
  MapPin,
  Package,
  Zap,
  LineChart,
  Search,
  DollarSign,
  Megaphone,
  Globe,
  Hash,
  ChevronRight,
} from 'lucide-react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface DataProfile {
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
  all_columns?: string[];
  numeric_columns?: string[];
  date_columns?: string[];
}

interface Warning {
  level: string;
  message: string;
  recommendation: string;
}

interface ProfileResponse {
  success: boolean;
  profile: DataProfile;
  warnings: Warning[];
  config_preview: any;
}

// Model comparison entry
interface ModelComparisonEntry {
  model: string;
  eval_mape: number;
  holdout_mape: number;
  mape_difference: number;
  overfit_warning: boolean;
}

// Data split info
interface DataSplitInfo {
  total_rows: number;
  train_size: number;
  eval_size: number;
  holdout_size: number;
  train_pct: number;
  eval_pct: number;
  holdout_pct: number;
  train_date_range?: [string, string];
  eval_date_range?: [string, string];
  holdout_date_range?: [string, string];
  explanation?: string;
}

// Anomaly detection
interface ForecastAnomaly {
  type: string;
  severity: 'critical' | 'warning' | 'info';
  periods?: number[];
  description: string;
  cause: string;
  fix: string;
  forecast_values?: number[];
  threshold?: number;
}

interface ForecastResponse {
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
    totals: {
      base: number;
      trend: number;
      seasonal: number;
      holiday: number;
    };
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
  warnings: Warning[];
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
  // NEW: Data split and model selection
  data_split?: DataSplitInfo;
  model_comparison?: ModelComparisonEntry[];
  selection_reason?: string;
  trained_on_full_data?: boolean;
  // NEW: Anomaly detection
  anomalies?: ForecastAnomaly[];
  // NEW: Holdout performance
  holdout_mape?: number;
  eval_mape?: number;
  best_model?: string;
  all_models_trained?: string[];
  // NEW: Future covariates
  future_covariates_used?: boolean;
  future_covariates_count?: number;
  future_covariates_date_range?: string[];
  // NEW: By-slice forecasting
  forecast_mode?: 'aggregate' | 'by_slice';
  slice_forecasts?: SliceForecastResult[];
  slice_columns?: string[];
}

// Slice forecast result
interface SliceForecastResult {
  slice_id: string;
  slice_filters: Record<string, string>;
  forecast: number[];
  dates: string[];
  lower_bounds: number[];
  upper_bounds: number[];
  best_model?: string;
  holdout_mape?: number;
  data_points: number;
}

interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

// Slice/Segment information (single column)
interface SliceInfo {
  column: string;
  uniqueValues: string[];
  count: number;
  sampleValues: string[];
  suggestedType: 'region' | 'product' | 'category' | 'customer' | 'channel' | 'other';
}

// Combined slice info (multiple columns)
interface CombinedSliceInfo {
  columns: string[];  // e.g., ['region', 'product_category']
  combinedKey: string; // e.g., "region + product_category"
  uniqueCombinations: string[]; // e.g., ["North|Electronics", "North|Apparel", ...]
  count: number;
  sampleCombinations: string[];
}

// NEW: Column grouping for wide datasets
interface ColumnGroup {
  type: 'date' | 'target' | 'slice' | 'holiday' | 'numeric' | 'other';
  label: string;
  icon: string;
  columns: string[];
  color: string;
  bgColor: string;
  description: string;
}

// NEW: Slice combination with selection state
interface SliceCombination {
  id: string;
  values: Record<string, string>;
  displayName: string;
  rowCount: number;
  targetSum: number;
  targetAvg: number;
  isSelected: boolean;
}

// NEW: Covariate category for organized selection
interface CovariateCategory {
  type: 'holiday' | 'pricing' | 'promotion' | 'external' | 'numeric' | 'other';
  label: string;
  icon: string;
  columns: string[];
  color: string;
  description: string;
}

// AI Guidance for choices
interface ChoiceGuidance {
  title: string;
  description: string;
  consequence: string;
  recommendation: string;
  impact: 'high' | 'medium' | 'low';
}

interface SimpleModeState {
  step: 'upload' | 'profile' | 'configure' | 'forecasting' | 'results';
  file: File | null;
  rawData: any[] | null;
  columns: string[];
  profile: ProfileResponse | null;
  forecast: ForecastResponse | null;
  isLoading: boolean;
  error: string | null;
  horizon: number;
  // User-selected columns
  selectedDateCol: string;
  selectedTargetCol: string;
  selectedCovariates: string[];
  // Validation
  validation: ValidationResult | null;
  // Slice/Segment detection
  detectedSlices: SliceInfo[];
  selectedSliceCols: string[];  // Changed: now supports multiple columns
  selectedSliceValues: string[];
  combinedSliceInfo: CombinedSliceInfo | null;  // New: info about combined slices
  // Forecast mode
  forecastMode: 'aggregate' | 'by_slice';
  // Slice view selection (for results page)
  selectedSliceView: string; // "all" or slice_id
  // NEW: Enhanced UI state for wide datasets
  selectedSliceCombinations: string[];  // IDs of selected slice combinations
  sliceSearchQuery: string;  // Search filter for slices
  covariateSearchQuery: string;  // Search filter for covariates
  expandedColumnGroups: string[];  // Which column groups are expanded
  expandedCovariateGroups: string[];  // Which covariate categories are expanded
  sliceSelectionExplicit: boolean;  // True when user has explicitly modified slice selection
  // User overrides for column classification
  columnOverrides: Record<string, 'date' | 'slice' | 'holiday' | 'numeric' | 'other'>;
  reclassifyingColumn: string | null;  // Column currently being reclassified
  draggingColumn: string | null;  // Column currently being dragged
  dragOverGroup: string | null;  // Group being hovered over during drag
  holidaysAutoSelected: boolean;  // Track if holidays have been auto-selected as covariates
}

// Data format examples
const DATA_EXAMPLES = {
  timeseries: {
    title: 'Main Time Series Data',
    description: 'Your historical data with dates and values to forecast',
    required: true,
    columns: ['Date column (e.g., date, week, month)', 'Value column (numeric, e.g., revenue, sales, volume)'],
    example: `date,revenue,region
2024-01-01,150000,North
2024-01-08,162000,North
2024-01-15,148000,North
...`,
  },
  promotions: {
    title: 'Promotions/Events (Optional)',
    description: 'External factors that affect your values',
    required: false,
    columns: ['Date column (matching main data)', 'Binary flags (0/1) or numeric values'],
    example: `date,promo,holiday,campaign
2024-01-01,0,1,0
2024-01-08,1,0,0
2024-01-15,0,0,1
...`,
  },
};

// Minimum data requirements
const MIN_REQUIREMENTS = {
  rows: 12,
  months: 6,
  recommended_rows: 52,
  recommended_months: 24,
};

// AI Guidance messages for different scenarios
const AI_GUIDANCE = {
  sliceExplanation: {
    title: "Understanding Data Slices",
    intro: "Your data contains multiple segments (like regions, products, or customer types).",
    description: "Think of slices as different \"groups\" in your data that might have different patterns:",
    examples: [
      { type: "Business segment", values: "Enterprise, SMB, Subscription", impact: "each may have different growth patterns" },
      { type: "Channel type", values: "Pickup vs Classic vs Subscription", impact: "may respond differently to seasonality" },
      { type: "Customer type", values: "CGNA vs Non-CGNA", impact: "may have different volume trends" },
    ],
  },
  aggregateMode: {
    title: "Aggregate Forecast",
    consequence: "Creates ONE forecast for your total/combined data",
    details: [
      "Faster to compute (single model)",
      "Best when patterns are consistent across segments",
      "May miss segment-specific behaviors",
      "Good for high-level planning"
    ],
    whenToUse: "Choose this when you need overall trends and don't need segment-specific predictions",
    example: {
      title: "Example: Regional Sales Data",
      before: [
        { region: "North", week: "Week 1", sales: 100 },
        { region: "South", week: "Week 1", sales: 150 },
        { region: "North", week: "Week 2", sales: 120 },
        { region: "South", week: "Week 2", sales: 180 },
      ],
      after: [
        { week: "Week 1", sales: 250 },
        { week: "Week 2", sales: 300 },
      ],
      explanation: "Data is summed across all regions. The model sees total sales (250, 300...) and predicts future totals. You get ONE forecast number per period.",
      useCase: "Perfect for: Company-wide budgeting, total inventory planning, executive dashboards",
    },
  },
  sliceMode: {
    title: "Forecast by Slice",
    consequence: "Creates SEPARATE forecasts for each segment",
    details: [
      "More accurate for segment-specific planning",
      "Captures unique patterns per segment",
      "Takes longer (trains multiple models)",
      "Requires sufficient data per segment"
    ],
    whenToUse: "Choose this when segments behave differently and you need granular predictions",
    example: {
      title: "Example: Regional Sales Data",
      explanation: "Each region gets its own model trained on its own data. North learns North's patterns (maybe winter peaks), South learns South's patterns (maybe summer peaks).",
      result: [
        { region: "North", forecast: "Own trend + seasonality" },
        { region: "South", forecast: "Own trend + seasonality" },
      ],
      useCase: "Perfect for: Regional inventory, territory planning, segment-specific marketing budgets",
    },
  },
  targetColumn: {
    title: "Target Column Selection",
    content: "This is the value you want to predict. Choose carefully as it determines what you're forecasting.",
    consequences: {
      revenue: "Revenue forecasts help with budgeting and financial planning",
      sales: "Sales volume forecasts help with inventory and capacity planning",
      volume: "Volume forecasts help with operations and supply chain",
    }
  },
  covariates: {
    title: "Feature Selection Impact",
    content: "Covariates are external factors that influence your target. Including relevant ones improves accuracy.",
    consequences: {
      include: "More features = potentially better accuracy BUT risk of overfitting with limited data",
      exclude: "Fewer features = simpler model, more robust, but may miss important patterns",
    }
  },
};

// Slice type icons and colors
const SLICE_TYPE_CONFIG: Record<string, { icon: any; color: string; bgColor: string }> = {
  region: { icon: MapPin, color: 'text-blue-600', bgColor: 'bg-blue-100' },
  product: { icon: Package, color: 'text-purple-600', bgColor: 'bg-purple-100' },
  category: { icon: Layers, color: 'text-green-600', bgColor: 'bg-green-100' },
  customer: { icon: Users, color: 'text-orange-600', bgColor: 'bg-orange-100' },
  channel: { icon: GitBranch, color: 'text-pink-600', bgColor: 'bg-pink-100' },
  other: { icon: PieChart, color: 'text-gray-600', bgColor: 'bg-gray-100' },
};

export const SimpleModePanel: React.FC = () => {
  const [state, setState] = useState<SimpleModeState>({
    step: 'upload',
    file: null,
    rawData: null,
    columns: [],
    profile: null,
    forecast: null,
    isLoading: false,
    error: null,
    horizon: 12,
    selectedDateCol: '',
    selectedTargetCol: '',
    selectedCovariates: [],
    validation: null,
    detectedSlices: [],
    selectedSliceCols: [],
    selectedSliceValues: [],
    combinedSliceInfo: null,
    forecastMode: 'aggregate',
    selectedSliceView: 'all',
    // NEW: Enhanced UI state
    selectedSliceCombinations: [],
    sliceSearchQuery: '',
    covariateSearchQuery: '',
    expandedColumnGroups: ['slice', 'holiday'],  // Default expanded
    expandedCovariateGroups: ['holiday', 'pricing'],  // Default expanded
    sliceSelectionExplicit: false,  // Becomes true when user explicitly selects/clears
    columnOverrides: {},  // User overrides for column classification
    reclassifyingColumn: null,  // Column currently being reclassified
    draggingColumn: null,  // Column currently being dragged
    dragOverGroup: null,  // Group being hovered over during drag
    holidaysAutoSelected: false,  // Track if holidays have been auto-selected
  });

  const [showComponents, setShowComponents] = useState(false);
  const [showAudit, setShowAudit] = useState(false);
  const [showDataGuide, setShowDataGuide] = useState(true);
  const [showSliceGuide, setShowSliceGuide] = useState(false);
  const [showImpactPreview, setShowImpactPreview] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Parse CSV file locally for preview and validation
  const parseCSV = (text: string): { headers: string[]; rows: any[] } => {
    const lines = text.trim().split('\n');
    if (lines.length === 0) return { headers: [], rows: [] };

    const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
    const rows = lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
      const row: Record<string, any> = {};
      headers.forEach((h, i) => {
        row[h] = values[i];
      });
      return row;
    });

    return { headers, rows };
  };

  // Detect column types
  const detectColumnTypes = (headers: string[], rows: any[]) => {
    // Holiday patterns - check these FIRST to avoid false positives with "day" pattern
    const holidayPatterns = [
      /valentine/i, /patrick/i, /mother/i, /father/i, /easter/i, /christmas/i,
      /thanksgiving/i, /halloween/i, /july.?4/i, /independence/i, /memorial/i,
      /labor/i, /veterans/i, /mlk/i, /president/i, /black.?friday/i, /cyber/i,
      /super.?bowl/i, /cinco/i, /new.?year/i, /holiday/i, /eve$/i
    ];

    // Date patterns - more specific to avoid matching holiday names
    const datePatterns = [
      /^date$/i, /^ds$/i, /^time$/i, /^week$/i, /^month$/i, /^period$/i,
      /^year$/i, /_date$/i, /_time$/i, /_week$/i, /_month$/i, /^datetime/i,
      /timestamp/i, /^dt$/i
    ];

    const dateColumns: string[] = [];
    const numericColumns: string[] = [];
    const categoricalColumns: string[] = [];
    const holidayColumns: string[] = [];

    headers.forEach(col => {
      const colLower = col.toLowerCase();

      // Sample values to determine type by content
      const sampleValues = rows.slice(0, 20).map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '');

      // Check if it's a binary column (likely a holiday indicator)
      const isBinary = sampleValues.length > 0 && sampleValues.every(v => {
        const val = String(v).trim();
        return val === '0' || val === '1' || val === 'true' || val === 'false';
      });

      // Check if column name matches holiday patterns
      const isHolidayName = holidayPatterns.some(p => p.test(col));

      // If it's binary OR matches holiday name patterns, it's likely a holiday column
      if (isHolidayName || (isBinary && !datePatterns.some(p => p.test(col)))) {
        holidayColumns.push(col);
        // Also count as numeric if binary
        if (isBinary) {
          numericColumns.push(col);
        }
        return;
      }

      // Check if it's a date column by name pattern
      if (datePatterns.some(p => p.test(col))) {
        dateColumns.push(col);
        return;
      }

      // Check if values look like dates
      if (sampleValues.length > 0) {
        const looksLikeDate = sampleValues.slice(0, 5).every(v => {
          const str = String(v);
          // Check for date-like patterns: YYYY-MM-DD, MM/DD/YYYY, etc.
          return !isNaN(Date.parse(str)) && str.length >= 8 && /[-\/]/.test(str);
        });
        if (looksLikeDate) {
          dateColumns.push(col);
          return;
        }
      }

      // Check if numeric
      const numericCount = sampleValues.filter(v => !isNaN(parseFloat(String(v)))).length;
      if (numericCount >= sampleValues.length * 0.8) {
        numericColumns.push(col);
      } else {
        categoricalColumns.push(col);
      }
    });

    // If no date column found, try first column
    if (dateColumns.length === 0 && rows.length > 0) {
      const firstCol = headers[0];
      const sampleDate = rows[0][firstCol];
      if (sampleDate && !isNaN(Date.parse(String(sampleDate)))) {
        dateColumns.push(firstCol);
      }
    }

    return { dateColumns, numericColumns, categoricalColumns, holidayColumns };
  };

  // Detect slice/segment columns (categorical with reasonable cardinality)
  const detectSlices = (headers: string[], rows: any[]): SliceInfo[] => {
    const slicePatterns: Record<string, RegExp[]> = {
      region: [/region/i, /area/i, /territory/i, /zone/i, /location/i, /geo/i, /country/i, /state/i, /city/i],
      product: [/product/i, /item/i, /sku/i, /brand/i],
      category: [/category/i, /cat/i, /type/i, /class/i, /group/i, /segment/i],
      customer: [/customer/i, /client/i, /account/i, /buyer/i],
      channel: [/channel/i, /source/i, /medium/i, /platform/i, /store/i, /outlet/i],
    };

    const slices: SliceInfo[] = [];

    headers.forEach(col => {
      // Skip date/time-like columns
      if (/date|time|ds|week|month|day|year|period/i.test(col)) return;

      // Get unique values
      const values = rows.map(r => String(r[col] || '')).filter(v => v && v.trim());
      const uniqueValues = [...new Set(values)];

      // Good slice candidates: 2-50 unique values (not too few, not too many)
      if (uniqueValues.length >= 2 && uniqueValues.length <= 50) {
        // Check if values are mostly non-numeric (categorical)
        const numericCount = uniqueValues.filter(v => !isNaN(parseFloat(v)) && isFinite(Number(v))).length;
        const isLikelyCategorical = numericCount < uniqueValues.length * 0.3;

        // Skip if looks like a numeric ID or continuous variable
        if (!isLikelyCategorical && uniqueValues.length > 10) return;

        // Determine suggested type
        let suggestedType: SliceInfo['suggestedType'] = 'other';
        const colLower = col.toLowerCase();

        for (const [type, patterns] of Object.entries(slicePatterns)) {
          if (patterns.some(p => p.test(colLower))) {
            suggestedType = type as SliceInfo['suggestedType'];
            break;
          }
        }

        slices.push({
          column: col,
          uniqueValues,
          count: uniqueValues.length,
          sampleValues: uniqueValues.slice(0, 5),
          suggestedType,
        });
      }
    });

    // Sort by likelihood of being a useful slice (prefer known types over 'other')
    return slices.sort((a, b) => {
      if (a.suggestedType !== 'other' && b.suggestedType === 'other') return -1;
      if (a.suggestedType === 'other' && b.suggestedType !== 'other') return 1;
      return a.count - b.count; // Prefer fewer unique values
    });
  };

  // Calculate combined slice info when multiple columns are selected
  const getCombinedSliceInfo = useMemo((): CombinedSliceInfo | null => {
    if (!state.rawData || state.selectedSliceCols.length === 0) return null;

    const cols = state.selectedSliceCols;
    const combinationsSet = new Set<string>();

    state.rawData.forEach(row => {
      const combinedValue = cols.map(col => String(row[col] || 'Unknown')).join(' | ');
      combinationsSet.add(combinedValue);
    });

    const uniqueCombinations = Array.from(combinationsSet).sort();

    return {
      columns: cols,
      combinedKey: cols.join(' + '),
      uniqueCombinations,
      count: uniqueCombinations.length,
      sampleCombinations: uniqueCombinations.slice(0, 5),
    };
  }, [state.rawData, state.selectedSliceCols]);

  // Calculate data stats per slice for impact preview (supports multiple columns)
  const getSliceStats = useMemo(() => {
    if (!state.rawData || state.selectedSliceCols.length === 0 || !state.selectedTargetCol) return null;

    const sliceCols = state.selectedSliceCols;
    const targetCol = state.selectedTargetCol;

    const stats: Record<string, { count: number; sum: number; avg: number }> = {};

    state.rawData.forEach(row => {
      // Create combined key for multiple columns
      const sliceKey = sliceCols.map(col => String(row[col] || 'Unknown')).join(' | ');
      const value = parseFloat(row[targetCol]) || 0;

      if (!stats[sliceKey]) {
        stats[sliceKey] = { count: 0, sum: 0, avg: 0 };
      }
      stats[sliceKey].count++;
      stats[sliceKey].sum += value;
    });

    // Calculate averages
    Object.keys(stats).forEach(slice => {
      stats[slice].avg = stats[slice].sum / stats[slice].count;
    });

    return stats;
  }, [state.rawData, state.selectedSliceCols, state.selectedTargetCol]);

  // NEW: Group columns by type for organized display (with user overrides)
  const getColumnGroups = useMemo((): ColumnGroup[] => {
    if (!state.columns.length || !state.rawData) return [];

    const { dateColumns, numericColumns, categoricalColumns, holidayColumns } = detectColumnTypes(state.columns, state.rawData);
    const sliceColumns = state.detectedSlices.map(s => s.column);

    // Build initial classification for each column
    const columnClassification: Record<string, 'date' | 'slice' | 'holiday' | 'numeric' | 'other'> = {};

    state.columns.forEach(col => {
      // Check user override first
      if (state.columnOverrides[col]) {
        columnClassification[col] = state.columnOverrides[col];
        return;
      }

      // Auto-classification logic
      if (holidayColumns.includes(col)) {
        columnClassification[col] = 'holiday';
      } else if (dateColumns.includes(col)) {
        columnClassification[col] = 'date';
      } else if (sliceColumns.includes(col)) {
        columnClassification[col] = 'slice';
      } else if (numericColumns.includes(col)) {
        columnClassification[col] = 'numeric';
      } else {
        columnClassification[col] = 'other';
      }
    });

    // Group columns by their classification
    const groupedColumns: Record<string, string[]> = {
      date: [],
      slice: [],
      holiday: [],
      numeric: [],
      other: [],
    };

    state.columns.forEach(col => {
      const type = columnClassification[col];
      groupedColumns[type].push(col);
    });

    // Build groups array
    const groups: ColumnGroup[] = [];

    if (groupedColumns.date.length > 0) {
      groups.push({
        type: 'date',
        label: 'Date Columns',
        icon: 'Calendar',
        columns: groupedColumns.date,
        color: 'text-blue-600',
        bgColor: 'bg-blue-50',
        description: 'Time/date columns for your time series'
      });
    }

    if (groupedColumns.slice.length > 0) {
      groups.push({
        type: 'slice',
        label: 'Segment Columns',
        icon: 'Layers',
        columns: groupedColumns.slice,
        color: 'text-purple-600',
        bgColor: 'bg-purple-50',
        description: 'Categorical columns that segment your data (region, product, etc.)'
      });
    }

    if (groupedColumns.holiday.length > 0) {
      groups.push({
        type: 'holiday',
        label: 'Holidays & Events',
        icon: 'Calendar',
        columns: groupedColumns.holiday,
        color: 'text-orange-600',
        bgColor: 'bg-orange-50',
        description: 'Binary indicators for holidays and special events'
      });
    }

    if (groupedColumns.numeric.length > 0) {
      groups.push({
        type: 'numeric',
        label: 'Numeric Columns',
        icon: 'TrendingUp',
        columns: groupedColumns.numeric,
        color: 'text-green-600',
        bgColor: 'bg-green-50',
        description: 'Numeric values including your target and potential covariates'
      });
    }

    if (groupedColumns.other.length > 0) {
      groups.push({
        type: 'other',
        label: 'Other Columns',
        icon: 'FileText',
        columns: groupedColumns.other,
        color: 'text-gray-600',
        bgColor: 'bg-gray-50',
        description: 'Text and other non-categorized columns'
      });
    }

    return groups;
  }, [state.columns, state.rawData, state.detectedSlices, state.columnOverrides]);

  // Auto-select holiday columns as covariates when data is loaded
  useEffect(() => {
    // Only run once when column groups are computed and holidays haven't been auto-selected yet
    if (getColumnGroups.length > 0 && !state.holidaysAutoSelected && state.step === 'configure') {
      const holidayGroup = getColumnGroups.find(g => g.type === 'holiday');
      if (holidayGroup && holidayGroup.columns.length > 0) {
        // Auto-select all holiday columns as covariates
        const holidayCols = holidayGroup.columns.filter(col =>
          col !== state.selectedDateCol &&
          col !== state.selectedTargetCol
        );

        if (holidayCols.length > 0) {
          setState(s => ({
            ...s,
            selectedCovariates: [...new Set([...s.selectedCovariates, ...holidayCols])],
            holidaysAutoSelected: true,
          }));
          console.log(`🎄 Auto-selected ${holidayCols.length} holiday columns as covariates:`, holidayCols);
        } else {
          setState(s => ({ ...s, holidaysAutoSelected: true }));
        }
      } else {
        // No holidays found, mark as done
        setState(s => ({ ...s, holidaysAutoSelected: true }));
      }
    }
  }, [getColumnGroups, state.holidaysAutoSelected, state.step, state.selectedDateCol, state.selectedTargetCol]);

  // NEW: Get all slice combinations with stats for selection UI
  const getSliceCombinations = useMemo((): SliceCombination[] => {
    if (!state.rawData || state.selectedSliceCols.length === 0) return [];

    const sliceCols = state.selectedSliceCols;
    const targetCol = state.selectedTargetCol;
    const combinations: Map<string, SliceCombination> = new Map();

    state.rawData.forEach(row => {
      const values: Record<string, string> = {};
      sliceCols.forEach(col => {
        values[col] = String(row[col] || 'Unknown');
      });
      const id = sliceCols.map(col => values[col]).join(' | ');
      const displayName = id;
      const targetValue = targetCol ? parseFloat(row[targetCol]) || 0 : 0;

      if (!combinations.has(id)) {
        combinations.set(id, {
          id,
          values,
          displayName,
          rowCount: 0,
          targetSum: 0,
          targetAvg: 0,
          isSelected: state.selectedSliceCombinations.includes(id) || state.selectedSliceCombinations.length === 0,
        });
      }

      const combo = combinations.get(id)!;
      combo.rowCount++;
      combo.targetSum += targetValue;
    });

    // Calculate averages and convert to array
    return Array.from(combinations.values())
      .map(c => ({ ...c, targetAvg: c.targetSum / c.rowCount }))
      .sort((a, b) => b.targetSum - a.targetSum);
  }, [state.rawData, state.selectedSliceCols, state.selectedTargetCol, state.selectedSliceCombinations]);

  // NEW: Compute effective segments that respect user's column reclassifications
  const getEffectiveSegments = useMemo((): SliceInfo[] => {
    if (!state.rawData) return state.detectedSlices;

    const { numericColumns, holidayColumns } = detectColumnTypes(state.columns, state.rawData);

    // Start with original detected slices, but filter based on overrides AND auto-detected types
    let effectiveSlices = state.detectedSlices.filter(slice => {
      const override = state.columnOverrides[slice.column];

      // If user explicitly reclassified, respect that
      if (override) {
        return override === 'slice';
      }

      // If auto-detected as holiday, don't show in segment list
      if (holidayColumns.includes(slice.column)) {
        return false;
      }

      // If auto-detected as numeric, don't show in segment list
      if (numericColumns.includes(slice.column)) {
        return false;
      }

      return true;
    });

    // Add any columns that user explicitly reclassified as 'slice'
    Object.entries(state.columnOverrides).forEach(([col, type]) => {
      if (type === 'slice' && !effectiveSlices.some(s => s.column === col)) {
        // Calculate unique values for this column
        const uniqueValues = [...new Set(state.rawData!.map(row => String(row[col] ?? '')))].filter(v => v !== '');

        // Determine suggested type based on column characteristics
        let suggestedType: 'category' | 'region' | 'product' | 'customer' | 'other' = 'category';
        const lowerCol = col.toLowerCase();
        if (lowerCol.includes('region') || lowerCol.includes('area') || lowerCol.includes('zone')) {
          suggestedType = 'region';
        } else if (lowerCol.includes('product') || lowerCol.includes('item') || lowerCol.includes('sku')) {
          suggestedType = 'product';
        } else if (lowerCol.includes('customer') || lowerCol.includes('client') || lowerCol.includes('segment')) {
          suggestedType = 'customer';
        }

        effectiveSlices.push({
          column: col,
          count: uniqueValues.length,
          uniqueValues: uniqueValues.slice(0, 20),
          suggestedType,
          hasNulls: state.rawData!.some(row => row[col] === null || row[col] === undefined || row[col] === ''),
        });
      }
    });

    return effectiveSlices;
  }, [state.detectedSlices, state.columnOverrides, state.rawData, state.columns]);

  // NEW: Categorize covariates for organized selection
  const getCovariateCategories = useMemo((): CovariateCategory[] => {
    if (!state.columns.length || !state.rawData) return [];

    const categories: CovariateCategory[] = [];
    const { dateColumns, numericColumns, holidayColumns } = detectColumnTypes(state.columns, state.rawData);

    // Use effective segments (respects column overrides) instead of raw detectedSlices
    // This ensures holiday columns reclassified from segments are available as covariates
    const effectiveSliceColumns = getEffectiveSegments.map(s => s.column);

    // Get available covariates - include both numeric AND holiday columns
    // (exclude date, target, and EFFECTIVE slice columns only)
    const allPotentialCovariates = [...new Set([...numericColumns, ...holidayColumns])];
    const availableCovariates = allPotentialCovariates.filter(c =>
      c !== state.selectedDateCol &&
      c !== state.selectedTargetCol &&
      !effectiveSliceColumns.includes(c)
    );

    // Categorization patterns
    const patterns = {
      holiday: [
        /holiday/i, /easter/i, /christmas/i, /thanksgiving/i, /halloween/i,
        /new.?year/i, /memorial/i, /labor/i, /independence/i, /veterans/i,
        /black.?friday/i, /cyber/i, /super.?bowl/i, /cinco/i, /valentine/i,
        /mother/i, /father/i, /mlk/i, /president/i
      ],
      pricing: [/price/i, /cost/i, /fee/i, /rate/i, /discount/i, /margin/i],
      promotion: [/promo/i, /campaign/i, /marketing/i, /ad/i, /event/i],
      external: [/weather/i, /temp/i, /econ/i, /gdp/i, /index/i, /rate/i, /competitor/i],
    };

    const categorized: Record<string, string[]> = {
      holiday: [],
      pricing: [],
      promotion: [],
      external: [],
      numeric: [],
    };

    availableCovariates.forEach(col => {
      let found = false;
      for (const [category, patternList] of Object.entries(patterns)) {
        if (patternList.some(p => p.test(col))) {
          categorized[category].push(col);
          found = true;
          break;
        }
      }
      if (!found) {
        // Check if it's a binary column (likely a holiday/flag)
        const isBinary = state.rawData!.slice(0, 50).every(row => {
          const val = row[col];
          return val === '0' || val === '1' || val === 0 || val === 1 || val === '' || val === null || val === undefined;
        });
        if (isBinary) {
          categorized.holiday.push(col);
        } else {
          categorized.numeric.push(col);
        }
      }
    });

    // Build category objects
    if (categorized.holiday.length > 0) {
      categories.push({
        type: 'holiday',
        label: `Holidays & Events (${categorized.holiday.length})`,
        icon: 'Calendar',
        columns: categorized.holiday,
        color: 'text-orange-600',
        description: 'Binary indicators for holidays and special events'
      });
    }

    if (categorized.pricing.length > 0) {
      categories.push({
        type: 'pricing',
        label: `Pricing Features (${categorized.pricing.length})`,
        icon: 'DollarSign',
        columns: categorized.pricing,
        color: 'text-green-600',
        description: 'Price-related variables that may affect demand'
      });
    }

    if (categorized.promotion.length > 0) {
      categories.push({
        type: 'promotion',
        label: `Promotions & Marketing (${categorized.promotion.length})`,
        icon: 'Megaphone',
        columns: categorized.promotion,
        color: 'text-pink-600',
        description: 'Marketing campaigns and promotional activities'
      });
    }

    if (categorized.external.length > 0) {
      categories.push({
        type: 'external',
        label: `External Factors (${categorized.external.length})`,
        icon: 'Globe',
        columns: categorized.external,
        color: 'text-blue-600',
        description: 'Weather, economic indicators, and other external data'
      });
    }

    if (categorized.numeric.length > 0) {
      categories.push({
        type: 'numeric',
        label: `Other Numeric (${categorized.numeric.length})`,
        icon: 'Hash',
        columns: categorized.numeric,
        color: 'text-gray-600',
        description: 'Other numeric columns that may be useful predictors'
      });
    }

    return categories;
  }, [state.columns, state.rawData, state.detectedSlices, state.selectedDateCol, state.selectedTargetCol]);

  // NEW: Filter slice combinations by search query
  const filteredSliceCombinations = useMemo(() => {
    if (!state.sliceSearchQuery.trim()) return getSliceCombinations;
    const query = state.sliceSearchQuery.toLowerCase();
    return getSliceCombinations.filter(c =>
      c.displayName.toLowerCase().includes(query)
    );
  }, [getSliceCombinations, state.sliceSearchQuery]);

  // NEW: Toggle slice combination selection
  const toggleSliceCombination = (id: string) => {
    setState(s => {
      const isCurrentlySelected = s.selectedSliceCombinations.includes(id);
      const newSelection = isCurrentlySelected
        ? s.selectedSliceCombinations.filter(x => x !== id)
        : [...s.selectedSliceCombinations, id];
      return { ...s, selectedSliceCombinations: newSelection };
    });
  };

  // NEW: Select/deselect all slice combinations
  const selectAllSliceCombinations = (selectAll: boolean) => {
    setState(s => ({
      ...s,
      selectedSliceValues: selectAll ? getSliceCombinations.map(c => c.id) : [],
      selectedSliceCombinations: selectAll ? getSliceCombinations.map(c => c.id) : [],
      sliceSelectionExplicit: true,  // User has explicitly modified selection
      // Clear forecast when selection changes
      forecast: null,
    }));
  };

  // NEW: Toggle covariate category (select all in category)
  const toggleCovariateCategory = (category: CovariateCategory) => {
    setState(s => {
      const categoryColumns = category.columns;
      const allSelected = categoryColumns.every(c => s.selectedCovariates.includes(c));

      if (allSelected) {
        // Deselect all in category
        return {
          ...s,
          selectedCovariates: s.selectedCovariates.filter(c => !categoryColumns.includes(c))
        };
      } else {
        // Select all in category
        const newSelection = new Set([...s.selectedCovariates, ...categoryColumns]);
        return { ...s, selectedCovariates: Array.from(newSelection) };
      }
    });
  };

  // NEW: Generate dynamic examples based on actual uploaded data
  const getDynamicExamples = useMemo(() => {
    if (!state.rawData || !state.selectedDateCol || !state.selectedTargetCol) {
      return null;
    }

    const dateCol = state.selectedDateCol;
    const targetCol = state.selectedTargetCol;
    const sliceCols = state.selectedSliceCols;

    // Get sample data from actual uploaded file
    const sampleRows = state.rawData.slice(0, 4);

    // Get unique dates for examples
    const uniqueDates = [...new Set(state.rawData.map(r => r[dateCol]))].slice(0, 2);

    // Format target value for display
    const formatValue = (val: any) => {
      const num = parseFloat(String(val).replace(/,/g, ''));
      if (isNaN(num)) return val;
      if (num >= 1000000) return `$${(num / 1000000).toFixed(1)}M`;
      if (num >= 1000) return `$${(num / 1000).toFixed(0)}K`;
      return `$${num.toFixed(0)}`;
    };

    // Build dynamic before/after examples
    const beforeData: any[] = [];
    const afterData: { [key: string]: { date: string; total: number } } = {};

    if (sliceCols.length > 0) {
      // Get unique slice values
      const sliceValues = [...new Set(state.rawData.map(r =>
        sliceCols.map(col => r[col]).join(' | ')
      ))].slice(0, 2);

      // Build before data with slices
      uniqueDates.forEach((date, dateIdx) => {
        sliceValues.forEach(slice => {
          const matchingRow = state.rawData!.find(r =>
            r[dateCol] === date &&
            sliceCols.map(col => r[col]).join(' | ') === slice
          );
          if (matchingRow) {
            beforeData.push({
              slice,
              date: `Period ${dateIdx + 1}`,
              value: matchingRow[targetCol]
            });
          }
        });
      });

      // Calculate aggregated totals
      uniqueDates.forEach((date, dateIdx) => {
        const total = state.rawData!
          .filter(r => r[dateCol] === date)
          .reduce((sum, r) => sum + (parseFloat(String(r[targetCol]).replace(/,/g, '')) || 0), 0);
        afterData[`Period ${dateIdx + 1}`] = { date: `Period ${dateIdx + 1}`, total };
      });
    }

    // Generate slice-specific examples for by_slice mode
    const sliceExamples = sliceCols.length > 0
      ? [...new Set(state.rawData.map(r =>
          sliceCols.map(col => r[col]).join(' | ')
        ))].slice(0, 3).map(slice => ({
          slice,
          description: `Own trend + seasonality`
        }))
      : [];

    return {
      dateColumn: dateCol,
      targetColumn: targetCol,
      sliceColumns: sliceCols,
      sliceColumnLabel: sliceCols.join(' + ') || 'Segment',
      beforeData,
      afterData: Object.values(afterData),
      sliceExamples,
      formatValue,
      // Generate meaningful use cases based on column names
      aggregateUseCase: targetCol.toLowerCase().includes('revenue')
        ? `Perfect for: Company-wide ${targetCol} budgeting, total financial planning, executive dashboards`
        : targetCol.toLowerCase().includes('volume') || targetCol.toLowerCase().includes('order')
        ? `Perfect for: Total ${targetCol} forecasting, capacity planning, operations dashboards`
        : `Perfect for: Overall ${targetCol} trends, high-level planning, aggregate budgeting`,
      sliceUseCase: sliceCols.length > 0
        ? `Perfect for: ${sliceCols.map(c => c.replace(/_/g, ' ')).join(' and ')}-specific planning, granular budgeting, segment performance tracking`
        : `Perfect for: Segment-specific planning, granular predictions, detailed analysis`,
    };
  }, [state.rawData, state.selectedDateCol, state.selectedTargetCol, state.selectedSliceCols]);

  // Validate data
  const validateData = (headers: string[], rows: any[]): ValidationResult => {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check minimum rows
    if (rows.length < MIN_REQUIREMENTS.rows) {
      errors.push(`Insufficient data: ${rows.length} rows found, minimum ${MIN_REQUIREMENTS.rows} required`);
    } else if (rows.length < MIN_REQUIREMENTS.recommended_rows) {
      warnings.push(`Limited data: ${rows.length} rows found. Recommend at least ${MIN_REQUIREMENTS.recommended_rows} rows for better accuracy`);
    }

    // Check for required columns
    const { dateColumns, numericColumns } = detectColumnTypes(headers, rows);

    if (dateColumns.length === 0) {
      errors.push('No date column detected. Please ensure your data has a date/time column');
    }

    if (numericColumns.length === 0) {
      errors.push('No numeric columns detected. Please ensure your data has at least one numeric column to forecast');
    }

    // Check for empty values
    const emptyCount = rows.filter(r => Object.values(r).some(v => v === '' || v === null || v === undefined)).length;
    if (emptyCount > rows.length * 0.1) {
      warnings.push(`${emptyCount} rows have missing values. This may affect forecast accuracy`);
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setState(s => ({ ...s, file, isLoading: true, error: null }));

    try {
      // Read file locally first for validation
      const text = await file.text();
      const { headers, rows } = parseCSV(text);

      // Validate
      const validation = validateData(headers, rows);
      const { dateColumns, numericColumns } = detectColumnTypes(headers, rows);

      // Detect slices/segments
      const detectedSlices = detectSlices(headers, rows);

      // Auto-select columns
      const autoDateCol = dateColumns[0] || '';
      const autoTargetCol = numericColumns[0] || '';

      // Auto-select first detected slice if any
      const autoSliceCol = detectedSlices.length > 0 ? detectedSlices[0].column : '';
      const autoSliceValues = autoSliceCol && detectedSlices[0]
        ? detectedSlices[0].uniqueValues
        : [];

      // Get slice column names to exclude from covariates
      const sliceColumnNames = detectedSlices.map(s => s.column);

      // Auto-select covariates (excluding date, target, and slice columns)
      const autoCovariates = numericColumns
        .slice(1)
        .filter(c => c !== autoTargetCol && !sliceColumnNames.includes(c));

      setState(s => ({
        ...s,
        rawData: rows,
        columns: headers,
        selectedDateCol: autoDateCol,
        selectedTargetCol: autoTargetCol,
        selectedCovariates: autoCovariates.slice(0, 5), // Limit to 5 covariates
        validation,
        detectedSlices,
        selectedSliceCols: autoSliceCol ? [autoSliceCol] : [],  // Changed to array
        selectedSliceValues: autoSliceValues,
        combinedSliceInfo: null,  // Will be computed by useMemo
        forecastMode: detectedSlices.length > 0 ? 'aggregate' : 'aggregate', // Default to aggregate
        isLoading: false,
        step: validation.isValid ? 'configure' : 'upload',
        error: validation.isValid ? null : validation.errors.join('. '),
      }));

      // Show slice guide if slices detected
      if (detectedSlices.length > 0) {
        setShowSliceGuide(true);
      }

      // If valid, also fetch profile from backend
      if (validation.isValid) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/simple/profile', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data: ProfileResponse = await response.json();
          // Add extra column info to profile
          data.profile.all_columns = headers;
          data.profile.numeric_columns = numericColumns;
          data.profile.date_columns = dateColumns;

          setState(s => {
            const newTargetCol = data.profile.target_column || s.selectedTargetCol;
            return {
              ...s,
              profile: data,
              horizon: data.profile.recommended_horizon,
              selectedDateCol: data.profile.date_column || s.selectedDateCol,
              selectedTargetCol: newTargetCol,
              // Remove target from covariates if backend updated target column
              selectedCovariates: s.selectedCovariates.filter(c => c !== newTargetCol),
            };
          });
        }
      }
    } catch (err: any) {
      setState(s => ({
        ...s,
        isLoading: false,
        error: err.message || 'Failed to read file',
        step: 'upload',
      }));
    }
  };

  const handleCovariateToggle = (col: string) => {
    setState(s => {
      // Don't allow selecting target as covariate
      if (col === s.selectedTargetCol) return s;
      return {
        ...s,
        selectedCovariates: s.selectedCovariates.includes(col)
          ? s.selectedCovariates.filter(c => c !== col)
          : [...s.selectedCovariates, col],
      };
    });
  };

  const handleForecast = async () => {
    if (!state.file) return;

    // Validate column selections
    if (!state.selectedDateCol) {
      setState(s => ({ ...s, error: 'Please select a date column' }));
      return;
    }
    if (!state.selectedTargetCol) {
      setState(s => ({ ...s, error: 'Please select a target column to forecast' }));
      return;
    }

    setState(s => ({ ...s, isLoading: true, error: null, step: 'forecasting' }));

    // ========== LOGGING: START FORECAST REQUEST ==========
    console.log('========================================');
    console.log('🚀 SIMPLE MODE FORECAST - REQUEST START');
    console.log('========================================');
    console.log('📋 Configuration:');
    console.log('  - Date Column:', state.selectedDateCol);
    console.log('  - Target Column:', state.selectedTargetCol);
    console.log('  - Horizon:', state.horizon);
    console.log('  - Covariates:', state.selectedCovariates);
    console.log('  - Forecast Mode:', state.forecastMode);
    console.log('  - Selected Slice Columns:', state.selectedSliceCols);
    console.log('  - Selected Slice Values:', state.selectedSliceValues);
    console.log('  - Number of slices selected:', state.selectedSliceValues.length);

    try {
      const formData = new FormData();
      formData.append('file', state.file);

      // Add user selections to the request
      const params = new URLSearchParams({
        horizon: String(state.horizon),
        date_col: state.selectedDateCol,
        target_col: state.selectedTargetCol,
        covariates: state.selectedCovariates.join(','),
      });

      // Add slice/segment parameters if by_slice mode is selected
      if (state.forecastMode === 'by_slice' && state.selectedSliceCols.length > 0) {
        params.set('forecast_mode', 'by_slice');
        params.set('slice_columns', state.selectedSliceCols.join(','));
        const sliceValuesJoined = state.selectedSliceValues.join('|||');
        params.set('slice_values', sliceValuesJoined);

        console.log('📤 SENDING TO BACKEND:');
        console.log('  - forecast_mode: by_slice');
        console.log('  - slice_columns:', state.selectedSliceCols.join(','));
        console.log('  - slice_values (raw):', sliceValuesJoined);
        console.log('  - slice_values (parsed):', state.selectedSliceValues);
      } else if (state.detectedSlices.length > 0) {
        params.set('forecast_mode', 'aggregate');
        console.log('📤 SENDING TO BACKEND: forecast_mode: aggregate');
      }

      console.log('🌐 Full URL:', `/api/simple/forecast?${params}`);

      const response = await fetch(`/api/simple/forecast?${params}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        console.error('❌ FORECAST FAILED:', errData);
        throw new Error(errData.detail || `Forecast failed: ${response.statusText}`);
      }

      const data: ForecastResponse = await response.json();

      // ========== LOGGING: RESPONSE RECEIVED ==========
      console.log('========================================');
      console.log('📥 SIMPLE MODE FORECAST - RESPONSE');
      console.log('========================================');
      console.log('✅ Success:', data.success);
      console.log('📊 Mode:', data.forecast_mode);
      console.log('🏷️ Slice Columns:', data.slice_columns);
      console.log('📈 Number of slice forecasts:', data.slice_forecasts?.length || 0);
      if (data.slice_forecasts) {
        console.log('📋 Slice Details:');
        data.slice_forecasts.forEach((sf, idx) => {
          console.log(`  [${idx}] slice_id: "${sf.slice_id}"`);
          console.log(`       slice_filters:`, sf.slice_filters);
          console.log(`       data_points: ${sf.data_points}`);
          console.log(`       best_model: ${sf.best_model}`);
          console.log(`       holdout_mape: ${sf.holdout_mape}`);
          console.log(`       forecast length: ${sf.forecast?.length || 0}`);
        });
      }
      console.log('========================================');

      setState(s => ({
        ...s,
        forecast: data,
        isLoading: false,
        step: 'results',
      }));
    } catch (err: any) {
      setState(s => ({
        ...s,
        isLoading: false,
        error: err.message || 'Failed to generate forecast',
        step: 'configure',
      }));
    }
  };

  const handleDownloadExcel = () => {
    if (!state.forecast) return;
    window.open(state.forecast.excel_download_url, '_blank');
  };

  const handleDownloadCSV = () => {
    if (!state.forecast) return;
    const csvUrl = state.forecast.excel_download_url.replace('/excel', '/csv');
    window.open(csvUrl, '_blank');
  };

  const handleReset = () => {
    setState({
      step: 'upload',
      file: null,
      rawData: null,
      columns: [],
      profile: null,
      forecast: null,
      isLoading: false,
      error: null,
      horizon: 12,
      selectedDateCol: '',
      selectedTargetCol: '',
      selectedCovariates: [],
      validation: null,
      detectedSlices: [],
      selectedSliceCols: [],
      selectedSliceValues: [],
      combinedSliceInfo: null,
      forecastMode: 'aggregate',
      selectedSliceView: 'all',
      // Reset enhanced UI state
      selectedSliceCombinations: [],
      sliceSearchQuery: '',
      covariateSearchQuery: '',
      expandedColumnGroups: ['slice', 'holiday'],
      expandedCovariateGroups: ['holiday', 'pricing'],
      sliceSelectionExplicit: false,
      columnOverrides: {},
      reclassifyingColumn: null,
      draggingColumn: null,
      dragOverGroup: null,
      holidaysAutoSelected: false,
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setShowSliceGuide(false);
    setShowImpactPreview(false);
  };

  // Handle slice value toggle
  const handleSliceValueToggle = (value: string) => {
    setState(s => {
      // Special case: In default state (empty array = all selected visually),
      // clicking to "deselect" should keep all EXCEPT the clicked item
      const isDefaultAllSelected = !s.sliceSelectionExplicit && s.selectedSliceValues.length === 0;

      if (isDefaultAllSelected) {
        // Get all possible combinations from the slice columns
        const allCombinations = new Set<string>();
        if (s.rawData && s.selectedSliceCols.length > 0) {
          s.rawData.forEach(row => {
            const id = s.selectedSliceCols.map(col => String(row[col] || 'Unknown')).join(' | ');
            allCombinations.add(id);
          });
        }
        // Keep all EXCEPT the clicked one (user wants to deselect it)
        const newSelection = Array.from(allCombinations).filter(v => v !== value);
        return {
          ...s,
          selectedSliceValues: newSelection,
          selectedSliceCombinations: newSelection,
          sliceSelectionExplicit: true,
          forecast: null,
          step: 'configure',
        };
      }

      // Normal toggle behavior
      const isCurrentlySelected = s.selectedSliceValues.includes(value);
      return {
        ...s,
        selectedSliceValues: isCurrentlySelected
          ? s.selectedSliceValues.filter(v => v !== value)
          : [...s.selectedSliceValues, value],
        sliceSelectionExplicit: true,
        forecast: null,
        step: 'configure',
      };
    });
  };

  // Handle slice column toggle (supports multiple selection)
  const handleSliceColToggle = (col: string) => {
    setState(s => {
      const currentCols = s.selectedSliceCols;
      const isSelected = currentCols.includes(col);

      let newCols: string[];
      if (isSelected) {
        // Remove the column
        newCols = currentCols.filter(c => c !== col);
      } else {
        // Add the column
        newCols = [...currentCols, col];
      }

      // Calculate new unique combinations for slice values
      let newSliceValues: string[] = [];
      if (newCols.length > 0 && s.rawData) {
        const combinationsSet = new Set<string>();
        s.rawData.forEach(row => {
          const combinedValue = newCols.map(c => String(row[c] || 'Unknown')).join(' | ');
          combinationsSet.add(combinedValue);
        });
        newSliceValues = Array.from(combinationsSet).sort();
      }

      return {
        ...s,
        selectedSliceCols: newCols,
        selectedSliceValues: newSliceValues,
        // Clear old forecast results when slice columns change
        forecast: null,
        step: 'configure',
      };
    });
  };

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-red-600 bg-red-50 border-red-200';
    }
  };

  const getWarningIcon = (level: string) => {
    switch (level) {
      case 'high': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'medium': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      default: return <Info className="w-4 h-4 text-blue-500" />;
    }
  };

  // Helper to get category icon by type
  const getCategoryIcon = (type: string, className: string = "w-4 h-4") => {
    switch (type) {
      case 'holiday': return <Calendar className={className} />;
      case 'pricing': return <DollarSign className={className} />;
      case 'promotion': return <Megaphone className={className} />;
      case 'external': return <Globe className={className} />;
      case 'numeric': return <Hash className={className} />;
      case 'date': return <Calendar className={className} />;
      case 'slice': return <Layers className={className} />;
      case 'target': return <Target className={className} />;
      default: return <FileText className={className} />;
    }
  };

  // Reclassify a column to a different type
  // Also updates covariates: holiday columns are auto-added as covariates
  const reclassifyColumn = (column: string, newType: 'date' | 'slice' | 'holiday' | 'numeric' | 'other') => {
    setState(s => {
      // Determine current type of the column
      const currentType = s.columnOverrides[column] || getCurrentColumnType(column);
      const wasHoliday = currentType === 'holiday';
      const willBeHoliday = newType === 'holiday';

      let newCovariates = [...s.selectedCovariates];

      // If moving TO holiday category, add to covariates (if not already there)
      if (willBeHoliday && !wasHoliday) {
        if (!newCovariates.includes(column)) {
          newCovariates.push(column);
        }
      }
      // If moving FROM holiday category, remove from covariates
      else if (wasHoliday && !willBeHoliday) {
        newCovariates = newCovariates.filter(c => c !== column);
      }

      return {
        ...s,
        columnOverrides: {
          ...s.columnOverrides,
          [column]: newType,
        },
        selectedCovariates: newCovariates,
        reclassifyingColumn: null,
      };
    });
  };

  // Helper to get current column type (before any override)
  const getCurrentColumnType = (column: string): 'date' | 'slice' | 'holiday' | 'numeric' | 'other' => {
    if (!state.rawData) return 'other';
    const { dateColumns, numericColumns, holidayColumns } = detectColumnTypes(state.columns, state.rawData);
    const sliceColumns = state.detectedSlices.map(s => s.column);

    if (holidayColumns.includes(column)) return 'holiday';
    if (dateColumns.includes(column)) return 'date';
    if (sliceColumns.includes(column)) return 'slice';
    if (numericColumns.includes(column)) return 'numeric';
    return 'other';
  };

  // Drag-and-drop handlers for column reclassification
  const handleDragStart = (e: React.DragEvent, column: string) => {
    e.dataTransfer.setData('text/plain', column);
    e.dataTransfer.effectAllowed = 'move';
    setState(s => ({ ...s, draggingColumn: column }));
  };

  const handleDragEnd = () => {
    setState(s => ({ ...s, draggingColumn: null, dragOverGroup: null }));
  };

  const handleDragOver = (e: React.DragEvent, groupType: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (state.dragOverGroup !== groupType) {
      setState(s => ({ ...s, dragOverGroup: groupType }));
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    // Only clear if leaving the group entirely
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const { clientX, clientY } = e;
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
      setState(s => ({ ...s, dragOverGroup: null }));
    }
  };

  const handleDrop = (e: React.DragEvent, groupType: 'date' | 'slice' | 'holiday' | 'numeric' | 'other') => {
    e.preventDefault();
    const column = e.dataTransfer.getData('text/plain');
    if (column) {
      reclassifyColumn(column, groupType);
    }
    setState(s => ({ ...s, draggingColumn: null, dragOverGroup: null }));
  };

  // Get available numeric columns for covariate selection (excluding date, target, and slice columns)
  const getAvailableCovariates = () => {
    const numericColumns = (state.profile?.profile as any)?.numeric_columns || [];
    // Get all slice column names to exclude them
    const sliceColumnNames = state.detectedSlices.map(s => s.column);

    return state.columns.filter(c =>
      c !== state.selectedDateCol &&
      c !== state.selectedTargetCol &&
      !sliceColumnNames.includes(c) && // Exclude slice columns
      (numericColumns.includes(c) || !state.profile)
    );
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6 text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Simple Mode - Autopilot Forecasting</h2>
        <p className="text-gray-600">Upload your data, select columns, and get accurate forecasts automatically</p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-center space-x-4">
          {['upload', 'configure', 'forecasting', 'results'].map((stepName, idx) => {
            const stepIndex = ['upload', 'configure', 'forecasting', 'results'].indexOf(state.step);
            const isComplete = idx < stepIndex;
            const isCurrent = stepName === state.step;

            return (
              <React.Fragment key={stepName}>
                <div className="flex items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    isComplete ? 'bg-green-500 text-white' :
                    isCurrent ? 'bg-blue-600 text-white' :
                    'bg-gray-200 text-gray-500'
                  }`}>
                    {isComplete ? <Check className="w-4 h-4" /> : idx + 1}
                  </div>
                  <span className={`ml-2 text-sm ${isCurrent ? 'text-gray-800 font-medium' : 'text-gray-500'}`}>
                    {stepName === 'upload' ? 'Upload' :
                     stepName === 'configure' ? 'Configure' :
                     stepName === 'forecasting' ? 'Processing' : 'Results'}
                  </span>
                </div>
                {idx < 3 && <div className="w-12 h-0.5 bg-gray-200" />}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Error Display */}
      {state.error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
          <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-red-700 font-medium">Error</p>
            <p className="text-red-600 text-sm">{state.error}</p>
          </div>
          <button onClick={() => setState(s => ({ ...s, error: null }))} className="ml-auto text-red-500 hover:text-red-700">
            <XCircle className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Step 1: Upload with Data Guide */}
      {state.step === 'upload' && (
        <div className="space-y-6">
          {/* Data Format Guide */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <button
              onClick={() => setShowDataGuide(!showDataGuide)}
              className="w-full px-6 py-4 flex items-center justify-between bg-blue-50 hover:bg-blue-100 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <HelpCircle className="w-5 h-5 text-blue-600" />
                <span className="font-medium text-blue-800">What data format do I need?</span>
              </div>
              {showDataGuide ? <ChevronUp className="w-5 h-5 text-blue-600" /> : <ChevronDown className="w-5 h-5 text-blue-600" />}
            </button>

            {showDataGuide && (
              <div className="p-6 space-y-6">
                {/* Time Series Data */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3 mb-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Table className="w-4 h-4 text-blue-600" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-800">{DATA_EXAMPLES.timeseries.title}</h4>
                      <p className="text-sm text-gray-600">{DATA_EXAMPLES.timeseries.description}</p>
                    </div>
                    <span className="ml-auto px-2 py-1 bg-red-100 text-red-700 text-xs rounded font-medium">Required</span>
                  </div>
                  <div className="bg-gray-50 rounded p-3">
                    <div className="text-xs text-gray-500 mb-2 font-medium">Required Columns:</div>
                    <ul className="text-sm text-gray-700 space-y-1 mb-3">
                      {DATA_EXAMPLES.timeseries.columns.map((col, i) => (
                        <li key={i} className="flex items-center">
                          <Check className="w-3 h-3 text-green-500 mr-2" />
                          {col}
                        </li>
                      ))}
                    </ul>
                    <div className="text-xs text-gray-500 mb-1 font-medium">Example:</div>
                    <pre className="text-xs bg-white p-2 rounded border border-gray-200 overflow-x-auto">
                      {DATA_EXAMPLES.timeseries.example}
                    </pre>
                  </div>
                </div>

                {/* Promotions/Features Data */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3 mb-3">
                    <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Layers className="w-4 h-4 text-purple-600" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-800">{DATA_EXAMPLES.promotions.title}</h4>
                      <p className="text-sm text-gray-600">{DATA_EXAMPLES.promotions.description}</p>
                    </div>
                    <span className="ml-auto px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded font-medium">Optional</span>
                  </div>
                  <div className="bg-gray-50 rounded p-3">
                    <div className="text-xs text-gray-500 mb-2 font-medium">Typical Columns:</div>
                    <ul className="text-sm text-gray-700 space-y-1 mb-3">
                      {DATA_EXAMPLES.promotions.columns.map((col, i) => (
                        <li key={i} className="flex items-center">
                          <Check className="w-3 h-3 text-green-500 mr-2" />
                          {col}
                        </li>
                      ))}
                    </ul>
                    <div className="text-xs text-gray-500 mb-1 font-medium">Example:</div>
                    <pre className="text-xs bg-white p-2 rounded border border-gray-200 overflow-x-auto">
                      {DATA_EXAMPLES.promotions.example}
                    </pre>
                  </div>
                </div>

                {/* Minimum Requirements */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-yellow-800">Minimum Data Requirements</h4>
                      <ul className="mt-2 text-sm text-yellow-700 space-y-1">
                        <li>• At least <strong>{MIN_REQUIREMENTS.rows} data points</strong> (rows)</li>
                        <li>• At least <strong>{MIN_REQUIREMENTS.months} months</strong> of historical data</li>
                        <li>• Recommended: {MIN_REQUIREMENTS.recommended_months}+ months for holiday patterns</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Upload Area */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
            <div className="text-center">
              <div className="mx-auto w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-4">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Upload Your Data</h3>
              <p className="text-gray-500 text-sm mb-6">
                Upload a CSV or Excel file with your time series data.<br />
                Include promotions/features in the same file or merge them first.
              </p>

              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileSelect}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={state.isLoading}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors flex items-center mx-auto"
              >
                {state.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Reading file...
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4 mr-2" />
                    Select File
                  </>
                )}
              </button>

              <div className="mt-6 text-xs text-gray-400">
                Supported formats: CSV, Excel (.xlsx, .xls) • Max size: 10MB
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Configure - Column Selection */}
      {state.step === 'configure' && (
        <div className="space-y-6">
          {/* File Info */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-green-50 rounded-lg flex items-center justify-center">
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-800">Data Loaded Successfully</h3>
                  <p className="text-sm text-gray-500">{state.file?.name} • {state.rawData?.length.toLocaleString()} rows • {state.columns.length} columns</p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="text-sm text-gray-500 hover:text-gray-700 flex items-center"
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                Change File
              </button>
            </div>

            {/* Data Preview */}
            {state.rawData && state.rawData.length > 0 && (
              <div className="mt-4 border border-gray-200 rounded-lg overflow-hidden">
                <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 text-xs font-medium text-gray-600">
                  Data Preview (first 5 rows)
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {state.columns.map(col => (
                          <th key={col} className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {state.rawData.slice(0, 5).map((row, idx) => (
                        <tr key={idx}>
                          {state.columns.map(col => (
                            <td key={`${idx}-${col}`} className="px-3 py-2 whitespace-nowrap text-xs text-gray-500">
                              {row[col]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Column Overview for Wide Datasets - with Drag & Drop */}
          {getColumnGroups.length > 0 && state.columns.length > 8 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800 flex items-center">
                  <Layers className="w-5 h-5 mr-2 text-blue-600" />
                  Column Overview
                  <span className="ml-2 text-xs font-normal text-gray-500">({state.columns.length} columns detected)</span>
                </h3>
                <div className="flex items-center space-x-3">
                  {Object.keys(state.columnOverrides).length > 0 && (
                    <button
                      onClick={() => setState(s => ({ ...s, columnOverrides: {} }))}
                      className="text-xs text-gray-500 hover:text-gray-700 flex items-center"
                    >
                      <RefreshCw className="w-3 h-3 mr-1" />
                      Reset Classifications
                    </button>
                  )}
                  <button
                    onClick={() => setState(s => ({
                      ...s,
                      expandedColumnGroups: s.expandedColumnGroups.length === getColumnGroups.length
                        ? []
                        : getColumnGroups.map(g => g.type)
                    }))}
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    {state.expandedColumnGroups.length === getColumnGroups.length ? 'Collapse All' : 'Expand All'}
                  </button>
                </div>
              </div>

              <p className="text-sm text-gray-500 mb-2">
                <span className="font-medium text-gray-700">Drag and drop</span> column chips between categories to reclassify them.
                {Object.keys(state.columnOverrides).length > 0 && (
                  <span className="ml-2 text-blue-600">({Object.keys(state.columnOverrides).length} manually reclassified)</span>
                )}
              </p>

              {/* Color Legend */}
              <div className="flex flex-wrap items-center gap-3 mb-4 text-xs text-gray-500">
                <span className="font-medium text-gray-600">Legend:</span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-full ring-2 ring-blue-400 bg-white"></span>
                  <span>Date</span>
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-full ring-2 ring-green-400 bg-white"></span>
                  <span>Target</span>
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-full ring-2 ring-purple-400 bg-white"></span>
                  <span>Covariate</span>
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-full bg-yellow-100 border border-yellow-300"></span>
                  <span>Reclassified</span>
                </span>
              </div>

              <div className="grid md:grid-cols-2 gap-3">
                {getColumnGroups.map(group => {
                  const isExpanded = state.expandedColumnGroups.includes(group.type);
                  const isDragOver = state.dragOverGroup === group.type;

                  return (
                    <div
                      key={group.type}
                      className={`rounded-lg border-2 overflow-hidden transition-all ${group.bgColor} ${
                        isDragOver
                          ? 'border-blue-400 ring-2 ring-blue-200 scale-[1.02]'
                          : 'border-gray-200'
                      }`}
                      onDragOver={(e) => handleDragOver(e, group.type)}
                      onDragLeave={handleDragLeave}
                      onDrop={(e) => handleDrop(e, group.type as 'date' | 'slice' | 'holiday' | 'numeric' | 'other')}
                    >
                      <button
                        onClick={() => setState(s => ({
                          ...s,
                          expandedColumnGroups: isExpanded
                            ? s.expandedColumnGroups.filter(g => g !== group.type)
                            : [...s.expandedColumnGroups, group.type]
                        }))}
                        className="w-full px-3 py-2 flex items-center justify-between hover:bg-white/50 transition-colors"
                      >
                        <div className="flex items-center space-x-2">
                          {getCategoryIcon(group.type, `w-4 h-4 ${group.color}`)}
                          <span className={`font-medium text-sm ${group.color}`}>{group.label}</span>
                          <span className="text-xs text-gray-500 bg-white/50 px-1.5 py-0.5 rounded">
                            {group.columns.length}
                          </span>
                        </div>
                        {isExpanded ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
                      </button>

                      {isExpanded && (
                        <div className={`px-3 pb-3 pt-1 bg-white/30 min-h-[60px] ${isDragOver ? 'bg-blue-50/50' : ''}`}>
                          <p className="text-xs text-gray-500 mb-2">{group.description}</p>
                          <div className="flex flex-wrap gap-1.5">
                            {group.columns.map(col => {
                              const isOverridden = state.columnOverrides[col] !== undefined;
                              const isDragging = state.draggingColumn === col;
                              const isNumericGroup = group.type === 'numeric';
                              const isTarget = col === state.selectedTargetCol;
                              const isDate = col === state.selectedDateCol;
                              const isDateGroup = group.type === 'date';

                              const isCovariate = state.selectedCovariates.includes(col);

                              // Handle click for numeric columns (toggle covariate) or date columns
                              const handleChipClick = () => {
                                if (isNumericGroup && !isTarget) {
                                  // Toggle covariate selection
                                  setState(s => ({
                                    ...s,
                                    selectedCovariates: isCovariate
                                      ? s.selectedCovariates.filter(c => c !== col)
                                      : [...s.selectedCovariates, col],
                                  }));
                                } else if (isDateGroup && !isDate) {
                                  setState(s => ({ ...s, selectedDateCol: col }));
                                }
                              };

                              // Handle double-click for numeric columns (set as target)
                              const handleChipDoubleClick = () => {
                                if (isNumericGroup && !isTarget) {
                                  setState(s => ({
                                    ...s,
                                    selectedTargetCol: col,
                                    selectedCovariates: s.selectedCovariates.filter(c => c !== col),
                                  }));
                                }
                              };

                              return (
                                <span
                                  key={col}
                                  draggable
                                  onDragStart={(e) => handleDragStart(e, col)}
                                  onDragEnd={handleDragEnd}
                                  onClick={handleChipClick}
                                  onDoubleClick={handleChipDoubleClick}
                                  className={`px-2 py-1 text-xs rounded-full select-none transition-all ${
                                    isDragging
                                      ? 'opacity-50 scale-95 bg-gray-200 cursor-grabbing'
                                      : isDate
                                      ? 'ring-2 ring-blue-400 text-blue-700 bg-white cursor-pointer'
                                      : isTarget
                                      ? 'ring-2 ring-green-400 text-green-700 bg-white cursor-default'
                                      : isCovariate
                                      ? 'ring-2 ring-purple-400 text-purple-700 bg-white cursor-pointer hover:bg-purple-50'
                                      : isOverridden
                                      ? 'bg-yellow-100 text-yellow-800 border border-yellow-300 cursor-grab hover:bg-yellow-200'
                                      : isNumericGroup
                                      ? 'bg-white text-gray-600 hover:bg-purple-50 hover:ring-1 hover:ring-purple-300 cursor-pointer'
                                      : isDateGroup
                                      ? 'bg-white text-gray-600 hover:bg-blue-50 hover:ring-1 hover:ring-blue-300 cursor-pointer'
                                      : 'bg-white text-gray-600 hover:bg-gray-100 cursor-grab'
                                  }`}
                                  title={
                                    isTarget ? 'Current target column (double-click another to change)'
                                    : isDate ? 'Current date column'
                                    : isCovariate ? 'Covariate - click to remove, double-click to set as target'
                                    : isNumericGroup ? 'Click to add as covariate, double-click to set as target'
                                    : isDateGroup ? 'Click to set as date column'
                                    : isOverridden ? 'Manually reclassified - drag to move'
                                    : 'Drag to reclassify'
                                  }
                                >
                                  {col}
                                  {isDate && <Calendar className="w-3 h-3 inline ml-1" />}
                                  {isTarget && <Target className="w-3 h-3 inline ml-1" />}
                                  {isOverridden && !isDate && !isTarget && (
                                    <span className="ml-1 text-yellow-600">*</span>
                                  )}
                                </span>
                              );
                            })}
                            {group.columns.length === 0 && (
                              <span className="text-xs text-gray-400 italic py-2">Drop columns here</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Drag hint */}
              {state.draggingColumn && (
                <div className="mt-3 p-2 bg-blue-50 rounded-lg border border-blue-200 text-center">
                  <p className="text-xs text-blue-700">
                    Drop <strong>{state.draggingColumn}</strong> into a category to reclassify it
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Validation Warnings */}
          {state.validation && state.validation.warnings.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-yellow-800">Warnings</h4>
                  <ul className="mt-1 text-sm text-yellow-700 space-y-1">
                    {state.validation.warnings.map((w, i) => (
                      <li key={i}>• {w}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* AI Guidance: Slice Detection */}
          {getEffectiveSegments.length > 0 && (
            <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border border-purple-200 overflow-hidden">
              <button
                onClick={() => setShowSliceGuide(!showSliceGuide)}
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/50 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-purple-600" />
                  </div>
                  <div className="text-left">
                    <span className="font-semibold text-purple-800 flex items-center">
                      AI Detected {getEffectiveSegments.length} Data Segment{getEffectiveSegments.length > 1 ? 's' : ''}
                      <span className="ml-2 px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full">
                        Important
                      </span>
                    </span>
                    <p className="text-sm text-purple-600">Click to understand how segments affect your forecast</p>
                  </div>
                </div>
                {showSliceGuide ? <ChevronUp className="w-5 h-5 text-purple-600" /> : <ChevronDown className="w-5 h-5 text-purple-600" />}
              </button>

              {showSliceGuide && (
                <div className="p-6 border-t border-purple-200 bg-white/70">
                  {/* AI Explanation */}
                  <div className="mb-6 p-4 bg-white rounded-lg border border-gray-200">
                    <div className="flex items-start space-x-3">
                      <Lightbulb className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-semibold text-gray-800 mb-2">{AI_GUIDANCE.sliceExplanation.title}</h4>
                        <p className="text-sm text-gray-600 mb-3">{AI_GUIDANCE.sliceExplanation.intro}</p>
                        <p className="text-sm text-gray-600 mb-2">{AI_GUIDANCE.sliceExplanation.description}</p>
                        <ul className="space-y-2">
                          {AI_GUIDANCE.sliceExplanation.examples.map((ex, i) => (
                            <li key={i} className="flex items-start text-sm">
                              <span className="text-purple-500 mr-2">•</span>
                              <span>
                                <span className="font-semibold text-gray-700">{ex.type}:</span>{' '}
                                <span className="text-gray-600">{ex.values}</span>{' '}
                                <span className="text-gray-500">— {ex.impact}</span>
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>

                  {/* Detected Slices List */}
                  <div className="mb-6">
                    <h4 className="font-medium text-gray-700 mb-2">Detected Segments in Your Data:</h4>
                    <p className="text-sm text-gray-500 mb-3">
                      Select one or more columns to create slice combinations. For example, selecting both "Region" and "Product" will create slices like "North | Electronics".
                    </p>
                    <div className="grid gap-3">
                      {getEffectiveSegments.map((slice) => {
                        const config = SLICE_TYPE_CONFIG[slice.suggestedType] || SLICE_TYPE_CONFIG.other;
                        const IconComponent = config.icon;
                        const isSelected = state.selectedSliceCols.includes(slice.column);
                        const isOverridden = state.columnOverrides[slice.column] === 'slice';
                        return (
                          <div
                            key={slice.column}
                            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                              isSelected
                                ? 'border-purple-500 bg-purple-50'
                                : 'border-gray-200 bg-white hover:border-purple-300'
                            }`}
                            onClick={() => handleSliceColToggle(slice.column)}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <div className={`w-6 h-6 rounded border-2 flex items-center justify-center ${
                                  isSelected ? 'bg-purple-500 border-purple-500' : 'border-gray-300 bg-white'
                                }`}>
                                  {isSelected && <Check className="w-4 h-4 text-white" />}
                                </div>
                                <div className={`w-8 h-8 ${config.bgColor} rounded-lg flex items-center justify-center`}>
                                  <IconComponent className={`w-4 h-4 ${config.color}`} />
                                </div>
                                <div>
                                  <span className="font-medium text-gray-800">{slice.column}</span>
                                  <span className="ml-2 text-xs text-gray-500 capitalize">({slice.suggestedType})</span>
                                </div>
                              </div>
                              <div className="text-right">
                                <span className="text-sm font-medium text-gray-700">{slice.count} values</span>
                                <div className="text-xs text-gray-500">
                                  {slice.sampleValues.slice(0, 3).join(', ')}
                                  {slice.count > 3 && '...'}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Combined Slice Info (when multiple columns selected) */}
                    {state.selectedSliceCols.length > 0 && getCombinedSliceInfo && (
                      <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <h5 className="font-medium text-purple-800 mb-2 flex items-center">
                          <Layers className="w-4 h-4 mr-2" />
                          {state.selectedSliceCols.length === 1
                            ? `Selected: ${state.selectedSliceCols[0]}`
                            : `Combined Slices: ${getCombinedSliceInfo.combinedKey}`}
                        </h5>
                        <p className="text-sm text-purple-700 mb-2">
                          {getCombinedSliceInfo.count} unique {state.selectedSliceCols.length === 1 ? 'values' : 'combinations'}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {getCombinedSliceInfo.sampleCombinations.map((combo, i) => (
                            <span key={i} className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                              {combo}
                            </span>
                          ))}
                          {getCombinedSliceInfo.count > 5 && (
                            <span className="text-xs text-purple-600 px-2 py-1">
                              +{getCombinedSliceInfo.count - 5} more...
                            </span>
                          )}
                        </div>
                        {state.selectedSliceCols.length > 1 && (
                          <p className="text-xs text-purple-600 mt-2">
                            Each combination will be forecasted separately when using "By Slice" mode.
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Forecast Mode Selection */}
                  <div className="mb-4">
                    <h4 className="font-medium text-gray-700 mb-3">Choose Forecasting Approach:</h4>
                    <div className="grid md:grid-cols-2 gap-4">
                      {/* Aggregate Mode */}
                      <div
                        className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                          state.forecastMode === 'aggregate'
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 bg-white hover:border-blue-300'
                        }`}
                        onClick={() => setState(s => ({ ...s, forecastMode: 'aggregate' }))}
                      >
                        <div className="flex items-center space-x-3 mb-3">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            state.forecastMode === 'aggregate' ? 'bg-blue-100' : 'bg-gray-100'
                          }`}>
                            <BarChart3 className={`w-5 h-5 ${state.forecastMode === 'aggregate' ? 'text-blue-600' : 'text-gray-500'}`} />
                          </div>
                          <div>
                            <h5 className="font-semibold text-gray-800">{AI_GUIDANCE.aggregateMode.title}</h5>
                            <p className="text-xs text-gray-500">{AI_GUIDANCE.aggregateMode.consequence}</p>
                          </div>
                        </div>
                        <ul className="text-sm text-gray-600 space-y-1 mb-3">
                          {AI_GUIDANCE.aggregateMode.details.map((d, i) => (
                            <li key={i} className="flex items-start">
                              <span className="text-blue-500 mr-2">•</span>
                              {d}
                            </li>
                          ))}
                        </ul>
                        <div className={`text-xs p-2 rounded ${state.forecastMode === 'aggregate' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}>
                          <Lightbulb className="w-3 h-3 inline mr-1" />
                          {AI_GUIDANCE.aggregateMode.whenToUse}
                        </div>
                      </div>

                      {/* By Slice Mode */}
                      <div
                        className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                          state.forecastMode === 'by_slice'
                            ? 'border-purple-500 bg-purple-50'
                            : 'border-gray-200 bg-white hover:border-purple-300'
                        }`}
                        onClick={() => setState(s => ({ ...s, forecastMode: 'by_slice' }))}
                      >
                        <div className="flex items-center space-x-3 mb-3">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            state.forecastMode === 'by_slice' ? 'bg-purple-100' : 'bg-gray-100'
                          }`}>
                            <GitBranch className={`w-5 h-5 ${state.forecastMode === 'by_slice' ? 'text-purple-600' : 'text-gray-500'}`} />
                          </div>
                          <div>
                            <h5 className="font-semibold text-gray-800">{AI_GUIDANCE.sliceMode.title}</h5>
                            <p className="text-xs text-gray-500">{AI_GUIDANCE.sliceMode.consequence}</p>
                          </div>
                        </div>
                        <ul className="text-sm text-gray-600 space-y-1 mb-3">
                          {AI_GUIDANCE.sliceMode.details.map((d, i) => (
                            <li key={i} className="flex items-start">
                              <span className="text-purple-500 mr-2">•</span>
                              {d}
                            </li>
                          ))}
                        </ul>
                        <div className={`text-xs p-2 rounded ${state.forecastMode === 'by_slice' ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-600'}`}>
                          <Lightbulb className="w-3 h-3 inline mr-1" />
                          {AI_GUIDANCE.sliceMode.whenToUse}
                        </div>
                      </div>
                    </div>

                    {/* Detailed Example Explanation - DYNAMIC based on actual data */}
                    <div className="mt-4 p-4 bg-white rounded-lg border border-gray-200">
                      <h5 className="font-medium text-gray-800 mb-3 flex items-center">
                        <Info className="w-4 h-4 mr-2 text-blue-500" />
                        {state.forecastMode === 'aggregate'
                          ? `Example: Your ${state.selectedTargetCol || 'Data'} Aggregation`
                          : `Example: Your ${getDynamicExamples?.sliceColumnLabel || 'Segmented'} Forecasts`}
                      </h5>

                      {state.forecastMode === 'aggregate' ? (
                        <div className="space-y-4">
                          {/* Before: Raw Data - DYNAMIC */}
                          {getDynamicExamples && getDynamicExamples.beforeData.length > 0 ? (
                            <>
                              <div>
                                <p className="text-xs font-medium text-gray-500 uppercase mb-2">Your Raw Data (Sample):</p>
                                <div className="overflow-x-auto">
                                  <table className="min-w-full text-xs border border-gray-200 rounded">
                                    <thead className="bg-gray-50">
                                      <tr>
                                        <th className="px-2 py-1 text-left text-gray-600 border-b">{getDynamicExamples.sliceColumnLabel}</th>
                                        <th className="px-2 py-1 text-left text-gray-600 border-b">{getDynamicExamples.dateColumn}</th>
                                        <th className="px-2 py-1 text-right text-gray-600 border-b">{getDynamicExamples.targetColumn}</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {getDynamicExamples.beforeData.slice(0, 4).map((row, i) => (
                                        <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                          <td className="px-2 py-1 border-b">{row.slice}</td>
                                          <td className="px-2 py-1 border-b">{row.date}</td>
                                          <td className="px-2 py-1 text-right border-b">{getDynamicExamples.formatValue(row.value)}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>

                              {/* Arrow */}
                              <div className="flex items-center justify-center">
                                <div className="flex items-center space-x-2 px-4 py-2 bg-blue-50 rounded-lg">
                                  <span className="text-xs text-blue-600 font-medium">Aggregation</span>
                                  <ArrowRight className="w-4 h-4 text-blue-500" />
                                  <span className="text-xs text-blue-600">Sum by {getDynamicExamples.dateColumn}</span>
                                </div>
                              </div>

                              {/* After: Aggregated Data */}
                              <div>
                                <p className="text-xs font-medium text-gray-500 uppercase mb-2">What the Model Sees:</p>
                                <div className="overflow-x-auto">
                                  <table className="min-w-full text-xs border border-blue-200 rounded bg-blue-50">
                                    <thead className="bg-blue-100">
                                      <tr>
                                        <th className="px-2 py-1 text-left text-blue-700 border-b border-blue-200">{getDynamicExamples.dateColumn}</th>
                                        <th className="px-2 py-1 text-right text-blue-700 border-b border-blue-200">Total {getDynamicExamples.targetColumn}</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {getDynamicExamples.afterData.map((row, i) => (
                                        <tr key={i}>
                                          <td className="px-2 py-1 border-b border-blue-200 text-blue-800">{row.date}</td>
                                          <td className="px-2 py-1 text-right border-b border-blue-200 font-medium text-blue-800">{getDynamicExamples.formatValue(row.total)}</td>
                                        </tr>
                                      ))}
                                      <tr className="bg-blue-100">
                                        <td className="px-2 py-1 text-blue-700">Next Period</td>
                                        <td className="px-2 py-1 text-right font-medium text-blue-700">$??? (Forecast)</td>
                                      </tr>
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            </>
                          ) : (
                            <p className="text-sm text-gray-500 italic">Select slice columns to see how aggregation works with your data.</p>
                          )}

                          {/* Explanation */}
                          <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-sm text-gray-700">
                              Data is summed across all {getDynamicExamples?.sliceColumnLabel || 'segments'}. The model sees total {state.selectedTargetCol || 'values'} and predicts future totals. You get ONE forecast number per period.
                            </p>
                            <p className="text-xs text-green-600 mt-2 font-medium">
                              {getDynamicExamples?.aggregateUseCase || AI_GUIDANCE.aggregateMode.example.useCase}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          {/* Slice Mode Explanation - DYNAMIC */}
                          <div className="p-3 bg-purple-50 rounded-lg">
                            <p className="text-sm text-gray-700">
                              Each {getDynamicExamples?.sliceColumnLabel || 'segment'} gets its own model trained on its own data. Each learns its unique patterns (seasonality, trends, holiday effects).
                            </p>
                          </div>

                          {/* Visual: Separate Models - DYNAMIC */}
                          <div className="grid grid-cols-2 gap-3">
                            {(getDynamicExamples?.sliceExamples || []).slice(0, 4).map((item, i) => (
                              <div key={i} className="p-3 bg-white border-2 border-purple-200 rounded-lg">
                                <div className="flex items-center space-x-2 mb-2">
                                  <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center">
                                    <Layers className="w-3 h-3 text-purple-600" />
                                  </div>
                                  <span className="font-medium text-purple-800 text-xs truncate">{item.slice}</span>
                                </div>
                                <p className="text-xs text-gray-600">{item.description}</p>
                                <div className="mt-2 h-8 bg-gradient-to-r from-purple-100 to-purple-200 rounded flex items-center justify-center">
                                  <span className="text-xs text-purple-700">Separate Forecast</span>
                                </div>
                              </div>
                            ))}
                          </div>

                          <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-green-600 font-medium">
                              {getDynamicExamples?.sliceUseCase || AI_GUIDANCE.sliceMode.example.useCase}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Enhanced Slice Values Selection (when by_slice mode) */}
                  {state.forecastMode === 'by_slice' && state.selectedSliceCols.length > 0 && getCombinedSliceInfo && (
                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h5 className="font-medium text-purple-800">
                            Select {state.selectedSliceCols.length === 1
                              ? state.selectedSliceCols[0]
                              : 'Segment Combinations'} to Forecast
                          </h5>
                          <p className="text-xs text-purple-600">
                            Each selection trains a separate forecast model
                          </p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => selectAllSliceCombinations(true)}
                            className="px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded hover:bg-purple-200 transition-colors"
                          >
                            Select All
                          </button>
                          <button
                            onClick={() => selectAllSliceCombinations(false)}
                            className="px-2 py-1 text-xs bg-white text-gray-600 border border-gray-200 rounded hover:bg-gray-100 transition-colors"
                          >
                            Clear All
                          </button>
                        </div>
                      </div>

                      {/* Search Filter for many slices */}
                      {getCombinedSliceInfo.count > 10 && (
                        <div className="mb-3 relative">
                          <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                          <input
                            type="text"
                            placeholder="Search segments..."
                            value={state.sliceSearchQuery}
                            onChange={(e) => setState(s => ({ ...s, sliceSearchQuery: e.target.value }))}
                            className="w-full pl-9 pr-4 py-2 text-sm border border-purple-200 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white"
                          />
                        </div>
                      )}

                      {/* Slice Combinations with Stats */}
                      {getSliceCombinations.length > 0 ? (
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                          {filteredSliceCombinations.map(combo => {
                            // If user hasn't explicitly modified selection, treat empty as "all selected"
                            // If user has explicitly cleared, empty means nothing selected
                            const isSelected = state.selectedSliceValues.includes(combo.id) ||
                              (!state.sliceSelectionExplicit && state.selectedSliceValues.length === 0);

                            return (
                              <div
                                key={combo.id}
                                onClick={() => handleSliceValueToggle(combo.id)}
                                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                                  isSelected
                                    ? 'bg-purple-100 border-purple-300'
                                    : 'bg-white border-gray-200 hover:border-purple-200 hover:bg-purple-50'
                                }`}
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center space-x-2">
                                    <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                                      isSelected ? 'bg-purple-500 border-purple-500' : 'border-gray-300'
                                    }`}>
                                      {isSelected && <Check className="w-3 h-3 text-white" />}
                                    </div>
                                    <span className={`font-medium text-sm ${isSelected ? 'text-purple-800' : 'text-gray-700'}`}>
                                      {combo.displayName}
                                    </span>
                                  </div>
                                  <div className="flex items-center space-x-3 text-xs text-gray-500">
                                    <span title="Data Points">
                                      <Table className="w-3 h-3 inline mr-1" />
                                      {combo.rowCount.toLocaleString()}
                                    </span>
                                    {state.selectedTargetCol && (
                                      <>
                                        <span title={`Total ${state.selectedTargetCol}`}>
                                          <TrendingUp className="w-3 h-3 inline mr-1" />
                                          {combo.targetSum >= 1000000
                                            ? `${(combo.targetSum / 1000000).toFixed(1)}M`
                                            : combo.targetSum >= 1000
                                            ? `${(combo.targetSum / 1000).toFixed(1)}K`
                                            : combo.targetSum.toFixed(0)}
                                        </span>
                                        <span title={`Avg ${state.selectedTargetCol}`} className="text-gray-400">
                                          avg: {combo.targetAvg >= 1000
                                            ? `${(combo.targetAvg / 1000).toFixed(1)}K`
                                            : combo.targetAvg.toFixed(0)}
                                        </span>
                                      </>
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
                          {getCombinedSliceInfo.uniqueCombinations.map(val => (
                            <button
                              key={val}
                              onClick={() => handleSliceValueToggle(val)}
                              className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                                state.selectedSliceValues.includes(val)
                                  ? 'bg-purple-100 border-purple-300 text-purple-700'
                                  : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                              }`}
                            >
                              {state.selectedSliceValues.includes(val) && <Check className="w-3 h-3 inline mr-1" />}
                              {val}
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Summary */}
                      <div className="mt-3 flex items-center justify-between text-xs">
                        <span className={state.sliceSelectionExplicit && state.selectedSliceValues.length === 0 ? 'text-amber-600' : 'text-purple-600'}>
                          {state.selectedSliceValues.length > 0
                            ? `${state.selectedSliceValues.length} of ${getCombinedSliceInfo.count} selected`
                            : state.sliceSelectionExplicit
                            ? `0 of ${getCombinedSliceInfo.count} selected - click Select All or choose segments`
                            : `All ${getCombinedSliceInfo.count} segments selected (default)`}
                        </span>
                        {state.selectedSliceValues.length > 5 && (
                          <span className="text-amber-600 flex items-center">
                            <AlertTriangle className="w-3 h-3 mr-1" />
                            Many segments may take longer to process
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Column Selection with AI Guidance */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-blue-600" />
              Configure Your Forecast
            </h3>

            <div className="space-y-6">
              {/* Date Column */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Calendar className="w-4 h-4 inline mr-1" />
                  Date/Time Column <span className="text-red-500">*</span>
                </label>
                <select
                  value={state.selectedDateCol}
                  onChange={(e) => setState(s => ({ ...s, selectedDateCol: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                >
                  <option value="">Select date column...</option>
                  {state.columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                <div className="mt-2 p-2 bg-blue-50 rounded border border-blue-100">
                  <p className="text-xs text-blue-700">
                    <Lightbulb className="w-3 h-3 inline mr-1" />
                    This determines how your data is ordered over time. The system will detect if it's daily, weekly, or monthly data.
                  </p>
                </div>
              </div>

              {/* Target Column with Guidance */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <TrendingUp className="w-4 h-4 inline mr-1" />
                  Target Column (What to Forecast) <span className="text-red-500">*</span>
                </label>
                <select
                  value={state.selectedTargetCol}
                  onChange={(e) => {
                    const newTarget = e.target.value;
                    setState(s => ({
                      ...s,
                      selectedTargetCol: newTarget,
                      // Remove the new target from covariates if it was selected
                      selectedCovariates: s.selectedCovariates.filter(c => c !== newTarget)
                    }));
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                >
                  <option value="">Select target column...</option>
                  {state.columns.filter(c => c !== state.selectedDateCol).map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                {state.selectedTargetCol && (
                  <div className="mt-2 p-2 bg-green-50 rounded border border-green-100">
                    <p className="text-xs text-green-700">
                      <Lightbulb className="w-3 h-3 inline mr-1" />
                      {state.selectedTargetCol.toLowerCase().includes('revenue')
                        ? 'Revenue forecasting: Great for budgeting, financial planning, and investor reporting.'
                        : state.selectedTargetCol.toLowerCase().includes('sales') || state.selectedTargetCol.toLowerCase().includes('units')
                        ? 'Sales/Units forecasting: Useful for inventory planning, staffing, and supply chain decisions.'
                        : state.selectedTargetCol.toLowerCase().includes('cost') || state.selectedTargetCol.toLowerCase().includes('expense')
                        ? 'Cost forecasting: Helps with budget allocation and cost control planning.'
                        : `The model will predict future values of "${state.selectedTargetCol}" based on historical patterns.`}
                    </p>
                  </div>
                )}
              </div>

              {/* Covariates with Enhanced Categorized Selection */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    <Layers className="w-4 h-4 inline mr-1" />
                    Features / Drivers (Optional)
                  </label>
                  <span className="text-xs px-2 py-1 bg-gray-200 text-gray-600 rounded">
                    {state.selectedCovariates.length} selected
                  </span>
                </div>

                {/* Compact Guidance Box */}
                <div className="mb-3 p-2 bg-purple-50 rounded-lg border border-purple-100">
                  <div className="flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-purple-600 flex-shrink-0" />
                    <p className="text-xs text-purple-700">
                      Select features that influence your target. Use category headers to bulk select/deselect.
                    </p>
                  </div>
                </div>

                {/* Search Filter */}
                {getAvailableCovariates().length > 10 && (
                  <div className="mb-3 relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search features..."
                      value={state.covariateSearchQuery}
                      onChange={(e) => setState(s => ({ ...s, covariateSearchQuery: e.target.value }))}
                      className="w-full pl-9 pr-4 py-2 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                  </div>
                )}

                {/* Categorized Covariate Selection */}
                {getCovariateCategories.length > 0 ? (
                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {getCovariateCategories.map(category => {
                      const filteredColumns = state.covariateSearchQuery.trim()
                        ? category.columns.filter(c => c.toLowerCase().includes(state.covariateSearchQuery.toLowerCase()))
                        : category.columns;

                      if (filteredColumns.length === 0) return null;

                      const selectedInCategory = filteredColumns.filter(c => state.selectedCovariates.includes(c)).length;
                      const allSelected = selectedInCategory === filteredColumns.length;
                      const isExpanded = state.expandedCovariateGroups.includes(category.type);

                      return (
                        <div key={category.type} className="border border-gray-200 rounded-lg bg-white overflow-hidden">
                          {/* Category Header - Clickable to expand/collapse */}
                          <div
                            className={`flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50 ${category.color}`}
                            onClick={() => setState(s => ({
                              ...s,
                              expandedCovariateGroups: isExpanded
                                ? s.expandedCovariateGroups.filter(g => g !== category.type)
                                : [...s.expandedCovariateGroups, category.type]
                            }))}
                          >
                            <div className="flex items-center space-x-2">
                              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                              {getCategoryIcon(category.type, `w-4 h-4 ${category.color}`)}
                              <span className="font-medium text-sm text-gray-800">{category.label}</span>
                              <span className="text-xs text-gray-500">({selectedInCategory}/{filteredColumns.length})</span>
                            </div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleCovariateCategory(category);
                              }}
                              className={`px-2 py-1 text-xs rounded transition-colors ${
                                allSelected
                                  ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                              }`}
                            >
                              {allSelected ? 'Deselect All' : 'Select All'}
                            </button>
                          </div>

                          {/* Category Items */}
                          {isExpanded && (
                            <div className="px-3 pb-3 pt-1">
                              <p className="text-xs text-gray-500 mb-2">{category.description}</p>
                              <div className="flex flex-wrap gap-2">
                                {filteredColumns.map(col => (
                                  <button
                                    key={col}
                                    onClick={() => handleCovariateToggle(col)}
                                    className={`px-2 py-1 text-xs rounded-lg border transition-colors ${
                                      state.selectedCovariates.includes(col)
                                        ? 'bg-purple-100 border-purple-300 text-purple-700'
                                        : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-100'
                                    }`}
                                  >
                                    {state.selectedCovariates.includes(col) && <Check className="w-3 h-3 inline mr-1" />}
                                    {col}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {getAvailableCovariates().map(col => (
                      <button
                        key={col}
                        onClick={() => handleCovariateToggle(col)}
                        className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                          state.selectedCovariates.includes(col)
                            ? 'bg-purple-100 border-purple-300 text-purple-700'
                            : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-100'
                        }`}
                      >
                        {state.selectedCovariates.includes(col) && <Check className="w-3 h-3 inline mr-1" />}
                        {col}
                      </button>
                    ))}
                    {getAvailableCovariates().length === 0 && (
                      <span className="text-sm text-gray-400 italic">No additional numeric columns available</span>
                    )}
                  </div>
                )}

                {/* Impact Summary */}
                {state.selectedCovariates.length > 0 && (
                  <div className="mt-3 p-2 bg-amber-50 rounded border border-amber-100">
                    <p className="text-xs text-amber-700">
                      <AlertTriangle className="w-3 h-3 inline mr-1" />
                      {state.selectedCovariates.length <= 3
                        ? `Good selection. ${state.selectedCovariates.length} feature(s) will be used to improve forecast accuracy.`
                        : state.selectedCovariates.length <= 5
                        ? `${state.selectedCovariates.length} features selected. This is reasonable but consider if all are truly predictive.`
                        : `Many features (${state.selectedCovariates.length}) selected. Risk of overfitting - consider reducing to the most important ones.`}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Profile Info (if available) */}
          {state.profile && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                <Info className="w-5 h-5 mr-2 text-blue-600" />
                Auto-Detected Settings
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Frequency</div>
                  <div className="font-semibold text-gray-800 capitalize">
                    {state.profile.profile.frequency}
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">History</div>
                  <div className="font-semibold text-gray-800">
                    {state.profile.profile.history_months.toFixed(0)} months
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Data Quality</div>
                  <div className={`font-semibold ${
                    state.profile.profile.data_quality_score >= 80 ? 'text-green-600' :
                    state.profile.profile.data_quality_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {state.profile.profile.data_quality_score.toFixed(0)}/100
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Date Range</div>
                  <div className="font-semibold text-gray-800 text-xs">
                    {state.profile.profile.date_range[0]} to {state.profile.profile.date_range[1]}
                  </div>
                </div>
              </div>

              {/* Warnings from profile */}
              {state.profile.warnings.length > 0 && (
                <div className="mt-4 space-y-2">
                  {state.profile.warnings.map((warning, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border flex items-start space-x-2 ${
                        warning.level === 'high' ? 'bg-red-50 border-red-200' :
                        warning.level === 'medium' ? 'bg-yellow-50 border-yellow-200' :
                        'bg-blue-50 border-blue-200'
                      }`}
                    >
                      {getWarningIcon(warning.level)}
                      <div>
                        <p className="text-sm font-medium text-gray-800">{warning.message}</p>
                        <p className="text-xs text-gray-600 mt-0.5">{warning.recommendation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Impact Preview - What Will Happen */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200 overflow-hidden">
            <button
              onClick={() => setShowImpactPreview(!showImpactPreview)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/50 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Zap className="w-5 h-5 text-green-600" />
                </div>
                <div className="text-left">
                  <span className="font-semibold text-green-800">Review Your Configuration</span>
                  <p className="text-sm text-green-600">See what will happen when you run the forecast</p>
                </div>
              </div>
              {showImpactPreview ? <ChevronUp className="w-5 h-5 text-green-600" /> : <ChevronDown className="w-5 h-5 text-green-600" />}
            </button>

            {showImpactPreview && (
              <div className="p-6 border-t border-green-200 bg-white/70">
                {/* Configuration Summary */}
                <div className="mb-6">
                  <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                    <Target className="w-4 h-4 mr-2 text-green-600" />
                    Your Configuration Summary
                  </h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-3 bg-white rounded-lg border border-gray-200">
                      <div className="text-xs text-gray-500 uppercase mb-1">Target to Forecast</div>
                      <div className="font-medium text-gray-800">{state.selectedTargetCol || 'Not selected'}</div>
                      {state.selectedTargetCol && (
                        <p className="text-xs text-gray-500 mt-1">
                          {state.selectedTargetCol.toLowerCase().includes('revenue')
                            ? 'Revenue forecasts help with budgeting and financial planning'
                            : state.selectedTargetCol.toLowerCase().includes('sales')
                            ? 'Sales forecasts help with inventory and capacity planning'
                            : 'Numeric values will be predicted for future periods'}
                        </p>
                      )}
                    </div>
                    <div className="p-3 bg-white rounded-lg border border-gray-200">
                      <div className="text-xs text-gray-500 uppercase mb-1">Features Included</div>
                      {(() => {
                        // Filter out target column from covariates for display
                        const validCovariates = state.selectedCovariates.filter(c => c !== state.selectedTargetCol);
                        return (
                          <>
                            <div className="font-medium text-gray-800">
                              {validCovariates.length} covariate{validCovariates.length !== 1 ? 's' : ''}
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                              {validCovariates.length === 0
                                ? 'Simple forecast using historical patterns only'
                                : `Including: ${validCovariates.slice(0, 3).join(', ')}${validCovariates.length > 3 ? '...' : ''}`}
                            </p>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                </div>

                {/* What Will Happen */}
                <div className="mb-6">
                  <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                    <ArrowRight className="w-4 h-4 mr-2 text-green-600" />
                    What Will Happen
                  </h4>
                  <div className="p-4 bg-white rounded-lg border border-gray-200">
                    <ul className="space-y-3">
                      <li className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-medium text-blue-600">1</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-800">Data Analysis</span>
                          <p className="text-sm text-gray-600">
                            System will analyze {state.rawData?.length.toLocaleString() || 0} rows to detect patterns, seasonality, and trends
                          </p>
                        </div>
                      </li>
                      <li className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-medium text-purple-600">2</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-800">Model Training</span>
                          <p className="text-sm text-gray-600">
                            {state.forecastMode === 'by_slice' && state.selectedSliceValues.length > 0
                              ? `${state.selectedSliceValues.length} separate models will be trained (one per ${state.selectedSliceCols.join(' + ')} combination)`
                              : 'One model will be trained on your aggregated data'}
                          </p>
                        </div>
                      </li>
                      <li className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-medium text-green-600">3</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-800">Forecast Generation</span>
                          <p className="text-sm text-gray-600">
                            {state.horizon} {state.profile?.profile.frequency || 'period'} forecast with confidence intervals
                          </p>
                        </div>
                      </li>
                    </ul>
                  </div>
                </div>

                {/* Data Per Slice Preview (if by_slice mode) */}
                {state.forecastMode === 'by_slice' && getSliceStats && (
                  <div className="mb-4">
                    <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                      <PieChart className="w-4 h-4 mr-2 text-green-600" />
                      Data per Segment
                      <span className="ml-2 text-xs font-normal text-gray-500">
                        ({state.selectedSliceValues.length} of {Object.keys(getSliceStats).length} selected for forecasting)
                      </span>
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase w-20">Selected</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">{state.selectedSliceCols.join(' + ')}</th>
                            <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Rows</th>
                            <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Avg {state.selectedTargetCol}</th>
                            <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Status</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {/* Show selected segments first, then others */}
                          {Object.entries(getSliceStats)
                            .sort(([a], [b]) => {
                              const aSelected = state.selectedSliceValues.includes(a);
                              const bSelected = state.selectedSliceValues.includes(b);
                              if (aSelected && !bSelected) return -1;
                              if (!aSelected && bSelected) return 1;
                              return 0;
                            })
                            .map(([slice, stats]) => {
                              const isSelected = state.selectedSliceValues.includes(slice);
                              return (
                                <tr key={slice} className={isSelected ? 'bg-green-50 border-l-4 border-l-green-500' : 'opacity-60'}>
                                  <td className="px-3 py-2 text-center">
                                    {isSelected ? (
                                      <span className="inline-flex items-center justify-center w-6 h-6 bg-green-500 rounded-full">
                                        <Check className="w-4 h-4 text-white" />
                                      </span>
                                    ) : (
                                      <span className="text-gray-400">-</span>
                                    )}
                                  </td>
                                  <td className={`px-3 py-2 font-medium ${isSelected ? 'text-gray-900' : 'text-gray-500'}`}>{slice}</td>
                                  <td className={`px-3 py-2 text-right ${isSelected ? 'text-gray-700' : 'text-gray-400'}`}>{stats.count.toLocaleString()}</td>
                                  <td className={`px-3 py-2 text-right ${isSelected ? 'text-gray-700' : 'text-gray-400'}`}>{stats.avg.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                  <td className="px-3 py-2 text-center">
                                    {stats.count < 12 ? (
                                      <span className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded">Insufficient data</span>
                                    ) : stats.count < 52 ? (
                                      <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded">Limited data</span>
                                    ) : (
                                      <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">Good</span>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </div>
                    <p className="mt-2 text-xs text-gray-500">
                      Only segments with a checkmark will be forecasted. Segments with less than 12 data points may have unreliable forecasts.
                    </p>
                  </div>
                )}

                {/* Consequences */}
                <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                  <div className="flex items-start space-x-3">
                    <Lightbulb className="w-5 h-5 text-amber-600 flex-shrink-0" />
                    <div>
                      <h5 className="font-medium text-amber-800 mb-2">Important Considerations</h5>
                      <ul className="text-sm text-amber-700 space-y-1">
                        {state.selectedCovariates.length > 5 && (
                          <li>• Many features ({state.selectedCovariates.length}) may lead to overfitting. Consider reducing.</li>
                        )}
                        {state.selectedCovariates.length === 0 && (
                          <li>• No features selected. Forecast will rely solely on historical patterns.</li>
                        )}
                        {state.forecastMode === 'by_slice' && state.selectedSliceValues.length > 10 && (
                          <li>• Many segments ({state.selectedSliceValues.length}) will increase processing time significantly.</li>
                        )}
                        {state.rawData && state.rawData.length < 52 && (
                          <li>• Limited historical data ({state.rawData.length} rows). Forecast confidence may be lower.</li>
                        )}
                        {state.horizon > 26 && (
                          <li>• Long forecast horizon ({state.horizon} periods). Accuracy typically decreases for distant forecasts.</li>
                        )}
                        {!state.selectedCovariates.length && !state.rawData?.some(r => r['promo'] || r['promotion'] || r['holiday']) && (
                          <li>• No promotions/events detected. Forecast won't account for marketing activities.</li>
                        )}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Forecast Settings & Run Button */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-end justify-between">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Forecast Horizon</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={state.horizon}
                    onChange={(e) => setState(s => ({ ...s, horizon: parseInt(e.target.value) || 12 }))}
                    min={1}
                    max={52}
                    className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-center"
                  />
                  <span className="text-sm text-gray-500">
                    {state.profile?.profile.frequency === 'weekly' ? 'weeks' :
                     state.profile?.profile.frequency === 'daily' ? 'days' : 'periods'}
                  </span>
                </div>
              </div>

              <button
                onClick={handleForecast}
                disabled={state.isLoading || !state.selectedDateCol || !state.selectedTargetCol}
                className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors flex items-center"
              >
                {state.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Generate Forecast
                  </>
                )}
              </button>
            </div>

            <p className="mt-3 text-xs text-gray-500">
              Models will be auto-selected based on your data characteristics
            </p>
          </div>
        </div>
      )}

      {/* Step 3: Loading */}
      {state.step === 'forecasting' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Generating Forecast...</h3>
          <p className="text-gray-500">
            Training models and generating predictions.<br />
            This may take a minute or two.
          </p>
          <div className="mt-6 text-sm text-gray-400">
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              <span>Analyzing patterns and seasonality...</span>
            </div>
          </div>
        </div>
      )}

      {/* Step 4: Results */}
      {state.step === 'results' && state.forecast && (
        <div className="space-y-6">
          {/* By-Slice Mode Banner */}
          {state.forecast.forecast_mode === 'by_slice' && state.forecast.slice_forecasts && state.forecast.slice_forecasts.length > 0 && (
            <div className="bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 rounded-xl p-4 shadow-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="bg-white/20 rounded-lg p-2">
                    <GitBranch className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-white font-semibold text-lg">Segmented Forecast Results</h3>
                    <p className="text-purple-100 text-sm">
                      {state.forecast.slice_forecasts.length} individual models trained for your data segments
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-white/70 text-xs uppercase tracking-wide">Slice Column</div>
                    <div className="text-white font-medium">{state.forecast.slice_columns?.join(' + ') || 'Unknown'}</div>
                  </div>
                  <div className="h-10 w-px bg-white/20"></div>
                  <div className="text-right">
                    <div className="text-white/70 text-xs uppercase tracking-wide">Segments</div>
                    <div className="text-white font-bold text-xl">{state.forecast.slice_forecasts.length}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Summary Card */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-gray-800 flex items-center">
                <CheckCircle2 className="w-5 h-5 mr-2 text-green-600" />
                Forecast Complete
              </h3>
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleDownloadExcel}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium flex items-center"
                >
                  <FileSpreadsheet className="w-4 h-4 mr-2" />
                  Download Excel
                </button>
                <button
                  onClick={handleDownloadCSV}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-sm font-medium flex items-center"
                >
                  <Download className="w-4 h-4 mr-2" />
                  CSV
                </button>
              </div>
            </div>

            {/* Summary Text */}
            <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm whitespace-pre-wrap">
              {state.forecast.summary}
            </div>
          </div>

          {/* NEW: How We Built Your Forecast - Plain English Explanation */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-sm border border-blue-200 p-6">
            <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
              <Lightbulb className="w-5 h-5 mr-2 text-blue-600" />
              How We Built Your Forecast
              <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">Plain English</span>
            </h3>

            {/* Model Selection Explanation */}
            <div className="space-y-4">
              {/* What Models Were Tested */}
              <div className="bg-white/70 rounded-lg p-4">
                <h4 className="font-medium text-gray-700 mb-2 flex items-center">
                  <Target className="w-4 h-4 mr-2 text-purple-600" />
                  Models We Tested
                </h4>
                <p className="text-sm text-gray-600 mb-2">
                  We automatically tested {state.forecast.all_models_trained?.length || 3} different forecasting approaches to find the best fit for your data:
                </p>
                <div className="flex flex-wrap gap-2">
                  {(state.forecast.all_models_trained || ['Prophet', 'ARIMA', 'XGBoost']).map((model, idx) => {
                    const isBest = state.forecast.best_model?.includes(model.split('(')[0]);
                    return (
                      <span key={idx} className={`px-3 py-1 rounded-full text-xs font-medium ${
                        isBest ? 'bg-green-100 text-green-700 ring-2 ring-green-300' : 'bg-gray-100 text-gray-600'
                      }`}>
                        {model} {isBest && '✓ Winner'}
                      </span>
                    );
                  })}
                </div>
              </div>

              {/* Why Winner Was Chosen */}
              <div className="bg-white/70 rounded-lg p-4">
                <h4 className="font-medium text-gray-700 mb-2 flex items-center">
                  <Shield className="w-4 h-4 mr-2 text-green-600" />
                  Why {state.forecast.best_model || 'This Model'} Won
                </h4>
                <div className="text-sm text-gray-600 space-y-2">
                  {(() => {
                    const bestModel = state.forecast.best_model || '';
                    const mape = state.forecast.holdout_mape || state.forecast.confidence?.mape || 5;

                    // Generate explanation based on model type
                    if (bestModel.toLowerCase().includes('arima')) {
                      return (
                        <>
                          <p><strong>ARIMA</strong> (Auto-Regressive Integrated Moving Average) was chosen because:</p>
                          <ul className="list-disc list-inside space-y-1 ml-2">
                            <li>Your data shows consistent patterns that ARIMA captures well</li>
                            <li>It achieved the lowest error rate ({mape.toFixed(1)}% MAPE) on unseen test data</li>
                            <li>ARIMA is excellent for data with clear trends and stable seasonality</li>
                            <li>It's particularly good when you have regular weekly/monthly patterns</li>
                          </ul>
                        </>
                      );
                    } else if (bestModel.toLowerCase().includes('xgboost')) {
                      return (
                        <>
                          <p><strong>XGBoost</strong> (Extreme Gradient Boosting) was chosen because:</p>
                          <ul className="list-disc list-inside space-y-1 ml-2">
                            <li>Your data has complex patterns that machine learning captures best</li>
                            <li>It achieved the lowest error rate ({mape.toFixed(1)}% MAPE) on unseen test data</li>
                            <li>XGBoost excels at learning from multiple features (covariates)</li>
                            <li>It's especially powerful when holidays and external factors affect your numbers</li>
                          </ul>
                        </>
                      );
                    } else if (bestModel.toLowerCase().includes('prophet')) {
                      return (
                        <>
                          <p><strong>Prophet</strong> (Facebook's forecasting model) was chosen because:</p>
                          <ul className="list-disc list-inside space-y-1 ml-2">
                            <li>Your data has strong yearly/weekly seasonality that Prophet handles well</li>
                            <li>It achieved the lowest error rate ({mape.toFixed(1)}% MAPE) on unseen test data</li>
                            <li>Prophet is designed for business data with holiday effects</li>
                            <li>It automatically detects changepoints where trends shift</li>
                          </ul>
                        </>
                      );
                    } else {
                      return (
                        <>
                          <p>The winning model achieved the lowest error rate ({mape.toFixed(1)}% MAPE) when tested on data it hadn't seen before.</p>
                          <ul className="list-disc list-inside space-y-1 ml-2">
                            <li>We held back 15% of your historical data as a "test set"</li>
                            <li>Each model was trained on the remaining 85% and then predicted the test set</li>
                            <li>The model with the smallest prediction errors wins</li>
                          </ul>
                        </>
                      );
                    }
                  })()}
                </div>
              </div>

              {/* Data Treatments Applied */}
              <div className="bg-white/70 rounded-lg p-4">
                <h4 className="font-medium text-gray-700 mb-2 flex items-center">
                  <RefreshCw className="w-4 h-4 mr-2 text-orange-600" />
                  What We Did With Your Data
                </h4>
                <div className="text-sm text-gray-600">
                  <p className="mb-2">Before training, we automatically prepared your data:</p>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div className="flex items-start space-x-2">
                      <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span><strong>Sorted by date</strong> - Ensured chronological order</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span><strong>Handled missing values</strong> - Filled gaps appropriately</span>
                    </div>
                    {state.forecast?.forecast_mode === 'by_slice' && (
                      <div className="flex items-start space-x-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span><strong>Trained per segment</strong> - Each slice got its own model</span>
                      </div>
                    )}
                    {state.selectedCovariates.length > 0 && (
                      <div className="flex items-start space-x-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span><strong>Included {state.selectedCovariates.length} features</strong> - Used to improve predictions</span>
                      </div>
                    )}
                    <div className="flex items-start space-x-2">
                      <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span><strong>Detected frequency</strong> - {state.profile?.profile?.frequency || 'weekly'} patterns</span>
                    </div>
                    {state.profile?.profile?.has_seasonality && (
                      <div className="flex items-start space-x-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span><strong>Found seasonality</strong> - Recurring patterns in your data</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Holiday & Seasonality Handling */}
              {(state.selectedCovariates.some(c => /holiday|easter|christmas|thanksgiving/i.test(c)) || state.profile?.profile?.holiday_coverage_score > 0) && (
                <div className="bg-white/70 rounded-lg p-4">
                  <h4 className="font-medium text-gray-700 mb-2 flex items-center">
                    <Calendar className="w-4 h-4 mr-2 text-red-600" />
                    How Holidays & Special Days Are Handled
                  </h4>
                  <div className="text-sm text-gray-600 space-y-2">
                    <p>
                      <strong>Holiday indicators</strong> in your data (like Christmas, Thanksgiving, etc.) tell the model
                      when to expect unusual spikes or dips in your numbers.
                    </p>
                    <div className="bg-yellow-50 p-3 rounded border border-yellow-200">
                      <p className="text-yellow-800">
                        <strong>What this means for your forecast:</strong> When a future date falls on a holiday,
                        the model will automatically adjust the prediction based on how your data typically behaves
                        during that holiday in previous years.
                      </p>
                    </div>
                    {state.selectedCovariates.filter(c => /holiday|easter|christmas|thanksgiving|halloween|super.?bowl/i.test(c)).length > 0 && (
                      <p className="text-gray-500 text-xs mt-2">
                        Holidays included: {state.selectedCovariates.filter(c => /holiday|easter|christmas|thanksgiving|halloween|super.?bowl/i.test(c)).join(', ')}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* Confidence Explanation */}
              <div className="bg-white/70 rounded-lg p-4">
                <h4 className="font-medium text-gray-700 mb-2 flex items-center">
                  <Info className="w-4 h-4 mr-2 text-blue-600" />
                  Understanding the Confidence Interval
                </h4>
                <div className="text-sm text-gray-600">
                  <p>
                    The shaded area around the forecast line shows the <strong>95% confidence interval</strong> -
                    there's a 95% chance the actual value will fall within this range.
                  </p>
                  <div className="mt-2 p-2 bg-blue-50 rounded">
                    <p className="text-blue-800 text-xs">
                      <strong>Tip:</strong> Wider intervals mean more uncertainty. If the interval is very wide,
                      consider adding more historical data or relevant features to improve accuracy.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Forecast Chart */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-gray-800 flex items-center">
                <LineChart className="w-5 h-5 mr-2 text-purple-600" />
                Forecast Visualization
              </h3>

              {/* Slice selector - enhanced with visual chips for by_slice mode */}
              {state.forecast?.forecast_mode === 'by_slice' && state.forecast?.slice_forecasts && state.forecast.slice_forecasts.length > 0 && (
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-500 font-medium">View Slice:</span>
                  <div className="flex flex-wrap gap-2">
                    {/* All Slices chip */}
                    <button
                      onClick={() => setState(s => ({ ...s, selectedSliceView: 'all' }))}
                      className={`px-3 py-1.5 text-sm font-medium rounded-full transition-all duration-200 ${
                        state.selectedSliceView === 'all'
                          ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      All ({state.forecast.slice_forecasts.length})
                    </button>
                    {/* Individual slice chips */}
                    {state.forecast.slice_forecasts.map((slice, idx) => {
                      const color = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][idx % 8];
                      const isSelected = state.selectedSliceView === slice.slice_id;
                      return (
                        <button
                          key={idx}
                          onClick={() => setState(s => ({ ...s, selectedSliceView: slice.slice_id }))}
                          className={`px-3 py-1.5 text-sm font-medium rounded-full transition-all duration-200 flex items-center gap-1.5 ${
                            isSelected
                              ? 'text-white shadow-md'
                              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                          }`}
                          style={isSelected ? { backgroundColor: color } : {}}
                        >
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: isSelected ? 'white' : color }}
                          ></span>
                          {slice.slice_id}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
            <div className="h-80">
              {(() => {
                // Helper to parse date strings and normalize to YYYY-MM-DD
                const parseDate = (val: unknown): string => {
                  if (!val) return '';
                  const str = String(val).trim();

                  // Handle ISO format (2024-01-15T00:00:00)
                  if (str.includes('T')) {
                    return str.split('T')[0];
                  }

                  // Handle datetime with space (2024-01-15 00:00:00)
                  if (str.includes(' ')) {
                    const datePart = str.split(' ')[0];
                    // Check if already YYYY-MM-DD
                    if (datePart.match(/^\d{4}-\d{2}-\d{2}$/)) {
                      return datePart;
                    }
                  }

                  // Handle M/D/YY or MM/DD/YY format (e.g., "12/8/25" or "1/15/24")
                  if (str.includes('/')) {
                    const parts = str.split('/');
                    if (parts.length === 3) {
                      let [month, day, year] = parts.map(p => parseInt(p, 10));
                      // Assume 2-digit year means 20XX
                      if (year < 100) year += 2000;
                      // Pad month and day
                      const mm = String(month).padStart(2, '0');
                      const dd = String(day).padStart(2, '0');
                      return `${year}-${mm}-${dd}`;
                    }
                  }

                  // Handle YYYY-MM-DD already
                  if (str.match(/^\d{4}-\d{2}-\d{2}$/)) {
                    return str;
                  }

                  return str;
                };

                // Format date for display (expects YYYY-MM-DD)
                const formatDate = (str: string): string => {
                  if (!str) return '';
                  const parts = str.split('-');
                  if (parts.length === 3) {
                    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                    const monthIdx = parseInt(parts[1], 10) - 1;
                    if (monthIdx >= 0 && monthIdx < 12) {
                      return `${monthNames[monthIdx]} ${parseInt(parts[2], 10)}`;
                    }
                  }
                  return str;
                };

                // Check if we have slice forecasts (by-slice mode)
                const hasSliceForecasts = state.forecast?.forecast_mode === 'by_slice' &&
                                          state.forecast?.slice_forecasts &&
                                          state.forecast.slice_forecasts.length > 0;

                // Get the selected slice (if viewing a specific slice)
                const selectedSlice = hasSliceForecasts && state.selectedSliceView !== 'all'
                  ? state.forecast!.slice_forecasts!.find(s => s.slice_id === state.selectedSliceView)
                  : null;

                // Colors for different slices
                const sliceColors = [
                  '#7c3aed', // purple
                  '#2563eb', // blue
                  '#059669', // green
                  '#d97706', // amber
                  '#dc2626', // red
                  '#ec4899', // pink
                  '#0891b2', // cyan
                  '#4f46e5', // indigo
                ];

                // Build chart data - different for individual slice vs all slices
                type ChartDataPoint = {
                  date: string;
                  actual?: number;
                  forecast?: number;
                  lower?: number;
                  upper?: number;
                  [key: string]: number | string | undefined;
                };
                const chartData: ChartDataPoint[] = [];

                // Add historical data - filter by slice to match forecasted slices
                if (state.rawData && state.selectedDateCol && state.selectedTargetCol) {
                  const dateAggregates: Record<string, number> = {};

                  // Get the slice columns and determine which slices to include
                  const sliceColumns = state.forecast?.slice_columns || state.selectedSliceCols;
                  const hasSliceColumns = sliceColumns && sliceColumns.length > 0;

                  // Build a set of forecasted slice IDs for quick lookup
                  const forecastedSliceIds = new Set<string>();
                  if (hasSliceForecasts && state.forecast?.slice_forecasts) {
                    state.forecast.slice_forecasts.forEach(slice => {
                      forecastedSliceIds.add(slice.slice_id);
                    });
                  }

                  state.rawData.forEach(row => {
                    // If viewing individual slice, filter by that slice only
                    if (selectedSlice && hasSliceColumns) {
                      // Split by ' | ' (space-pipe-space) to match how slice_id is constructed
                      const sliceParts = selectedSlice.slice_id.split(' | ').map(s => s.trim());
                      let matches = true;
                      sliceColumns.forEach((col, idx) => {
                        const expectedValue = sliceParts[idx] || sliceParts[0];
                        const rowValue = String(row[col] ?? '').trim();
                        if (rowValue !== expectedValue) {
                          matches = false;
                        }
                      });
                      if (!matches) return;
                    }
                    // If viewing all slices in by_slice mode, only include rows from forecasted slices
                    else if (hasSliceForecasts && hasSliceColumns && forecastedSliceIds.size > 0) {
                      // Build the slice_id for this row
                      const rowSliceId = sliceColumns.map(col => String(row[col] ?? '')).join(' | ');
                      if (!forecastedSliceIds.has(rowSliceId)) return;
                    }

                    const dateVal = row[state.selectedDateCol];
                    const targetVal = row[state.selectedTargetCol];
                    if (!dateVal) return;
                    if (targetVal === null || targetVal === undefined || targetVal === '') return;

                    let numVal: number;
                    if (typeof targetVal === 'string') {
                      numVal = Number(targetVal.replace(/,/g, ''));
                    } else {
                      numVal = Number(targetVal);
                    }
                    if (isNaN(numVal)) return;

                    const dateStr = parseDate(dateVal);
                    if (!dateStr) return;

                    if (!dateAggregates[dateStr]) {
                      dateAggregates[dateStr] = 0;
                    }
                    dateAggregates[dateStr] += numVal;
                  });

                  const sortedDates = Object.keys(dateAggregates)
                    .sort((a, b) => a.localeCompare(b))
                    .slice(-15);

                  sortedDates.forEach(dateStr => {
                    chartData.push({ date: dateStr, actual: dateAggregates[dateStr] });
                  });
                }

                // Track where forecast starts
                const forecastStartIdx = chartData.length;

                // Add forecast data
                if (selectedSlice) {
                  // INDIVIDUAL SLICE VIEW: Show single slice forecast with confidence interval
                  selectedSlice.dates.forEach((date, idx) => {
                    const dateStr = parseDate(date);
                    if (dateStr) {
                      chartData.push({
                        date: dateStr,
                        forecast: selectedSlice.forecast[idx],
                        lower: selectedSlice.lower_bounds?.[idx],
                        upper: selectedSlice.upper_bounds?.[idx],
                      });
                    }
                  });
                } else if (hasSliceForecasts && state.forecast?.slice_forecasts && state.selectedSliceView === 'all') {
                  // ALL SLICES VIEW: Show multiple forecast lines
                  const sliceForecasts = state.forecast.slice_forecasts;
                  const forecastByDate: Record<string, ChartDataPoint> = {};

                  sliceForecasts.forEach((slice, sliceIdx) => {
                    slice.dates.forEach((date, dateIdx) => {
                      const dateStr = parseDate(date);
                      if (!dateStr) return;

                      if (!forecastByDate[dateStr]) {
                        forecastByDate[dateStr] = { date: dateStr };
                      }

                      const sliceKey = `forecast_${sliceIdx}`;
                      forecastByDate[dateStr][sliceKey] = slice.forecast[dateIdx];
                    });
                  });

                  const forecastDates = Object.keys(forecastByDate).sort((a, b) => a.localeCompare(b));
                  forecastDates.forEach(dateStr => {
                    chartData.push(forecastByDate[dateStr]);
                  });
                } else if (state.forecast?.dates && state.forecast?.forecast) {
                  // AGGREGATE MODE: Single forecast line
                  state.forecast.dates.forEach((date, idx) => {
                    const dateStr = parseDate(date);
                    if (dateStr) {
                      chartData.push({
                        date: dateStr,
                        forecast: state.forecast!.forecast[idx],
                        lower: state.forecast!.lower_bounds?.[idx],
                        upper: state.forecast!.upper_bounds?.[idx],
                      });
                    }
                  });
                }

                const forecastStartDate = chartData[forecastStartIdx]?.date;

                // Determine chart color based on view
                const chartColor = selectedSlice
                  ? sliceColors[state.forecast!.slice_forecasts!.findIndex(s => s.slice_id === selectedSlice.slice_id) % sliceColors.length]
                  : '#7c3aed';

                return (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: '#666' }}
                        tickLine={false}
                        axisLine={{ stroke: '#e0e0e0' }}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                        tickFormatter={formatDate}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: '#666' }}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) =>
                          value >= 1000000 ? `${(value / 1000000).toFixed(1)}M` :
                          value >= 1000 ? `${(value / 1000).toFixed(0)}K` :
                          String(Math.round(value))
                        }
                      />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          const data = payload[0]?.payload;
                          return (
                            <div className="bg-white rounded-lg border border-gray-200 shadow-lg p-3 max-w-xs">
                              <p className="text-xs text-gray-500 mb-2 font-medium">{formatDate(label)}</p>
                              {data?.actual !== undefined && (
                                <div className="flex justify-between text-xs gap-4">
                                  <span className="text-gray-700">Actual:</span>
                                  <span className="font-mono">{Math.round(data.actual).toLocaleString()}</span>
                                </div>
                              )}
                              {/* Handle slice forecasts - different display for individual vs all slices */}
                              {selectedSlice ? (
                                // Individual slice view
                                <>
                                  {data?.forecast !== undefined && (
                                    <div className="flex justify-between text-xs gap-4">
                                      <span style={{ color: chartColor }}>{selectedSlice.slice_id}:</span>
                                      <span className="font-mono">{Math.round(data.forecast).toLocaleString()}</span>
                                    </div>
                                  )}
                                  {data?.lower !== undefined && data?.upper !== undefined && (
                                    <div className="flex justify-between text-xs gap-4" style={{ color: chartColor }}>
                                      <span>Range:</span>
                                      <span className="font-mono">{Math.round(data.lower).toLocaleString()} - {Math.round(data.upper).toLocaleString()}</span>
                                    </div>
                                  )}
                                </>
                              ) : hasSliceForecasts && state.forecast?.slice_forecasts && state.selectedSliceView === 'all' ? (
                                // All slices view
                                state.forecast.slice_forecasts.map((slice, idx) => {
                                  const sliceKey = `forecast_${idx}`;
                                  if (data?.[sliceKey] !== undefined) {
                                    return (
                                      <div key={idx} className="flex justify-between text-xs gap-4 mt-1">
                                        <span style={{ color: sliceColors[idx % sliceColors.length] }}>
                                          {slice.slice_id}:
                                        </span>
                                        <span className="font-mono">{Math.round(data[sliceKey] as number).toLocaleString()}</span>
                                      </div>
                                    );
                                  }
                                  return null;
                                })
                              ) : (
                                // Aggregate mode
                                <>
                                  {data?.forecast !== undefined && (
                                    <div className="flex justify-between text-xs gap-4">
                                      <span className="text-purple-600">Forecast:</span>
                                      <span className="font-mono">{Math.round(data.forecast).toLocaleString()}</span>
                                    </div>
                                  )}
                                  {data?.lower !== undefined && data?.upper !== undefined && (
                                    <div className="flex justify-between text-xs gap-4 text-purple-500">
                                      <span>Range:</span>
                                      <span className="font-mono">{Math.round(data.lower).toLocaleString()} - {Math.round(data.upper).toLocaleString()}</span>
                                    </div>
                                  )}
                                </>
                              )}
                            </div>
                          );
                        }}
                      />

                      {/* Confidence interval - only show for individual slice or aggregate mode */}
                      {(selectedSlice || !hasSliceForecasts || state.selectedSliceView !== 'all') && (
                        <>
                          <Area
                            type="monotone"
                            dataKey="upper"
                            stroke="transparent"
                            fill={chartColor}
                            fillOpacity={0.15}
                            connectNulls={false}
                            legendType="none"
                            activeDot={false}
                          />
                          <Area
                            type="monotone"
                            dataKey="lower"
                            stroke="transparent"
                            fill="#ffffff"
                            fillOpacity={1}
                            connectNulls={false}
                            legendType="none"
                            activeDot={false}
                          />
                        </>
                      )}

                      {/* Historical actuals - solid line */}
                      <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#374151"
                        strokeWidth={2}
                        dot={{ r: 3, fill: '#374151', strokeWidth: 0 }}
                        connectNulls={false}
                        name="Historical"
                      />

                      {/* Forecast lines - different rendering based on view mode */}
                      {selectedSlice ? (
                        // INDIVIDUAL SLICE VIEW: Single forecast line with the slice's color
                        <Line
                          type="monotone"
                          dataKey="forecast"
                          stroke={chartColor}
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={{ r: 4, fill: chartColor, strokeWidth: 0 }}
                          connectNulls={false}
                          name={selectedSlice.slice_id}
                        />
                      ) : hasSliceForecasts && state.forecast?.slice_forecasts && state.selectedSliceView === 'all' ? (
                        // ALL SLICES VIEW: Multiple forecast lines with different colors
                        state.forecast.slice_forecasts.map((slice, idx) => (
                          <Line
                            key={`slice-${idx}`}
                            type="monotone"
                            dataKey={`forecast_${idx}`}
                            stroke={sliceColors[idx % sliceColors.length]}
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={{ r: 3, fill: sliceColors[idx % sliceColors.length], strokeWidth: 0 }}
                            connectNulls={false}
                            name={slice.slice_id}
                          />
                        ))
                      ) : (
                        // AGGREGATE MODE: Single forecast line
                        <Line
                          type="monotone"
                          dataKey="forecast"
                          stroke="#7c3aed"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={{ r: 4, fill: '#7c3aed', strokeWidth: 0 }}
                          connectNulls={false}
                          name="Forecast"
                        />
                      )}

                      {/* Reference line at forecast start */}
                      {forecastStartDate && (
                        <ReferenceLine
                          x={forecastStartDate}
                          stroke="#9ca3af"
                          strokeDasharray="3 3"
                        />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                );
              })()}
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-center gap-4 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <span className="w-4 h-0.5 bg-gray-700"></span> Historical Data
              </span>
              {/* Legend - depends on view mode */}
              {state.forecast?.forecast_mode === 'by_slice' && state.forecast?.slice_forecasts ? (
                state.selectedSliceView === 'all' ? (
                  // Show all slices in legend
                  state.forecast.slice_forecasts.map((slice, idx) => (
                    <span key={idx} className="flex items-center gap-1">
                      <span
                        className="w-4 h-0.5"
                        style={{
                          backgroundColor: ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][idx % 8],
                          borderTop: `2px dashed ${['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][idx % 8]}`
                        }}
                      ></span>
                      {slice.slice_id}
                    </span>
                  ))
                ) : (
                  // Show selected slice legend
                  <>
                    <span className="flex items-center gap-1">
                      <span
                        className="w-4 h-0.5"
                        style={{
                          backgroundColor: ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][state.forecast.slice_forecasts.findIndex(s => s.slice_id === state.selectedSliceView) % 8],
                          borderTop: `2px dashed ${['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][state.forecast.slice_forecasts.findIndex(s => s.slice_id === state.selectedSliceView) % 8]}`
                        }}
                      ></span>
                      {state.selectedSliceView} Forecast
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="w-4 h-3 rounded"
                        style={{
                          backgroundColor: ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][state.forecast.slice_forecasts.findIndex(s => s.slice_id === state.selectedSliceView) % 8],
                          opacity: 0.15
                        }}
                      ></span>
                      Confidence Interval
                    </span>
                  </>
                )
              ) : (
                <>
                  <span className="flex items-center gap-1">
                    <span className="w-4 h-0.5 bg-purple-600" style={{ borderTop: '2px dashed #7c3aed' }}></span> Forecast
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-4 h-3 bg-purple-100 rounded"></span> Confidence Interval
                  </span>
                </>
              )}
            </div>
            {/* Info text */}
            <p className="text-xs text-gray-400 mt-2 text-center">
              {state.forecast?.forecast_mode === 'by_slice' && state.forecast?.slice_forecasts
                ? state.selectedSliceView === 'all'
                  ? `Showing ${state.forecast.slice_forecasts.length} slice forecasts × ${state.forecast.slice_forecasts[0]?.dates?.length || 0} periods each`
                  : `Showing ${state.selectedSliceView} forecast (${state.forecast.slice_forecasts.find(s => s.slice_id === state.selectedSliceView)?.dates?.length || 0} periods)`
                : `Showing last 15 historical points + ${state.forecast?.dates?.length || 0} forecast periods`
              }
            </p>
          </div>

          {/* Slice Forecast Details (only shown in by-slice mode) */}
          {state.forecast?.forecast_mode === 'by_slice' && state.forecast?.slice_forecasts && state.forecast.slice_forecasts.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800 flex items-center">
                  <GitBranch className="w-5 h-5 mr-2 text-purple-600" />
                  {state.selectedSliceView !== 'all'
                    ? `${state.selectedSliceView} Forecast Details`
                    : 'Slice Forecast Details'
                  }
                </h3>
                <span className="text-xs text-gray-500 bg-purple-50 px-2 py-1 rounded-full">
                  {state.selectedSliceView !== 'all'
                    ? `Viewing 1 of ${state.forecast.slice_forecasts.length} segments`
                    : `${state.forecast.slice_forecasts.length} segments analyzed`
                  }
                </span>
              </div>

              {/* Summary Cards - Show selected slice details when viewing individual slice */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
                {(() => {
                  const slices = state.forecast.slice_forecasts;
                  const viewingSlice = state.selectedSliceView !== 'all'
                    ? slices.find(s => s.slice_id === state.selectedSliceView)
                    : null;

                  if (viewingSlice) {
                    // Viewing individual slice - show slice-specific details
                    const sliceAvgForecast = viewingSlice.forecast.reduce((a, b) => a + b, 0) / viewingSlice.forecast.length;
                    const sliceMinForecast = Math.min(...viewingSlice.forecast);
                    const sliceMaxForecast = Math.max(...viewingSlice.forecast);

                    return (
                      <>
                        <div className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg p-3 border border-purple-100">
                          <div className="text-xs text-purple-600 font-medium mb-1">Accuracy</div>
                          <div className="text-lg font-bold text-purple-700">{(100 - (viewingSlice.holdout_mape || 0)).toFixed(1)}%</div>
                          <div className="text-xs text-purple-500">MAPE: {(viewingSlice.holdout_mape || 0).toFixed(1)}%</div>
                        </div>
                        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 border border-green-100">
                          <div className="text-xs text-green-600 font-medium mb-1">Best Model</div>
                          <div className="text-lg font-bold text-green-700">{viewingSlice.best_model || 'Unknown'}</div>
                          <div className="text-xs text-green-500">{viewingSlice.dates.length} periods</div>
                        </div>
                        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-3 border border-blue-100">
                          <div className="text-xs text-blue-600 font-medium mb-1">Avg Forecast</div>
                          <div className="text-lg font-bold text-blue-700">{sliceAvgForecast.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                          <div className="text-xs text-blue-500">min: {sliceMinForecast.toLocaleString(undefined, { maximumFractionDigits: 0 })} / max: {sliceMaxForecast.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                        </div>
                        <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 border border-amber-100">
                          <div className="text-xs text-amber-600 font-medium mb-1">Data Points</div>
                          <div className="text-lg font-bold text-amber-700">{viewingSlice.data_points.toLocaleString()}</div>
                          <div className="text-xs text-amber-500">rows trained</div>
                        </div>
                      </>
                    );
                  }

                  // Viewing all slices - show aggregate summary
                  const avgMape = slices.reduce((a, s) => a + (s.holdout_mape || 0), 0) / slices.length;
                  const bestSlice = slices.reduce((best, s) => (s.holdout_mape || 100) < (best.holdout_mape || 100) ? s : best, slices[0]);
                  const totalForecast = slices.reduce((a, s) => a + s.forecast.reduce((x, y) => x + y, 0) / s.forecast.length, 0);
                  const totalDataPoints = slices.reduce((a, s) => a + s.data_points, 0);

                  return (
                    <>
                      <div className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg p-3 border border-purple-100">
                        <div className="text-xs text-purple-600 font-medium mb-1">Avg Accuracy</div>
                        <div className="text-lg font-bold text-purple-700">{(100 - avgMape).toFixed(1)}%</div>
                        <div className="text-xs text-purple-500">MAPE: {avgMape.toFixed(1)}%</div>
                      </div>
                      <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 border border-green-100">
                        <div className="text-xs text-green-600 font-medium mb-1">Best Performer</div>
                        <div className="text-lg font-bold text-green-700">{bestSlice.slice_id}</div>
                        <div className="text-xs text-green-500">MAPE: {(bestSlice.holdout_mape || 0).toFixed(1)}%</div>
                      </div>
                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-3 border border-blue-100">
                        <div className="text-xs text-blue-600 font-medium mb-1">Total Forecast</div>
                        <div className="text-lg font-bold text-blue-700">{totalForecast.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                        <div className="text-xs text-blue-500">avg per period</div>
                      </div>
                      <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 border border-amber-100">
                        <div className="text-xs text-amber-600 font-medium mb-1">Data Points</div>
                        <div className="text-lg font-bold text-amber-700">{totalDataPoints.toLocaleString()}</div>
                        <div className="text-xs text-amber-500">total rows trained</div>
                      </div>
                    </>
                  );
                })()}
              </div>

              <p className="text-sm text-gray-600 mb-4">
                Each slice has its own trained model for more accurate segment-specific forecasting.
              </p>
              <div className="overflow-x-auto rounded-lg border border-gray-200">
                <table className="min-w-full divide-y divide-gray-200 text-sm">
                  <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Slice</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">Model</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">Accuracy</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Data Points</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Avg Forecast</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">Action</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-100">
                    {state.forecast.slice_forecasts.map((slice, idx) => {
                      const avgForecast = slice.forecast.reduce((a, b) => a + b, 0) / slice.forecast.length;
                      const color = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#ec4899', '#0891b2', '#4f46e5'][idx % 8];
                      const isSelected = state.selectedSliceView === slice.slice_id;
                      return (
                        <tr
                          key={idx}
                          className={`transition-colors duration-150 ${
                            isSelected
                              ? 'bg-purple-50 border-l-4 border-l-purple-500'
                              : 'hover:bg-gray-50'
                          }`}
                        >
                          <td className="px-4 py-3 font-medium">
                            <div className="flex items-center gap-2">
                              <span
                                className="w-3 h-3 rounded-full ring-2 ring-white shadow-sm"
                                style={{ backgroundColor: color }}
                              ></span>
                              <span className="text-gray-800">{slice.slice_id}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-gray-700">
                              {slice.best_model || 'Unknown'}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-center">
                            <div className="flex flex-col items-center">
                              <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold ${
                                (slice.holdout_mape || 0) < 10 ? 'bg-green-100 text-green-700' :
                                (slice.holdout_mape || 0) < 20 ? 'bg-yellow-100 text-yellow-700' :
                                'bg-red-100 text-red-700'
                              }`}>
                                {slice.holdout_mape?.toFixed(1) || 'N/A'}% MAPE
                              </span>
                              <span className="text-[10px] text-gray-400 mt-0.5">
                                {(100 - (slice.holdout_mape || 0)).toFixed(0)}% accurate
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-right text-gray-600 font-mono text-xs">
                            {slice.data_points.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-sm font-medium text-gray-800">
                            {avgForecast.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <button
                              onClick={() => setState(s => ({ ...s, selectedSliceView: slice.slice_id }))}
                              className={`text-xs font-medium px-3 py-1.5 rounded-full transition-all duration-200 ${
                                isSelected
                                  ? 'bg-purple-600 text-white'
                                  : 'bg-gray-100 text-gray-600 hover:bg-purple-100 hover:text-purple-700'
                              }`}
                            >
                              {isSelected ? 'Viewing' : 'View'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Confidence */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
              <Shield className="w-5 h-5 mr-2 text-blue-600" />
              Forecast Confidence
            </h3>

            <div className={`inline-flex items-center px-4 py-2 rounded-lg border ${getConfidenceColor(state.forecast.confidence.level)}`}>
              <span className="text-lg font-bold capitalize">{state.forecast.confidence.level}</span>
              <span className="mx-2">•</span>
              <span className="text-sm">Score: {state.forecast.confidence.score.toFixed(0)}/100</span>
              <span className="mx-2">•</span>
              <span className="text-sm">MAPE: {state.forecast.confidence.mape.toFixed(1)}%</span>
            </div>

            <p className="mt-3 text-sm text-gray-600">{state.forecast.confidence.explanation}</p>

            {/* Confidence Factors */}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
              {state.forecast.confidence.factors.map((factor, idx) => (
                <div key={idx} className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-1">{factor.factor}</div>
                  <div className="flex items-center justify-between">
                    <div className={`font-semibold ${
                      factor.score >= 80 ? 'text-green-600' :
                      factor.score >= 60 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {factor.score}/100
                    </div>
                    <div className="text-xs text-gray-400">{factor.note}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* How We Selected the Best Model - NEW SECTION */}
          {state.forecast.data_split && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                <Layers className="w-5 h-5 mr-2 text-purple-600" />
                How We Selected the Best Model
              </h3>

              {/* Visual Data Split */}
              <div className="mb-6">
                <div className="text-sm text-gray-600 mb-3">Your data was split into three sets to ensure reliable model selection:</div>
                <div className="flex h-8 rounded-lg overflow-hidden border border-gray-200">
                  <div
                    className="bg-blue-500 flex items-center justify-center text-white text-xs font-medium"
                    style={{ width: `${state.forecast.data_split.train_pct}%` }}
                    title={`Training: ${state.forecast.data_split.train_size} rows`}
                  >
                    Train ({state.forecast.data_split.train_pct}%)
                  </div>
                  <div
                    className="bg-yellow-500 flex items-center justify-center text-white text-xs font-medium"
                    style={{ width: `${state.forecast.data_split.eval_pct}%` }}
                    title={`Evaluation: ${state.forecast.data_split.eval_size} rows`}
                  >
                    Eval ({state.forecast.data_split.eval_pct}%)
                  </div>
                  <div
                    className="bg-green-500 flex items-center justify-center text-white text-xs font-medium"
                    style={{ width: `${state.forecast.data_split.holdout_pct}%` }}
                    title={`Holdout: ${state.forecast.data_split.holdout_size} rows`}
                  >
                    Holdout ({state.forecast.data_split.holdout_pct}%)
                  </div>
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Model learns patterns</span>
                  <span>Tune parameters</span>
                  <span>Final unbiased test</span>
                </div>
              </div>

              {/* Explanation */}
              {state.forecast.data_split.explanation && (
                <div className="bg-purple-50 rounded-lg p-4 mb-4 text-sm text-purple-800">
                  <div className="flex items-start">
                    <Lightbulb className="w-4 h-4 mr-2 mt-0.5 text-purple-600 flex-shrink-0" />
                    <span>{state.forecast.data_split.explanation}</span>
                  </div>
                </div>
              )}

              {/* Model Comparison Table */}
              {state.forecast.model_comparison && state.forecast.model_comparison.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Model Performance Comparison</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                          <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Eval MAPE</th>
                          <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Holdout MAPE</th>
                          <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {state.forecast.model_comparison.map((model, idx) => (
                          <tr key={idx} className={model.model === state.forecast?.best_model ? 'bg-green-50' : ''}>
                            <td className="px-3 py-2 text-gray-800 font-medium">
                              {model.model}
                              {model.model === state.forecast?.best_model && (
                                <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                                  <Check className="w-3 h-3 mr-1" /> Winner
                                </span>
                              )}
                            </td>
                            <td className="px-3 py-2 text-right">{model.eval_mape.toFixed(2)}%</td>
                            <td className="px-3 py-2 text-right font-semibold">{model.holdout_mape.toFixed(2)}%</td>
                            <td className="px-3 py-2 text-center">
                              {model.overfit_warning ? (
                                <span className="inline-flex items-center text-yellow-600">
                                  <AlertTriangle className="w-4 h-4 mr-1" />
                                  <span className="text-xs">Overfit</span>
                                </span>
                              ) : (
                                <span className="text-green-600">
                                  <CheckCircle2 className="w-4 h-4" />
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    MAPE = Mean Absolute Percentage Error (lower is better). Models are ranked by holdout performance.
                  </p>
                </div>
              )}

              {/* Selection Reason */}
              {state.forecast.selection_reason && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Why This Model?</h4>
                  <div className="text-sm text-gray-600 space-y-2">
                    {state.forecast.selection_reason.split('. ').filter(s => s.trim()).map((sentence, idx) => {
                      // Parse markdown-style bold (**text**)
                      const parts = sentence.split(/\*\*([^*]+)\*\*/g);
                      const hasWarning = sentence.includes('⚠️');
                      const hasSuccess = sentence.includes('✓');

                      return (
                        <p
                          key={idx}
                          className={`${hasWarning ? 'text-yellow-700 bg-yellow-50 p-2 rounded' : hasSuccess ? 'text-green-700 bg-green-50 p-2 rounded' : ''}`}
                        >
                          {parts.map((part, partIdx) =>
                            partIdx % 2 === 1 ? (
                              <strong key={partIdx} className="font-semibold text-gray-800">{part}</strong>
                            ) : (
                              <span key={partIdx}>{part}</span>
                            )
                          )}
                          {!sentence.endsWith('.') && '.'}
                        </p>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Full Data Training Note */}
              {state.forecast.trained_on_full_data && (
                <div className="mt-4 flex items-center text-sm text-green-700 bg-green-50 rounded-lg p-3">
                  <CheckCircle2 className="w-4 h-4 mr-2 flex-shrink-0" />
                  <span>
                    After selecting the best model, we retrained it on <strong>all {state.forecast.data_split.total_rows} data points</strong> to maximize forecast accuracy.
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Future Covariates Used - NEW SECTION */}
          {state.forecast.future_covariates_used && (
            <div className="bg-white rounded-xl shadow-sm border border-purple-200 p-6">
              <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                <Calendar className="w-5 h-5 mr-2 text-purple-600" />
                Future Covariate Data Detected
              </h3>
              <div className="bg-purple-50 rounded-lg p-4 text-sm text-purple-800">
                <div className="flex items-start">
                  <Lightbulb className="w-4 h-4 mr-2 mt-0.5 text-purple-600 flex-shrink-0" />
                  <div>
                    <p className="mb-2">
                      <strong>Great news!</strong> Your data contains {state.forecast.future_covariates_count} rows with
                      known future values (e.g., planned promotions, scheduled events, or budget allocations)
                      {state.forecast.future_covariates_date_range && (
                        <span> from <strong>{state.forecast.future_covariates_date_range[0]}</strong> to <strong>{state.forecast.future_covariates_date_range[1]}</strong></span>
                      )}.
                    </p>
                    <p className="text-purple-700">
                      Instead of guessing these values, we used your actual planned data to make the forecast more accurate.
                      This is particularly valuable when you know future promotions, marketing campaigns, or other events that affect your target.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Anomaly Detection - NEW SECTION */}
          {state.forecast.anomalies && state.forecast.anomalies.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-orange-600" />
                Forecast Anomalies Detected ({state.forecast.anomalies.length})
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                We automatically detected potential issues in your forecast. Review these before using the predictions.
              </p>

              <div className="space-y-4">
                {state.forecast.anomalies.map((anomaly, idx) => (
                  <div
                    key={idx}
                    className={`rounded-lg p-4 border ${
                      anomaly.severity === 'critical' ? 'bg-red-50 border-red-200' :
                      anomaly.severity === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                      'bg-blue-50 border-blue-200'
                    }`}
                  >
                    <div className="flex items-start">
                      <div className={`flex-shrink-0 mr-3 ${
                        anomaly.severity === 'critical' ? 'text-red-600' :
                        anomaly.severity === 'warning' ? 'text-yellow-600' :
                        'text-blue-600'
                      }`}>
                        {anomaly.severity === 'critical' ? <XCircle className="w-5 h-5" /> :
                         anomaly.severity === 'warning' ? <AlertTriangle className="w-5 h-5" /> :
                         <Info className="w-5 h-5" />}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className={`text-sm font-medium ${
                            anomaly.severity === 'critical' ? 'text-red-800' :
                            anomaly.severity === 'warning' ? 'text-yellow-800' :
                            'text-blue-800'
                          }`}>
                            {anomaly.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            anomaly.severity === 'critical' ? 'bg-red-100 text-red-700' :
                            anomaly.severity === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                            'bg-blue-100 text-blue-700'
                          }`}>
                            {anomaly.severity}
                          </span>
                        </div>

                        <p className={`text-sm mt-1 ${
                          anomaly.severity === 'critical' ? 'text-red-700' :
                          anomaly.severity === 'warning' ? 'text-yellow-700' :
                          'text-blue-700'
                        }`}>
                          {anomaly.description}
                        </p>

                        <div className="mt-3 space-y-2">
                          <div className="text-xs">
                            <span className="font-medium text-gray-600">Possible Cause: </span>
                            <span className="text-gray-600">{anomaly.cause}</span>
                          </div>
                          <div className="text-xs">
                            <span className="font-medium text-green-700">Recommended Fix: </span>
                            <span className="text-green-700">{anomaly.fix}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Component Breakdown (Collapsible) */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <button
              onClick={() => setShowComponents(!showComponents)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <span className="font-semibold text-gray-800 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
                Forecast Breakdown (Excel Formula)
              </span>
              {showComponents ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>

            {showComponents && (
              <div className="p-6 border-t border-gray-200">
                <div className="bg-blue-50 rounded-lg p-3 mb-4 text-center">
                  <code className="text-blue-800 font-mono">{state.forecast.components.formula}</code>
                </div>

                {/* Totals */}
                <div className="grid grid-cols-4 gap-4 mb-4">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-500 uppercase">Base</div>
                    <div className="font-semibold">${state.forecast.components.totals.base.toLocaleString()}</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-500 uppercase">Trend</div>
                    <div className="font-semibold">${state.forecast.components.totals.trend.toLocaleString()}</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-500 uppercase">Seasonal</div>
                    <div className="font-semibold">${state.forecast.components.totals.seasonal.toLocaleString()}</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-500 uppercase">Holiday</div>
                    <div className="font-semibold">${state.forecast.components.totals.holiday.toLocaleString()}</div>
                  </div>
                </div>

                {/* Period breakdown table */}
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Forecast</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Lower</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Upper</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Breakdown</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {state.forecast.components.periods.slice(0, 12).map((period, idx) => (
                        <tr key={idx}>
                          <td className="px-3 py-2 text-gray-800">{period.date}</td>
                          <td className="px-3 py-2 text-right font-medium">${period.forecast.toLocaleString()}</td>
                          <td className="px-3 py-2 text-right text-gray-500">${period.lower.toLocaleString()}</td>
                          <td className="px-3 py-2 text-right text-gray-500">${period.upper.toLocaleString()}</td>
                          <td className="px-3 py-2 text-gray-600 text-xs font-mono">{period.explanation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Caveats */}
          {state.forecast.caveats.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-800 mb-2 flex items-center">
                <AlertTriangle className="w-4 h-4 mr-2" />
                Important Notes
              </h4>
              <ul className="text-sm text-yellow-700 space-y-1">
                {state.forecast.caveats.map((caveat, idx) => (
                  <li key={idx}>• {caveat}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Audit Trail (Collapsible) */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <button
              onClick={() => setShowAudit(!showAudit)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <span className="font-semibold text-gray-800 flex items-center">
                <Shield className="w-5 h-5 mr-2 text-gray-600" />
                Audit Trail (Reproducibility)
              </span>
              {showAudit ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>

            {showAudit && (
              <div className="p-6 border-t border-gray-200">
                {/* Explanation Banner */}
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
                  <div className="font-medium mb-1">What is this?</div>
                  <p className="text-blue-700">
                    These values are <strong>automatically generated</strong> for audit purposes.
                    You don't need to input anything - they verify that running the same data with the same
                    settings will produce identical results. The <strong>Reproducibility Token</strong> combines
                    your data hash + config hash, so if two forecasts have the same token, they're guaranteed
                    to produce the same output.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-500">Run ID</div>
                    <div className="font-mono">{state.forecast.audit.run_id}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Timestamp</div>
                    <div className="font-mono">{state.forecast.audit.timestamp}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Data Hash</div>
                    <div className="font-mono text-xs">{state.forecast.audit.data_hash}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Config Hash</div>
                    <div className="font-mono text-xs">{state.forecast.audit.config_hash}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Model</div>
                    <div className="font-mono">{state.forecast.audit.model}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Reproducibility Token</div>
                    <div className="font-mono text-xs break-all">{state.forecast.audit.reproducibility_token}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* New Forecast Button */}
          <div className="text-center">
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium flex items-center mx-auto"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              New Forecast
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleModePanel;
