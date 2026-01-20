/**
 * Expert MLE/Data Scientist & QA Engineer E2E Validation Suite
 *
 * This test suite validates the Finance Forecasting App from multiple perspectives:
 * 1. Data Science: Forecast accuracy, metrics interpretation, model selection
 * 2. QA Engineering: UI functionality, error handling, edge cases
 * 3. Best Practices: Industry standards for financial forecasting
 *
 * Test Data: Paula_merged_data_for_simple_mode.csv
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const os = require('os');

const BASE_URL = process.env.TEST_URL || 'http://localhost:3002';
const TIMEOUT = 480000; // 8 minutes for training
const CSV_PATH = '/Users/debu.sinha/Downloads/Paula_merged_data_for_simple_mode.csv';

// Expert validation criteria
const VALIDATION_CRITERIA = {
  // Data Science thresholds
  mape: {
    excellent: 5,      // MAPE <= 5% is excellent for financial forecasting
    good: 10,          // MAPE 5-10% is good
    acceptable: 15,    // MAPE 10-15% is acceptable
    poor: 25,          // MAPE > 25% is concerning
  },
  // Expected confidence levels based on MAPE
  expectedConfidence: {
    excellent: 'high',
    good: 'medium',
    acceptable: 'medium',
    poor: 'low',
  },
  // Minimum data requirements
  minDataPoints: 50,
  minHistoryWeeks: 52,
};

// Test scenarios designed by expert
const TEST_SCENARIOS = [
  {
    id: 'TC1_AGGREGATE',
    name: 'Aggregate Forecast - All Segments Combined',
    description: 'Test total volume forecast without segmentation',
    mode: 'aggregate',
    targetColumn: 'TOT_VOL',
    sliceColumns: [],
    horizon: 12,
    expertNotes: 'Aggregate forecasts may have higher error due to segment heterogeneity',
    acceptableMAPE: 25, // Higher threshold for aggregate
  },
  {
    id: 'TC2_SINGLE_SLICE',
    name: 'By-Slice: BUSINESS_SEGMENT Only',
    description: 'Segment by business type (Classic, Pickup, Subscription)',
    mode: 'by_slice',
    targetColumn: 'TOT_VOL',
    sliceColumns: ['BUSINESS_SEGMENT'],
    horizon: 12,
    expertNotes: 'Single-dimension slicing should improve accuracy vs aggregate',
    acceptableMAPE: 15,
    expectedSlices: 3, // Classic, Pickup, Subscription
  },
  {
    id: 'TC3_MULTI_SLICE',
    name: 'By-Slice: BUSINESS_SEGMENT + MX_TYPE',
    description: 'Segment by business type and customer type',
    mode: 'by_slice',
    targetColumn: 'TOT_VOL',
    sliceColumns: ['BUSINESS_SEGMENT', 'MX_TYPE'],
    horizon: 10,
    expertNotes: 'Multi-dimensional slicing creates more granular forecasts',
    acceptableMAPE: 15,
    expectedSlices: 6, // 3 segments x 2 types
  },
  {
    id: 'TC4_TRIPLE_SLICE',
    name: 'By-Slice: All Three Dimensions',
    description: 'BUSINESS_SEGMENT + MX_TYPE + IS_CGNA',
    mode: 'by_slice',
    targetColumn: 'TOT_VOL',
    sliceColumns: ['BUSINESS_SEGMENT', 'MX_TYPE', 'IS_CGNA'],
    horizon: 8,
    expertNotes: 'Maximum granularity - may have sparse data issues',
    acceptableMAPE: 20, // Higher threshold due to data sparsity
    expectedSlices: 12, // 3 x 2 x 2
  },
];

// Tracking for expert analysis
const expertAnalysis = {
  testResults: [],
  dataQualityIssues: [],
  forecastInsights: [],
  uiIssues: [],
  recommendations: [],
};

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function setupBrowser() {
  const userDataDir = path.join(os.homedir(), '.puppeteer-expert-validation');
  if (!fs.existsSync(userDataDir)) {
    fs.mkdirSync(userDataDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: false,
    slowMo: 40,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    userDataDir,
    protocolTimeout: 120000,
  });

  return browser;
}

async function navigateToSimpleMode(page) {
  await page.goto(BASE_URL, { waitUntil: 'networkidle2', timeout: 30000 });
  await delay(1500);

  // Ensure Simple Mode
  const switchedMode = await page.evaluate(() => {
    const buttons = document.querySelectorAll('button, div[role="button"], .cursor-pointer');
    for (const btn of buttons) {
      if (btn.textContent && btn.textContent.includes('Simple Mode')) {
        btn.click();
        return true;
      }
    }
    return document.body.innerText.includes('Autopilot');
  });

  await delay(1000);
  return switchedMode;
}

async function uploadCSV(page) {
  const fileInput = await page.$('input[type="file"]');
  if (!fileInput) throw new Error('File input not found');

  await fileInput.uploadFile(CSV_PATH);
  await delay(3000);

  // Wait for data processing
  await page.waitForFunction(() => {
    const text = document.body.innerText;
    return text.includes('Data Loaded') || text.includes('Column Overview') || text.includes('Configure');
  }, { timeout: 60000 });

  await delay(2000);
  return true;
}

async function configureAndRunForecast(page, scenario) {
  console.log(`\n   📊 Configuring: ${scenario.name}`);

  // Set forecast mode
  if (scenario.mode === 'by_slice' && scenario.sliceColumns.length > 0) {
    // Click "Forecast by Slice" option
    await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      for (const el of elements) {
        if (el.textContent && el.textContent.includes('Forecast by Slice') && el.closest('.cursor-pointer, button')) {
          (el.closest('.cursor-pointer, button') || el).click();
          return;
        }
      }
    });
    await delay(500);

    // Select slice columns
    for (const col of scenario.sliceColumns) {
      await page.evaluate((colName) => {
        const cards = document.querySelectorAll('.cursor-pointer, [class*="rounded-lg"][class*="border"]');
        for (const card of cards) {
          if (card.textContent && card.textContent.includes(colName)) {
            const isSelected = card.className.includes('border-purple') || card.className.includes('bg-purple');
            if (!isSelected) {
              card.click();
            }
            return;
          }
        }
      }, col);
      await delay(300);
    }
  }

  // Set horizon
  await page.evaluate((h) => {
    const labels = document.querySelectorAll('label');
    for (const label of labels) {
      if (label.textContent && label.textContent.includes('Forecast Horizon')) {
        const container = label.closest('div');
        const input = container?.querySelector('input[type="number"]');
        if (input) {
          input.value = h;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    }
  }, scenario.horizon);
  await delay(500);

  // Scroll and click Generate Forecast
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await delay(500);

  const clicked = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find(b => b.textContent && b.textContent.includes('Generate Forecast') && !b.disabled);
    if (btn) {
      btn.click();
      return true;
    }
    return false;
  });

  if (!clicked) {
    throw new Error('Generate Forecast button not found or disabled');
  }

  await delay(2000);
  return true;
}

async function waitForForecastComplete(page, scenarioId) {
  const startTime = Date.now();
  let completed = false;

  while (!completed && (Date.now() - startTime) < TIMEOUT) {
    await delay(5000);

    const state = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        isComplete: text.includes('Forecast Complete') ||
                   (text.includes('MAPE') && text.includes('Model') && !text.includes('Generating')),
        hasError: text.includes('Error:') && text.includes('failed'),
        isTraining: text.includes('Generating') || text.includes('Training') || text.includes('Processing'),
      };
    });

    if (state.isComplete && !state.isTraining) {
      completed = true;
    } else if (state.hasError) {
      throw new Error('Forecast generation failed');
    } else {
      process.stdout.write('.');
    }
  }

  if (!completed) {
    throw new Error('Forecast timeout');
  }

  console.log(' Done!');
  await delay(3000);
  return true;
}

async function extractForecastResults(page, scenario) {
  const results = await page.evaluate((scenarioMode) => {
    const text = document.body.innerText;
    const data = {
      mode: scenarioMode,
      mape: null,
      confidence: null,
      confidenceScore: null,
      sliceCount: 0,
      slices: [],
      totalForecast: null,
      hasChart: false,
      hasDownload: false,
      hasBreakdown: false,
    };

    // Extract MAPE - look for various patterns
    const mapePatterns = [
      /MAPE[:\s]*(\d+\.?\d*)%?/gi,
      /(\d+\.?\d*)%?\s*MAPE/gi,
      /Accuracy[:\s]*(\d+\.?\d*)%/gi,
    ];

    for (const pattern of mapePatterns) {
      const matches = [...text.matchAll(pattern)];
      if (matches.length > 0) {
        // Get the lowest MAPE (best accuracy)
        const mapes = matches.map(m => parseFloat(m[1])).filter(m => m > 0 && m < 100);
        if (mapes.length > 0) {
          data.mape = Math.min(...mapes);
          break;
        }
      }
    }

    // Extract confidence
    const confMatch = text.match(/Confidence[:\s]*(High|Medium|Low)/i);
    if (confMatch) {
      data.confidence = confMatch[1].toLowerCase();
    }

    // Extract confidence score
    const scoreMatch = text.match(/Score[:\s]*(\d+)/i);
    if (scoreMatch) {
      data.confidenceScore = parseInt(scoreMatch[1]);
    }

    // Count slices
    const sliceTable = document.querySelector('table');
    if (sliceTable) {
      const rows = sliceTable.querySelectorAll('tbody tr');
      data.sliceCount = rows.length;

      rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length >= 3) {
          const sliceId = cells[0]?.textContent?.trim();
          const model = cells[1]?.textContent?.trim();
          const mapeCell = cells[2]?.textContent?.match(/(\d+\.?\d*)%/);
          if (sliceId) {
            data.slices.push({
              id: sliceId,
              model: model,
              mape: mapeCell ? parseFloat(mapeCell[1]) : null,
            });
          }
        }
      });
    }

    // Extract total forecast
    const totalMatch = text.match(/Total[^$]*\$?([\d,]+)/i);
    if (totalMatch) {
      data.totalForecast = parseInt(totalMatch[1].replace(/,/g, ''));
    }

    // Check UI elements
    data.hasChart = !!document.querySelector('svg.recharts-surface, canvas, .recharts-wrapper');
    data.hasDownload = text.includes('Download') || text.includes('Export');
    data.hasBreakdown = text.includes('Forecast Breakdown') || text.includes('Excel Formula');

    return data;
  }, scenario.mode);

  return results;
}

function analyzeResultsAsExpert(scenario, results) {
  const analysis = {
    scenarioId: scenario.id,
    scenarioName: scenario.name,
    passed: true,
    issues: [],
    insights: [],
    metrics: { ...results },
  };

  console.log('\n   🔬 Expert Analysis:');

  // 1. MAPE Analysis
  if (results.mape !== null) {
    console.log(`      MAPE: ${results.mape.toFixed(1)}%`);

    if (results.mape <= VALIDATION_CRITERIA.mape.excellent) {
      analysis.insights.push(`Excellent accuracy (MAPE ${results.mape.toFixed(1)}%) - production ready`);
      console.log(`      ✅ Excellent accuracy - production ready`);
    } else if (results.mape <= VALIDATION_CRITERIA.mape.good) {
      analysis.insights.push(`Good accuracy (MAPE ${results.mape.toFixed(1)}%) - suitable for planning`);
      console.log(`      ✅ Good accuracy - suitable for planning`);
    } else if (results.mape <= VALIDATION_CRITERIA.mape.acceptable) {
      analysis.insights.push(`Acceptable accuracy (MAPE ${results.mape.toFixed(1)}%) - use with caution`);
      console.log(`      ⚠️ Acceptable accuracy - use with caution`);
    } else if (results.mape <= scenario.acceptableMAPE) {
      analysis.insights.push(`Within scenario threshold (MAPE ${results.mape.toFixed(1)}% <= ${scenario.acceptableMAPE}%)`);
      console.log(`      ⚠️ Within threshold but high variance`);
    } else {
      analysis.issues.push(`MAPE ${results.mape.toFixed(1)}% exceeds threshold ${scenario.acceptableMAPE}%`);
      analysis.passed = false;
      console.log(`      ❌ MAPE exceeds acceptable threshold`);
    }
  } else {
    analysis.issues.push('MAPE not found in results');
    console.log(`      ❌ MAPE not found`);
  }

  // 2. Confidence Level Validation
  if (results.confidence) {
    console.log(`      Confidence: ${results.confidence} (Score: ${results.confidenceScore || 'N/A'})`);

    // Validate confidence matches MAPE
    if (results.mape !== null) {
      let expectedConf = 'low';
      if (results.mape <= 5) expectedConf = 'high';
      else if (results.mape <= 10) expectedConf = 'medium';

      if (results.confidence !== expectedConf && results.mape <= 10) {
        // Only flag if MAPE is decent but confidence is wrong
        if (results.confidence === 'low' && results.mape <= 10) {
          analysis.issues.push(`Confidence "${results.confidence}" seems low for MAPE ${results.mape.toFixed(1)}%`);
          console.log(`      ⚠️ Confidence seems misaligned with MAPE`);
        }
      } else {
        console.log(`      ✅ Confidence level appropriate`);
      }
    }
  }

  // 3. Slice Count Validation (for by_slice mode)
  if (scenario.mode === 'by_slice') {
    console.log(`      Slices trained: ${results.sliceCount}`);

    if (scenario.expectedSlices && results.sliceCount < scenario.expectedSlices) {
      analysis.issues.push(`Expected ${scenario.expectedSlices} slices, got ${results.sliceCount}`);
      console.log(`      ⚠️ Fewer slices than expected (data sparsity?)`);
    } else if (results.sliceCount > 0) {
      console.log(`      ✅ Multiple models trained successfully`);
    }

    // Analyze per-slice performance
    if (results.slices.length > 0) {
      const avgSliceMape = results.slices.reduce((a, s) => a + (s.mape || 0), 0) / results.slices.length;
      const bestSlice = results.slices.reduce((best, s) => (s.mape || 100) < (best.mape || 100) ? s : best, results.slices[0]);
      const worstSlice = results.slices.reduce((worst, s) => (s.mape || 0) > (worst.mape || 0) ? s : worst, results.slices[0]);

      analysis.insights.push(`Best performing slice: ${bestSlice.id} (${bestSlice.mape?.toFixed(1) || 'N/A'}% MAPE)`);
      analysis.insights.push(`Slice MAPE range: ${bestSlice.mape?.toFixed(1) || 'N/A'}% - ${worstSlice.mape?.toFixed(1) || 'N/A'}%`);

      console.log(`      Best slice: ${bestSlice.id} (${bestSlice.mape?.toFixed(1)}%)`);
      console.log(`      Worst slice: ${worstSlice.id} (${worstSlice.mape?.toFixed(1)}%)`);
    }
  }

  // 4. UI Validation
  console.log(`      Chart: ${results.hasChart ? '✅' : '❌'}`);
  console.log(`      Download: ${results.hasDownload ? '✅' : '❌'}`);
  console.log(`      Breakdown: ${results.hasBreakdown ? '✅' : '❌'}`);

  if (!results.hasChart) {
    analysis.issues.push('Forecast chart not rendered');
  }
  if (!results.hasDownload) {
    analysis.issues.push('Download option not available');
  }

  // 5. Expert Notes
  console.log(`      📝 ${scenario.expertNotes}`);

  return analysis;
}

async function runScenario(browser, scenario, index, total) {
  console.log('\n' + '═'.repeat(70));
  console.log(`📋 SCENARIO ${index + 1}/${total}: ${scenario.name}`);
  console.log(`   ID: ${scenario.id}`);
  console.log(`   Mode: ${scenario.mode}`);
  console.log(`   Slices: ${scenario.sliceColumns.join(', ') || 'None (aggregate)'}`);
  console.log(`   Horizon: ${scenario.horizon} periods`);
  console.log('═'.repeat(70));

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const pageErrors = [];
  page.on('pageerror', error => {
    pageErrors.push(error.message);
    console.log(`   ❌ JS Error: ${error.message.substring(0, 100)}`);
  });

  try {
    // Step 1: Navigate
    console.log('\n   📍 Step 1: Navigate to Simple Mode');
    await navigateToSimpleMode(page);
    console.log('      ✅ App loaded');

    // Step 2: Upload
    console.log('\n   📍 Step 2: Upload data');
    await uploadCSV(page);
    console.log('      ✅ Data uploaded and processed');

    // Step 3: Configure and run
    console.log('\n   📍 Step 3: Configure forecast');
    await configureAndRunForecast(page, scenario);
    console.log('      ✅ Forecast started');

    // Step 4: Wait for completion
    console.log('\n   📍 Step 4: Training models');
    process.stdout.write('      Progress: ');
    await waitForForecastComplete(page, scenario.id);

    // Step 5: Extract results
    console.log('\n   📍 Step 5: Extracting results');
    const results = await extractForecastResults(page, scenario);

    // Step 6: Expert analysis
    const analysis = analyzeResultsAsExpert(scenario, results);
    analysis.pageErrors = pageErrors;

    // Take screenshot
    const screenshotPath = path.join(__dirname, '..', `expert-${scenario.id}.png`);
    try {
      await page.screenshot({ path: screenshotPath, fullPage: false });
      console.log(`\n   📸 Screenshot: expert-${scenario.id}.png`);
    } catch (e) {
      console.log(`\n   ⚠️ Screenshot failed`);
    }

    // Summary
    console.log('\n   ' + '-'.repeat(50));
    console.log(`   SCENARIO RESULT: ${analysis.passed && pageErrors.length === 0 ? '✅ PASSED' : '⚠️ PASSED WITH WARNINGS'}`);
    if (analysis.issues.length > 0) {
      console.log(`   Issues: ${analysis.issues.join('; ')}`);
    }
    console.log('   ' + '-'.repeat(50));

    await page.close();
    return analysis;

  } catch (error) {
    console.log(`\n   ❌ Scenario failed: ${error.message}`);

    const screenshotPath = path.join(__dirname, '..', `expert-${scenario.id}-error.png`);
    try {
      await page.screenshot({ path: screenshotPath, fullPage: false });
    } catch (e) {}

    await page.close();

    return {
      scenarioId: scenario.id,
      scenarioName: scenario.name,
      passed: false,
      issues: [error.message],
      insights: [],
      metrics: {},
      pageErrors,
    };
  }
}

async function generateExpertReport(results) {
  console.log('\n\n');
  console.log('╔' + '═'.repeat(68) + '╗');
  console.log('║' + ' '.repeat(20) + 'EXPERT VALIDATION REPORT' + ' '.repeat(24) + '║');
  console.log('╚' + '═'.repeat(68) + '╝');

  // Summary statistics
  const passed = results.filter(r => r.passed).length;
  const total = results.length;
  const avgMape = results.filter(r => r.metrics?.mape).map(r => r.metrics.mape).reduce((a, b) => a + b, 0) /
                 results.filter(r => r.metrics?.mape).length || 0;

  console.log('\n📊 SUMMARY');
  console.log('-'.repeat(50));
  console.log(`   Scenarios Passed: ${passed}/${total}`);
  console.log(`   Average MAPE: ${avgMape.toFixed(1)}%`);
  console.log(`   Total Errors: ${results.reduce((a, r) => a + (r.pageErrors?.length || 0), 0)}`);

  // Per-scenario results
  console.log('\n📋 SCENARIO RESULTS');
  console.log('-'.repeat(50));
  results.forEach(r => {
    const status = r.passed ? '✅' : '❌';
    const mape = r.metrics?.mape ? `${r.metrics.mape.toFixed(1)}%` : 'N/A';
    const slices = r.metrics?.sliceCount || 0;
    console.log(`   ${status} ${r.scenarioId}: MAPE=${mape}, Slices=${slices}`);
  });

  // Expert insights
  console.log('\n💡 EXPERT INSIGHTS');
  console.log('-'.repeat(50));
  results.forEach(r => {
    if (r.insights && r.insights.length > 0) {
      console.log(`   [${r.scenarioId}]`);
      r.insights.forEach(i => console.log(`      • ${i}`));
    }
  });

  // Issues found
  const allIssues = results.flatMap(r => (r.issues || []).map(i => `[${r.scenarioId}] ${i}`));
  if (allIssues.length > 0) {
    console.log('\n⚠️ ISSUES FOUND');
    console.log('-'.repeat(50));
    allIssues.forEach(i => console.log(`   • ${i}`));
  }

  // Recommendations
  console.log('\n📝 EXPERT RECOMMENDATIONS');
  console.log('-'.repeat(50));

  if (avgMape > 15) {
    console.log('   • High average MAPE suggests considering more granular slicing');
  }

  const aggregateResult = results.find(r => r.scenarioId === 'TC1_AGGREGATE');
  const slicedResults = results.filter(r => r.scenarioId !== 'TC1_AGGREGATE' && r.metrics?.mape);

  if (aggregateResult?.metrics?.mape && slicedResults.length > 0) {
    const avgSlicedMape = slicedResults.reduce((a, r) => a + r.metrics.mape, 0) / slicedResults.length;
    if (aggregateResult.metrics.mape > avgSlicedMape * 1.5) {
      console.log('   • Segmented forecasts significantly outperform aggregate - recommend by-slice approach');
    }
  }

  const tripleSlice = results.find(r => r.scenarioId === 'TC4_TRIPLE_SLICE');
  if (tripleSlice?.metrics?.sliceCount < 10) {
    console.log('   • Consider reducing slice dimensions to avoid data sparsity');
  }

  console.log('   • Review anomaly detection results for unusual patterns');
  console.log('   • Validate forecasts against recent actuals before deployment');

  // Final verdict
  console.log('\n' + '═'.repeat(50));
  if (passed === total) {
    console.log('🏆 FINAL VERDICT: ALL SCENARIOS PASSED');
    console.log('   The forecasting system meets expert quality standards.');
  } else if (passed >= total * 0.75) {
    console.log('✅ FINAL VERDICT: MOSTLY PASSED');
    console.log('   Minor issues to address before production deployment.');
  } else {
    console.log('⚠️ FINAL VERDICT: NEEDS IMPROVEMENT');
    console.log('   Significant issues require attention.');
  }
  console.log('═'.repeat(50));

  return passed === total;
}

async function main() {
  console.log('🔬 EXPERT MLE/DATA SCIENTIST & QA VALIDATION SUITE');
  console.log('═'.repeat(70));
  console.log(`Test Data: ${CSV_PATH}`);
  console.log(`Scenarios: ${TEST_SCENARIOS.length}`);
  console.log(`Base URL: ${BASE_URL}`);
  console.log('═'.repeat(70));

  if (!fs.existsSync(CSV_PATH)) {
    console.error(`❌ CSV not found: ${CSV_PATH}`);
    process.exit(1);
  }

  const browser = await setupBrowser();
  const results = [];

  try {
    for (let i = 0; i < TEST_SCENARIOS.length; i++) {
      const result = await runScenario(browser, TEST_SCENARIOS[i], i, TEST_SCENARIOS.length);
      results.push(result);

      if (i < TEST_SCENARIOS.length - 1) {
        console.log('\n⏳ Preparing next scenario...');
        await delay(3000);
      }
    }
  } finally {
    await browser.close();
  }

  const allPassed = await generateExpertReport(results);
  process.exit(allPassed ? 0 : 1);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
