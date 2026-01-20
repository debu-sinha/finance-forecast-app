/**
 * Comprehensive E2E Tests for Finance Forecasting App - SIMPLE MODE
 * QA Lead: Multiple test scenarios covering all Simple Mode functionality
 *
 * Test Data: Paula_merged_data_for_simple_mode.csv
 * Columns:
 *   - WEEK (date column)
 *   - IS_CGNA, BUSINESS_SEGMENT, MX_TYPE (slice/segment columns)
 *   - TOT_VOL, TOT_SUB, AVG_SUB, AVG_ITEM_PRICE, AVG_ITEM_CT (numeric columns)
 *   - Holiday columns (Super Bowl, Valentine's Day, etc.)
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const os = require('os');

const BASE_URL = process.env.TEST_URL || 'http://localhost:3002';
const TIMEOUT = 420000; // 7 minutes for training
const CSV_PATH = '/Users/debu.sinha/Downloads/Paula_merged_data_for_simple_mode.csv';

// Test configurations - start with one test for debugging
const TEST_CASES = [
  {
    name: 'TC1: Basic Aggregate Forecast - TOT_VOL',
    description: 'Test basic aggregate forecast with TOT_VOL as target',
    targetColumn: 'TOT_VOL',
    forecastMode: 'aggregate',
    horizon: 12,
    expectedMetrics: { minMAPE: 0, maxMAPE: 50 },
  },
  {
    name: 'TC2: By-Slice Forecast - BUSINESS_SEGMENT',
    description: 'Test by-slice forecast segmented by BUSINESS_SEGMENT',
    targetColumn: 'TOT_VOL',
    forecastMode: 'by_slice',
    sliceColumns: ['BUSINESS_SEGMENT'],
    horizon: 8,
    expectedMetrics: { minMAPE: 0, maxMAPE: 50 },
  },
  {
    name: 'TC3: Multi-Slice Forecast - IS_CGNA + MX_TYPE',
    description: 'Test by-slice forecast with multiple slice columns',
    targetColumn: 'TOT_VOL',
    forecastMode: 'by_slice',
    sliceColumns: ['IS_CGNA', 'MX_TYPE'],
    horizon: 10,
    expectedMetrics: { minMAPE: 0, maxMAPE: 60 },
  },
];

// Track errors across all tests
let globalConsoleErrors = [];
let globalPageErrors = [];

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function setupBrowser() {
  const userDataDir = path.join(os.homedir(), '.puppeteer-simple-comprehensive');
  if (!fs.existsSync(userDataDir)) {
    fs.mkdirSync(userDataDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: false,
    slowMo: 50, // Slower for visibility
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    userDataDir,
    protocolTimeout: 60000, // 60 second protocol timeout for screenshots
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  return { browser, page };
}

async function switchToSimpleMode(page) {
  console.log('   Ensuring Simple Mode is active...');

  // Check current mode and switch if needed
  const switched = await page.evaluate(() => {
    // Look for Simple Mode tab/button in the sidebar
    const sidebar = document.querySelector('[class*="sidebar"]') || document.body;
    const allElements = sidebar.querySelectorAll('*');

    for (const el of allElements) {
      const text = el.textContent || '';
      if (text.includes('Simple Mode') && !text.includes('Expert')) {
        // Check if it's clickable
        if (el.tagName === 'BUTTON' || el.tagName === 'DIV' || el.tagName === 'SPAN') {
          const parent = el.closest('button') || el.closest('[role="button"]') || el.closest('.cursor-pointer') || el;
          if (parent) {
            parent.click();
            return 'clicked';
          }
        }
      }
    }

    // Check if already in Simple Mode by looking for autopilot content
    const bodyText = document.body.innerText;
    if (bodyText.includes('Autopilot') || bodyText.includes('Upload your data')) {
      return 'already_simple';
    }

    return 'not_found';
  });

  await delay(1500);
  return switched;
}

async function uploadCSV(page, csvPath) {
  console.log('   Uploading CSV file...');

  if (!fs.existsSync(csvPath)) {
    throw new Error(`CSV not found: ${csvPath}`);
  }

  // Find file input
  const fileInput = await page.$('input[type="file"]');
  if (!fileInput) {
    throw new Error('File input not found');
  }

  await fileInput.uploadFile(csvPath);
  console.log('   File selected, waiting for processing...');

  await delay(3000);

  // Wait for data to be loaded and configure step to appear
  await page.waitForFunction(() => {
    const text = document.body.innerText;
    return text.includes('Data Loaded Successfully') ||
           text.includes('Column Overview') ||
           text.includes('Configure Your Forecast');
  }, { timeout: 90000 });

  console.log('   CSV uploaded and processed');
  await delay(2000);
}

async function selectTargetColumn(page, targetColumn) {
  console.log(`   Selecting target column: ${targetColumn}...`);

  // Scroll to target selection area
  await page.evaluate(() => {
    const targetSection = document.querySelector('[class*="target"]') ||
                         Array.from(document.querySelectorAll('*')).find(el =>
                           el.textContent && el.textContent.includes('Target Column'));
    if (targetSection) {
      targetSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });

  await delay(500);

  // Find and click on the target column dropdown/select
  const selected = await page.evaluate((target) => {
    // Method 1: Look for select element
    const selects = document.querySelectorAll('select');
    for (const select of selects) {
      const parentText = select.closest('div')?.innerText || '';
      if (parentText.toLowerCase().includes('target')) {
        const options = select.querySelectorAll('option');
        for (const option of options) {
          if (option.textContent.includes(target) || option.value === target) {
            select.value = option.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            return 'select_changed';
          }
        }
      }
    }

    // Method 2: Look for clickable column buttons/pills
    const buttons = document.querySelectorAll('button, [role="button"], .cursor-pointer');
    for (const btn of buttons) {
      if (btn.textContent && btn.textContent.trim() === target) {
        btn.click();
        return 'button_clicked';
      }
    }

    // Method 3: Look in numeric columns section
    const numericSection = Array.from(document.querySelectorAll('*')).find(el =>
      el.textContent && el.textContent.includes('Numeric Columns'));
    if (numericSection) {
      const targetBtn = numericSection.querySelector(`[data-column="${target}"]`) ||
                       Array.from(numericSection.querySelectorAll('*')).find(el =>
                         el.textContent === target && el.classList.contains('cursor-pointer'));
      if (targetBtn) {
        targetBtn.click();
        return 'numeric_section_clicked';
      }
    }

    return 'not_found';
  }, targetColumn);

  await delay(500);
  console.log(`   Target selection result: ${selected}`);
  return selected;
}

async function setForecastMode(page, mode, sliceColumns) {
  console.log(`   Setting forecast mode to: ${mode}...`);

  // Scroll to forecasting approach section
  await page.evaluate(() => {
    const section = Array.from(document.querySelectorAll('*')).find(el =>
      el.textContent && el.textContent.includes('Choose Forecasting Approach'));
    if (section) {
      section.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });

  await delay(500);

  if (mode === 'by_slice') {
    // Click on "Forecast by Slice" option
    const clicked = await page.evaluate(() => {
      const options = document.querySelectorAll('[class*="cursor-pointer"], button, [role="button"]');
      for (const opt of options) {
        const text = opt.textContent || '';
        if (text.includes('Forecast by Slice') || text.includes('By Slice') || text.includes('Per Segment')) {
          opt.click();
          return true;
        }
      }

      // Try finding radio button or checkbox for by_slice
      const radios = document.querySelectorAll('input[type="radio"], input[type="checkbox"]');
      for (const radio of radios) {
        const label = radio.closest('label')?.textContent || radio.closest('div')?.textContent || '';
        if (label.includes('Slice') || label.includes('Segment')) {
          radio.click();
          return true;
        }
      }

      return false;
    });

    if (clicked && sliceColumns && sliceColumns.length > 0) {
      await delay(500);
      await selectSliceColumns(page, sliceColumns);
    }
  } else {
    // Aggregate mode - click on aggregate option if not already selected
    await page.evaluate(() => {
      const options = document.querySelectorAll('[class*="cursor-pointer"], button, [role="button"]');
      for (const opt of options) {
        const text = opt.textContent || '';
        if (text.includes('Aggregate') || text.includes('Total Forecast')) {
          opt.click();
          return true;
        }
      }
      return false;
    });
  }

  await delay(500);
}

async function selectSliceColumns(page, sliceColumns) {
  console.log(`   Selecting slice columns: ${sliceColumns.join(', ')}...`);

  // First, scroll to the Data Segments section
  await page.evaluate(() => {
    const segmentHeaders = document.querySelectorAll('h3, h4, .font-semibold, .font-medium');
    for (const h of segmentHeaders) {
      if (h.textContent && (h.textContent.includes('Data Segments') || h.textContent.includes('Segment'))) {
        h.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }
  });

  await delay(500);

  for (const col of sliceColumns) {
    const result = await page.evaluate((colName) => {
      // The slice columns are displayed as clickable cards with the column name
      // Look for cards that contain the column name
      const cards = document.querySelectorAll('.cursor-pointer, [class*="rounded-lg"][class*="border"]');

      for (const card of cards) {
        const cardText = card.textContent || '';

        // Check if this card contains the column name (could be exact match or part of text)
        if (cardText.includes(colName)) {
          // Check if it's a slice column card (has border and is clickable)
          const isClickableCard = card.classList.contains('cursor-pointer') ||
                                 card.className.includes('cursor-pointer');

          if (isClickableCard) {
            // Check if already selected (has purple border)
            const isSelected = card.className.includes('border-purple') ||
                              card.className.includes('bg-purple');

            if (!isSelected) {
              card.click();
              return `card_clicked_${colName}`;
            } else {
              return `already_selected_${colName}`;
            }
          }
        }
      }

      // Fallback: look for any element with exact column name text
      const allElements = document.querySelectorAll('*');
      for (const el of allElements) {
        if (el.childElementCount === 0 && el.textContent && el.textContent.trim() === colName) {
          // Find the closest clickable parent
          const clickable = el.closest('.cursor-pointer') ||
                           el.closest('[class*="rounded-lg"][class*="border"]');
          if (clickable) {
            clickable.click();
            return `parent_clicked_${colName}`;
          }
        }
      }

      return `not_found_${colName}`;
    }, col);

    console.log(`     Slice column ${col}: ${result}`);
    await delay(400);
  }
}

async function setHorizon(page, horizon) {
  console.log(`   Setting horizon to: ${horizon}...`);

  // Scroll to the forecast settings section (bottom of configure page)
  await page.evaluate(() => {
    // Look for "Forecast Horizon" label
    const labels = document.querySelectorAll('label');
    for (const label of labels) {
      if (label.textContent && label.textContent.includes('Forecast Horizon')) {
        label.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }
    // Fallback: scroll to bottom
    window.scrollTo(0, document.body.scrollHeight);
  });

  await delay(500);

  const set = await page.evaluate((h) => {
    // Find the horizon input - it's near a label containing "Forecast Horizon"
    const labels = document.querySelectorAll('label');
    for (const label of labels) {
      if (label.textContent && label.textContent.includes('Forecast Horizon')) {
        // Find the input in the same container
        const container = label.closest('div');
        if (container) {
          const input = container.querySelector('input[type="number"]');
          if (input) {
            input.value = h;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return 'found_by_label';
          }
        }
      }
    }

    // Fallback: find any number input with small width (horizon input is w-20)
    const inputs = document.querySelectorAll('input[type="number"]');
    for (const input of inputs) {
      // Check if it looks like a horizon input (small, near "periods" or "weeks" text)
      const nextSibling = input.nextElementSibling;
      const siblingText = nextSibling?.textContent?.toLowerCase() || '';
      if (siblingText.includes('week') || siblingText.includes('period') || siblingText.includes('day')) {
        input.value = h;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return 'found_by_sibling';
      }
    }

    return 'not_found';
  }, horizon);

  await delay(300);
  console.log(`   Horizon setting result: ${set}`);
  return set;
}

async function clickGenerateForecast(page) {
  console.log('   Looking for Generate Forecast button...');

  // Scroll to bottom to find the button
  await page.evaluate(() => {
    window.scrollTo(0, document.body.scrollHeight);
  });

  await delay(1000);

  // Find and click the Generate Forecast button
  const clicked = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));

    // Look for specific button text
    const generateBtn = buttons.find(btn => {
      const text = (btn.textContent || '').toLowerCase();
      return (text.includes('generate forecast') ||
              text.includes('run forecast') ||
              text.includes('start forecast')) &&
             !btn.disabled;
    });

    if (generateBtn) {
      console.log('Found Generate Forecast button:', generateBtn.textContent);
      generateBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
      generateBtn.click();
      return 'generate_clicked';
    }

    // Fallback: look for any green/primary action button at the bottom
    const actionBtns = buttons.filter(btn => {
      const classes = btn.className || '';
      return (classes.includes('bg-green') || classes.includes('bg-blue') ||
              classes.includes('primary')) && !btn.disabled;
    });

    if (actionBtns.length > 0) {
      const lastBtn = actionBtns[actionBtns.length - 1];
      lastBtn.click();
      return 'action_btn_clicked';
    }

    return 'not_found';
  });

  await delay(2000);
  console.log(`   Button click result: ${clicked}`);
  return clicked;
}

async function waitForTrainingComplete(page, testName) {
  console.log('   Waiting for training to complete...');

  const startTime = Date.now();
  let trainingComplete = false;
  let lastProgress = '';
  let checkCount = 0;

  while (!trainingComplete && (Date.now() - startTime) < TIMEOUT) {
    await delay(5000);
    checkCount++;

    const state = await page.evaluate(() => {
      const text = document.body.innerText;

      // Check for definitive completion indicators
      // "Forecast Complete" is the main header when done
      const hasForecastComplete = text.includes('Forecast Complete');

      // Check for results sections that only appear when done
      const hasResultsSections = text.includes('How We Built Your Forecast') ||
                                 text.includes('How We Selected the Best Model') ||
                                 text.includes('Forecast Visualization') ||
                                 text.includes('New Forecast'); // Reset button appears when done

      // Check for model comparison table (appears only in results)
      const hasModelComparison = text.includes('Model Performance Comparison') ||
                                 (text.includes('Prophet') && text.includes('SARIMAX') && text.includes('MAPE'));

      // Check for active training indicators
      const isActivelyTraining = text.includes('Generating forecast') ||
                                 text.includes('Training models') ||
                                 text.includes('Processing data') ||
                                 (text.includes('Running') && !text.includes('mlruns'));

      // Check for errors
      const hasError = text.includes('Error:') && text.includes('failed');

      // Check for progress percentage in training context
      const progressMatch = text.match(/(\d+)%/);
      const progress = progressMatch ? progressMatch[1] : null;

      // Determine if complete
      const isComplete = hasForecastComplete ||
                        (hasResultsSections && hasModelComparison && !isActivelyTraining);

      return {
        isComplete,
        hasForecastComplete,
        hasResultsSections,
        hasModelComparison,
        isActivelyTraining,
        hasError,
        progress,
      };
    });

    if (state.isComplete) {
      trainingComplete = true;
      console.log('   ✓ Training completed!');
    } else if (state.hasError) {
      console.log('   ✗ Training error detected');
      throw new Error('Training failed with error');
    } else {
      const progress = state.progress ? `${state.progress}%` : (state.isActivelyTraining ? 'training' : 'waiting');
      if (progress !== lastProgress || checkCount % 6 === 0) {
        console.log(`   Progress: ${progress}`);
        lastProgress = progress;
      }
    }

    // Log debug info every 30 seconds
    if (checkCount % 6 === 0) {
      console.log(`   [Debug] Check #${checkCount}, complete=${state.isComplete}, forecastComplete=${state.hasForecastComplete}, activeTraining=${state.isActivelyTraining}`);
    }
  }

  if (!trainingComplete) {
    // Take a screenshot before throwing
    console.log('   Training timeout - capturing final state');
    throw new Error('Training timeout');
  }

  return true;
}

async function validateResults(page, testConfig) {
  console.log('   Validating results...');
  await delay(3000);

  const results = await page.evaluate(() => {
    const text = document.body.innerText;

    // Extract MAPE value - look for various formats
    let mape = null;
    const mapePatterns = [
      /MAPE[:\s]*(\d+\.?\d*)%?/i,
      /(\d+\.?\d*)%?\s*MAPE/i,
      /Accuracy[:\s]*(\d+\.?\d*)%/i
    ];

    for (const pattern of mapePatterns) {
      const match = text.match(pattern);
      if (match) {
        mape = parseFloat(match[1]);
        break;
      }
    }

    return {
      hasMAPE: text.includes('MAPE') || text.includes('Accuracy'),
      mapeValue: mape,
      hasChart: !!document.querySelector('svg.recharts-surface, canvas, .recharts-wrapper, svg'),
      hasDownload: text.includes('Download') || text.includes('Export'),
      hasSummary: text.includes('Summary') || text.includes('Results'),
      hasConfidence: text.includes('Confidence') || text.includes('confidence'),
      hasForecastValues: !!text.match(/[\d,]+\.?\d*[KMB]?/), // Has formatted numbers
    };
  });

  console.log(`     MAPE displayed: ${results.hasMAPE ? '✓' : '✗'} ${results.mapeValue ? `(${results.mapeValue}%)` : ''}`);
  console.log(`     Chart rendered: ${results.hasChart ? '✓' : '✗'}`);
  console.log(`     Download option: ${results.hasDownload ? '✓' : '✗'}`);
  console.log(`     Summary section: ${results.hasSummary ? '✓' : '✗'}`);

  // Validate MAPE is within expected range
  if (results.mapeValue !== null) {
    const { minMAPE, maxMAPE } = testConfig.expectedMetrics;
    if (results.mapeValue < minMAPE || results.mapeValue > maxMAPE) {
      console.log(`     ⚠️ MAPE ${results.mapeValue}% outside expected range [${minMAPE}-${maxMAPE}]`);
    } else {
      console.log(`     ✓ MAPE ${results.mapeValue}% within expected range`);
    }
  }

  return results;
}

async function runSingleTest(browser, testConfig, testIndex, totalTests) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const consoleErrors = [];
  const pageErrors = [];

  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('favicon')) {
      consoleErrors.push(msg.text());
      globalConsoleErrors.push(`[${testConfig.name}] ${msg.text()}`);
    }
  });

  page.on('pageerror', error => {
    pageErrors.push(error.message);
    globalPageErrors.push(`[${testConfig.name}] ${error.message}`);
    console.log(`   ❌ Page Error: ${error.message.substring(0, 100)}`);
  });

  console.log('\n' + '='.repeat(70));
  console.log(`TEST ${testIndex + 1}/${totalTests}: ${testConfig.name}`);
  console.log(`Description: ${testConfig.description}`);
  console.log('='.repeat(70));

  try {
    // Step 1: Navigate to app
    console.log('\n📍 Step 1: Opening app...');
    await page.goto(BASE_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    await delay(2000);

    // Step 2: Ensure Simple Mode
    console.log('\n📍 Step 2: Switching to Simple Mode...');
    const modeState = await switchToSimpleMode(page);
    console.log(`   Mode state: ${modeState}`);
    await delay(1000);

    // Step 3: Upload CSV
    console.log('\n📍 Step 3: Uploading CSV...');
    await uploadCSV(page, CSV_PATH);

    // Step 4: Configure forecast
    console.log('\n📍 Step 4: Configuring forecast...');

    // Select target column
    await selectTargetColumn(page, testConfig.targetColumn);
    await delay(500);

    // Set horizon
    await setHorizon(page, testConfig.horizon);
    await delay(500);

    // Set forecast mode
    await setForecastMode(page, testConfig.forecastMode, testConfig.sliceColumns);
    await delay(1000);

    // Check for any errors before running
    if (pageErrors.length > 0) {
      throw new Error(`Errors before training: ${pageErrors.join(', ')}`);
    }

    // Step 5: Run forecast
    console.log('\n📍 Step 5: Running forecast...');
    const clicked = await clickGenerateForecast(page);

    if (clicked === 'not_found') {
      console.log('   ⚠️ Generate button not found, taking screenshot...');
      await page.screenshot({
        path: path.join(__dirname, '..', `simple-mode-tc${testIndex + 1}-no-button.png`),
        fullPage: true
      });
      throw new Error('Generate Forecast button not found');
    }

    // Step 6: Wait for training
    console.log('\n📍 Step 6: Monitoring training...');
    await waitForTrainingComplete(page, testConfig.name);

    // Step 7: Validate results
    console.log('\n📍 Step 7: Validating results...');
    const results = await validateResults(page, testConfig);

    // Take screenshot (with error handling)
    const screenshotName = `simple-mode-tc${testIndex + 1}-${testConfig.targetColumn}-success.png`;
    try {
      await page.screenshot({
        path: path.join(__dirname, '..', screenshotName),
        fullPage: false, // Use viewport only to avoid timeout
      });
      console.log(`   📸 Screenshot: ${screenshotName}`);
    } catch (screenshotErr) {
      console.log(`   ⚠️ Screenshot failed: ${screenshotErr.message}`);
    }

    // Determine pass/fail
    const passed = results.hasMAPE && results.hasChart && pageErrors.length === 0;

    console.log('\n' + '-'.repeat(40));
    console.log(`TEST RESULT: ${passed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`  Page Errors: ${pageErrors.length}`);
    console.log(`  Console Errors: ${consoleErrors.length}`);
    console.log('-'.repeat(40));

    await page.close();

    return {
      name: testConfig.name,
      passed,
      results,
      errors: pageErrors,
      consoleErrors,
    };

  } catch (error) {
    console.error(`\n❌ Test Error: ${error.message}`);

    // Take error screenshot (with error handling)
    const errorScreenshot = `simple-mode-tc${testIndex + 1}-error.png`;
    try {
      await page.screenshot({
        path: path.join(__dirname, '..', errorScreenshot),
        fullPage: false,
      });
      console.log(`   📸 Error screenshot: ${errorScreenshot}`);
    } catch (screenshotErr) {
      console.log(`   ⚠️ Error screenshot failed: ${screenshotErr.message}`);
    }

    await page.close();

    return {
      name: testConfig.name,
      passed: false,
      error: error.message,
      errors: pageErrors,
      consoleErrors,
    };
  }
}

async function runAllTests() {
  console.log('🚀 COMPREHENSIVE E2E TESTS: SIMPLE MODE');
  console.log('='.repeat(70));
  console.log(`Test File: ${CSV_PATH}`);
  console.log(`Total Test Cases: ${TEST_CASES.length}`);
  console.log(`Base URL: ${BASE_URL}`);
  console.log('='.repeat(70));

  // Verify CSV exists
  if (!fs.existsSync(CSV_PATH)) {
    console.error(`❌ CSV file not found: ${CSV_PATH}`);
    process.exit(1);
  }

  const { browser } = await setupBrowser();
  const testResults = [];

  try {
    for (let i = 0; i < TEST_CASES.length; i++) {
      const result = await runSingleTest(browser, TEST_CASES[i], i, TEST_CASES.length);
      testResults.push(result);

      // Brief pause between tests
      if (i < TEST_CASES.length - 1) {
        console.log('\n⏳ Waiting before next test...');
        await delay(3000);
      }
    }

  } finally {
    await browser.close();
  }

  // Final Summary
  console.log('\n' + '='.repeat(70));
  console.log('📊 FINAL TEST SUMMARY');
  console.log('='.repeat(70));

  const passed = testResults.filter(r => r.passed).length;
  const failed = testResults.filter(r => !r.passed).length;

  testResults.forEach((result, i) => {
    const status = result.passed ? '✅' : '❌';
    const mape = result.results?.mapeValue ? ` (MAPE: ${result.results.mapeValue}%)` : '';
    console.log(`  ${status} TC${i + 1}: ${result.name}${mape}`);
    if (result.error) {
      console.log(`     Error: ${result.error}`);
    }
  });

  console.log('\n' + '-'.repeat(40));
  console.log(`TOTAL: ${passed}/${TEST_CASES.length} PASSED`);
  console.log(`Global Page Errors: ${globalPageErrors.length}`);
  console.log(`Global Console Errors: ${globalConsoleErrors.length}`);
  console.log('-'.repeat(40));

  if (globalPageErrors.length > 0) {
    console.log('\nPage Errors:');
    globalPageErrors.slice(0, 5).forEach((e, i) => console.log(`  ${i + 1}. ${e.substring(0, 200)}`));
  }

  const allPassed = failed === 0;
  console.log(allPassed ? '\n✅ ALL TESTS PASSED\n' : '\n❌ SOME TESTS FAILED\n');

  return allPassed;
}

// Run tests
runAllTests()
  .then(passed => process.exit(passed ? 0 : 1))
  .catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
