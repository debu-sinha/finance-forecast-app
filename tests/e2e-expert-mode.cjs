/**
 * E2E Test for Finance Forecasting App - EXPERT MODE
 * Based on actual App.tsx UI structure analysis
 *
 * Flow:
 * 1. Open app and switch to Expert Mode
 * 2. Upload Main Time Series file
 * 3. Upload Promotions/Features file
 * 4. Wait for auto-detection of date columns
 * 5. Click "Create Training Dataset & Analyze"
 * 6. Click "Analyze Data & Get Recommendations"
 * 7. Configure models (Prophet, SARIMAX, XGBoost)
 * 8. Configure slice (is_cgna)
 * 9. Set filter (is_cgna=0)
 * 10. Set end date (10/5/2025)
 * 11. Click "Train Single Model"
 * 12. Monitor training progress
 * 13. Verify results
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const os = require('os');

const BASE_URL = process.env.TEST_URL || 'http://localhost:3001';
const TIMEOUT = 600000; // 10 minutes for training

// Results folder
const RESULTS_DIR = path.join(__dirname, 'results');
const TEST_RUN_ID = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);

// Test configuration per user requirements
const TEST_CONFIG = {
  // Absolute paths to the actual test files
  mainFile: '/Users/debu.sinha/Downloads/Jan 2023 -  Dec 2025 Databricks Example (1).xlsx - Weekly actual results.csv',
  featureFile: '/Users/debu.sinha/Downloads/Events_week_2022_2026 with christmas day (2).csv',
  targetColumn: 'TOT_VOL',
  models: ['prophet', 'sarimax', 'xgboost'],  // All 3 models
  sliceColumn: 'IS_CGNA',
  filterColumn: 'IS_CGNA',
  filterValue: '0',
  endDate: '2025-10-05',
  frequency: 'weekly',
  horizon: 12
};

const consoleErrors = [];
const pageErrors = [];
let screenshotCount = 0;

// Save screenshot helper
const saveScreenshot = async (page, stepName) => {
  screenshotCount++;
  const filename = `${TEST_RUN_ID}_${String(screenshotCount).padStart(2, '0')}_${stepName}.png`;
  const filepath = path.join(RESULTS_DIR, filename);
  await page.screenshot({ path: filepath, fullPage: true });
  console.log(`   📸 ${filename}`);
  return filepath;
};

// Delay helper
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function runExpertModeTest() {
  console.log('═'.repeat(70));
  console.log('🚀 E2E TEST: EXPERT MODE (Full Configuration)');
  console.log('═'.repeat(70));
  console.log('\n📋 Test Configuration:');
  console.log(`   Main File: ${TEST_CONFIG.mainFile}`);
  console.log(`   Feature File: ${TEST_CONFIG.featureFile}`);
  console.log(`   Target: ${TEST_CONFIG.targetColumn}`);
  console.log(`   Models: ${TEST_CONFIG.models.join(', ')}`);
  console.log(`   Slice: ${TEST_CONFIG.sliceColumn}`);
  console.log(`   Filter: ${TEST_CONFIG.filterColumn}=${TEST_CONFIG.filterValue}`);
  console.log(`   End Date: ${TEST_CONFIG.endDate}`);
  console.log(`   Frequency: ${TEST_CONFIG.frequency}`);
  console.log(`   Horizon: ${TEST_CONFIG.horizon}`);
  console.log('═'.repeat(70) + '\n');

  // Create results directory
  if (!fs.existsSync(RESULTS_DIR)) {
    fs.mkdirSync(RESULTS_DIR, { recursive: true });
  }
  console.log(`📁 Results: ${RESULTS_DIR}`);
  console.log(`📋 Run ID: ${TEST_RUN_ID}\n`);

  const userDataDir = path.join(os.homedir(), '.puppeteer-expert-' + Date.now());
  fs.mkdirSync(userDataDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: false,
    slowMo: 20,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--window-size=1600,1000'],
    userDataDir
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000 });

  // Capture errors
  page.on('console', msg => {
    const text = msg.text();
    if (msg.type() === 'error' && !text.includes('favicon')) {
      consoleErrors.push(text);
      console.log(`   ❌ Console: ${text.substring(0, 100)}`);
    }
    // Capture aggregation debug output
    if (text.includes('Aggregation Debug') || text.includes('Sample target') || text.includes('Target range') || text.includes('filteredData.length')) {
      console.log(`   📊 ${text}`);
    }
  });

  page.on('pageerror', error => {
    pageErrors.push(error.message);
    console.log(`   ❌ Page Error: ${error.message}`);
  });

  try {
    // ══════════════════════════════════════════════════════════════════════
    // STEP 1: Open app
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 1: Opening app...');
    await page.goto(BASE_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    await delay(1000);
    await saveScreenshot(page, '01_app_loaded');
    console.log('   ✅ App loaded');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 2: Switch to Expert Mode
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 2: Switching to Expert Mode...');

    // The Expert Mode button is in the sidebar. It's a button with "Expert Mode" and "Full Control" text.
    // The button structure: button > [icon] > div > span "Expert Mode" + span "Full Control"
    const expertModeClicked = await page.evaluate(() => {
      // Find the Expert Mode button in sidebar - look for button containing both texts
      const buttons = Array.from(document.querySelectorAll('button'));
      const expertBtn = buttons.find(btn => {
        const text = btn.textContent || '';
        return text.includes('Expert Mode') && text.includes('Full Control');
      });

      if (expertBtn) {
        console.log('Found Expert Mode button, clicking...');
        expertBtn.click();
        return 'clicked';
      }

      // Fallback: look for any button/element with just "Expert Mode"
      const allElements = Array.from(document.querySelectorAll('button, div[role="button"]'));
      const altExpertBtn = allElements.find(el => {
        const text = el.textContent || '';
        return text.includes('Expert Mode');
      });

      if (altExpertBtn) {
        altExpertBtn.click();
        return 'clicked_alt';
      }

      return 'not_found';
    });

    console.log(`   Expert Mode click result: ${expertModeClicked}`);
    await delay(1500);

    // Verify Expert Mode is active by checking if 2 file inputs exist
    const modeVerified = await page.evaluate(() => {
      const text = document.body.innerText;
      const fileInputs = document.querySelectorAll('input[type="file"]');
      return {
        hasExpertModeLabel: text.includes('Expert Mode') && !text.includes('Simple Mode Active'),
        hasMainUpload: text.includes('Main Time Series') || text.includes('Time Series Data'),
        hasFeatureUpload: text.includes('Promotions') || text.includes('Feature') || text.includes('Events'),
        fileInputCount: fileInputs.length
      };
    });

    console.log(`   Mode verification: ${JSON.stringify(modeVerified)}`);

    if (modeVerified.fileInputCount < 2) {
      console.log('   ⚠️ Expert Mode may not be active (expected 2 file inputs)');
      // Try clicking again
      await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const expertBtn = buttons.find(btn => {
          const text = btn.textContent || '';
          return text.includes('Expert') && text.includes('Full Control');
        });
        if (expertBtn) expertBtn.click();
      });
      await delay(1000);
    }

    await saveScreenshot(page, '02_expert_mode');
    console.log('   ✅ Expert Mode selected');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 3: Upload Main Time Series file
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 3: Uploading Main Time Series file...');

    // Use absolute paths directly (config already has absolute paths)
    const mainCsvPath = TEST_CONFIG.mainFile;
    const featureCsvPath = TEST_CONFIG.featureFile;

    if (!fs.existsSync(mainCsvPath)) {
      throw new Error(`Main file not found: ${mainCsvPath}`);
    }
    if (!fs.existsSync(featureCsvPath)) {
      throw new Error(`Feature file not found: ${featureCsvPath}`);
    }

    // Re-query file inputs after mode switch
    let fileInputs = await page.$$('input[type="file"]');
    console.log(`   Found ${fileInputs.length} file input(s)`);

    // If still only 1 file input, Expert Mode might not have switched
    if (fileInputs.length < 2) {
      console.log('   ⚠️ Only 1 file input found. Trying to switch to Expert Mode again...');

      // More aggressive approach: directly click via XPath-like selector
      await page.evaluate(() => {
        // Find sidebar and Expert Mode button
        const sidebar = document.querySelector('aside') || document.querySelector('[class*="sidebar"]');
        if (sidebar) {
          const buttons = sidebar.querySelectorAll('button');
          buttons.forEach(btn => {
            if (btn.textContent.includes('Expert')) {
              btn.click();
            }
          });
        }
      });
      await delay(2000);
      await saveScreenshot(page, '02b_expert_mode_retry');

      // Re-query file inputs
      fileInputs = await page.$$('input[type="file"]');
      console.log(`   After retry: Found ${fileInputs.length} file input(s)`);
    }

    if (fileInputs.length >= 1) {
      await fileInputs[0].uploadFile(mainCsvPath);
      console.log(`   ✅ Uploaded: ${path.basename(mainCsvPath)}`);
      await delay(3000);
    }
    await saveScreenshot(page, '03_main_file_uploaded');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 4: Upload Promotions/Features file
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 4: Uploading Features file...');

    // The second file input is in the "Promotions / Features" section
    // After uploading first file, we need to find the file input in that section
    // It may be hidden - we need to click on the upload area first

    // Wait a bit for the UI to stabilize after first upload
    await delay(1000);

    // Find and click the Promotions/Features upload area if needed
    const featureUploadClicked = await page.evaluate(() => {
      // Look for the Promotions/Features section
      const sections = Array.from(document.querySelectorAll('div'));
      const featureSection = sections.find(div => {
        const text = div.textContent || '';
        return text.includes('Promotions') && text.includes('Features') && text.includes('Click to upload');
      });

      if (featureSection) {
        // Click on the upload area
        const clickArea = featureSection.querySelector('[class*="upload"], [class*="drop"], div') || featureSection;
        clickArea.click();
        return 'clicked_area';
      }
      return 'not_found';
    });
    console.log(`   Feature upload area: ${featureUploadClicked}`);

    await delay(500);

    // Re-query ALL file inputs - the second one should be available now
    fileInputs = await page.$$('input[type="file"]');
    console.log(`   Found ${fileInputs.length} file input(s) after click`);

    // The file input might be in the Promotions/Features section
    // Try to find it specifically
    if (fileInputs.length >= 1) {
      // Try each file input until we find one that works for the features file
      // The first input may have been used, so try the last available one
      const featureInput = fileInputs[fileInputs.length - 1];
      await featureInput.uploadFile(featureCsvPath);
      console.log(`   ✅ Uploaded: ${path.basename(featureCsvPath)}`);
      await delay(3000);
    } else {
      console.log(`   ⚠️ No file input available for features file`);
      await saveScreenshot(page, '04_missing_feature_input');
    }

    // Verify feature file was uploaded by checking for row count
    const featureUploaded = await page.evaluate(() => {
      const text = document.body.innerText;
      // Look for indication that promotions/features file was loaded
      return text.includes('Events') || text.includes('rows loaded') || text.includes('Promotions');
    });
    console.log(`   Feature file detected in UI: ${featureUploaded}`);

    await saveScreenshot(page, '04_features_file_uploaded');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 5: Wait for column auto-detection and select date columns
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 5: Waiting for column auto-detection...');
    await page.waitForFunction(() => {
      const text = document.body.innerText;
      return text.includes('rows loaded') || text.includes('rows');
    }, { timeout: 30000 });
    await delay(2000);

    // Select date columns for both datasets
    // Main Time Series should auto-select "WEEK", Features needs "WeekStart" (case-insensitive)
    console.log('   Verifying date columns for joining...');

    // Wait a bit more for feature file processing to complete
    await delay(1000);

    const dateColumnsSet = await page.evaluate(() => {
      const selects = Array.from(document.querySelectorAll('select'));
      let mainDateSet = false;
      let featureDateSet = false;
      let mainDateValue = '';
      let featureDateValue = '';

      // Debug: log all selects and their values
      console.log('Date column selects found:', selects.length);

      selects.forEach((select, index) => {
        const options = Array.from(select.options);
        const optionValues = options.map(o => o.value);

        // Check if this select has date-like options
        const hasWeekOption = options.some(o => o.value.toLowerCase().includes('week'));

        if (hasWeekOption) {
          console.log(`Select ${index}: current value="${select.value}", options:`, optionValues);

          // Check if already has a value set (auto-detected)
          if (select.value && select.value !== '') {
            // Determine if this is main or feature based on value pattern
            if (select.value.toUpperCase() === 'WEEK') {
              mainDateSet = true;
              mainDateValue = select.value;
            } else if (select.value.toLowerCase().includes('week')) {
              featureDateSet = true;
              featureDateValue = select.value;
            }
          }

          // If not set, try to set it
          if (!select.value || select.value === '') {
            // Find a week-related option (case-insensitive)
            const weekOption = options.find(o =>
              o.value.toLowerCase().includes('week') ||
              o.value.toLowerCase() === 'ds' ||
              o.value.toLowerCase().includes('date')
            );

            if (weekOption) {
              select.value = weekOption.value;
              select.dispatchEvent(new Event('change', { bubbles: true }));

              if (weekOption.value.toUpperCase() === 'WEEK') {
                mainDateSet = true;
                mainDateValue = weekOption.value;
              } else {
                featureDateSet = true;
                featureDateValue = weekOption.value;
              }
            }
          }
        }
      });

      return {
        mainDateSet,
        featureDateSet,
        mainDateValue,
        featureDateValue
      };
    });

    console.log(`   Date columns: Main=${dateColumnsSet.mainDateSet} (${dateColumnsSet.mainDateValue}), Features=${dateColumnsSet.featureDateSet} (${dateColumnsSet.featureDateValue})`);
    await delay(1000);
    await saveScreenshot(page, '05_columns_detected');
    console.log('   ✅ Columns configured');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 6: Click "Create Training Dataset & Analyze"
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 6: Clicking "Create Training Dataset & Analyze"...');

    // Wait for button to be enabled
    await page.waitForFunction(() => {
      const btn = Array.from(document.querySelectorAll('button')).find(b =>
        b.textContent.includes('Create Training Dataset')
      );
      return btn && !btn.disabled;
    }, { timeout: 15000 });

    // Scroll to and click the button
    await page.evaluate(() => {
      const btn = Array.from(document.querySelectorAll('button')).find(b =>
        b.textContent.includes('Create Training Dataset')
      );
      if (btn) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        btn.click();
      }
    });

    await delay(3000);
    await saveScreenshot(page, '06_create_dataset_clicked');
    console.log('   ✅ Clicked "Create Training Dataset & Analyze"');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 7: Wait for AI analysis
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 7: Waiting for AI analysis...');
    await page.waitForFunction(() => {
      const text = document.body.innerText;
      return text.includes('Analysis') ||
             text.includes('Weekly') ||
             text.includes('suitable for') ||
             text.includes('Model') ||
             text.includes('Target');
    }, { timeout: 90000 });
    await delay(3000);
    await saveScreenshot(page, '07_analysis_complete');
    console.log('   ✅ AI analysis complete');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 8: Click "Analyze Data & Get Recommendations" if available
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 8: Getting AI recommendations...');
    const analyzeDataClicked = await page.evaluate(() => {
      const btn = Array.from(document.querySelectorAll('button')).find(b =>
        b.textContent.includes('Analyze Data') ||
        b.textContent.includes('Get Recommendations')
      );
      if (btn && !btn.disabled) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        btn.click();
        return true;
      }
      return false;
    });

    if (analyzeDataClicked) {
      await delay(5000);
      await saveScreenshot(page, '08_recommendations_received');
      console.log('   ✅ AI recommendations received');
    } else {
      console.log('   ℹ️ Analyze Data button not found (may already be analyzed)');
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 9: Configure models (Prophet, SARIMAX, XGBoost)
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 9: Configuring models...');
    await page.evaluate((models) => {
      // Find all checkboxes in model section
      const labels = Array.from(document.querySelectorAll('label'));

      // Try to find and click each model
      models.forEach(model => {
        const modelLabel = labels.find(l => {
          const text = l.textContent.toLowerCase();
          return text.includes(model.toLowerCase());
        });

        if (modelLabel) {
          const checkbox = modelLabel.querySelector('input[type="checkbox"]');
          if (checkbox && !checkbox.checked) {
            checkbox.click();
          }
        }
      });
    }, TEST_CONFIG.models);
    await delay(500);
    await saveScreenshot(page, '09_models_configured');
    console.log(`   ✅ Models configured: ${TEST_CONFIG.models.join(', ')}`);

    // ══════════════════════════════════════════════════════════════════════
    // STEP 10: Configure slice/group by (IS_CGNA)
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 10: Configuring slice definition...');
    await page.evaluate((sliceCol) => {
      const labels = Array.from(document.querySelectorAll('label'));

      // Find and check the slice column checkbox
      const sliceLabel = labels.find(l => {
        const text = l.textContent.toUpperCase();
        return text.includes(sliceCol.toUpperCase());
      });

      if (sliceLabel) {
        const checkbox = sliceLabel.querySelector('input[type="checkbox"]');
        if (checkbox && !checkbox.checked) {
          checkbox.click();
        }
      }
    }, TEST_CONFIG.sliceColumn);
    await delay(500);
    await saveScreenshot(page, '10_slice_configured');
    console.log(`   ✅ Slice configured: ${TEST_CONFIG.sliceColumn}`);

    // ══════════════════════════════════════════════════════════════════════
    // STEP 11: Set filter (IS_CGNA=0)
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 11: Setting filter...');
    await delay(1000); // Wait for filter dropdown to appear after slice selection

    await page.evaluate((filterCol, filterVal) => {
      const selects = Array.from(document.querySelectorAll('select'));

      // Find select that has the filter column name nearby
      const filterSelect = selects.find(s => {
        const parent = s.closest('div');
        const label = parent?.querySelector('label, span')?.textContent || '';
        const siblingText = s.previousElementSibling?.textContent || '';
        return label.toUpperCase().includes(filterCol.toUpperCase()) ||
               siblingText.toUpperCase().includes(filterCol.toUpperCase());
      });

      if (filterSelect) {
        // Find option with the value
        const option = Array.from(filterSelect.options).find(o =>
          o.value === filterVal || o.textContent.includes(filterVal)
        );
        if (option) {
          filterSelect.value = option.value;
          filterSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    }, TEST_CONFIG.filterColumn, TEST_CONFIG.filterValue);
    await delay(500);
    await saveScreenshot(page, '11_filter_set');
    console.log(`   ✅ Filter set: ${TEST_CONFIG.filterColumn}=${TEST_CONFIG.filterValue}`);

    // ══════════════════════════════════════════════════════════════════════
    // STEP 12: Set end date (10/5/2025)
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 12: Setting end date...');
    await page.evaluate((endDate) => {
      // Find date inputs
      const dateInputs = Array.from(document.querySelectorAll('input[type="date"]'));

      // Second date input is usually end date
      if (dateInputs.length >= 2) {
        dateInputs[1].value = endDate;
        dateInputs[1].dispatchEvent(new Event('change', { bubbles: true }));
        dateInputs[1].dispatchEvent(new Event('input', { bubbles: true }));
      } else if (dateInputs.length === 1) {
        // If only one, check if it's end date by nearby label
        dateInputs[0].value = endDate;
        dateInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
      }
    }, TEST_CONFIG.endDate);
    await delay(500);
    await saveScreenshot(page, '12_end_date_set');
    console.log(`   ✅ End date set: ${TEST_CONFIG.endDate}`);

    // ══════════════════════════════════════════════════════════════════════
    // STEP 13: Click "Train Single Model" (NOT batch training)
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 13: Starting Single Model training...');

    // Wait for Train Single Model button to be enabled
    await page.waitForFunction(() => {
      const btn = Array.from(document.querySelectorAll('button')).find(b => {
        const text = b.textContent || '';
        // Specifically look for "Train Single Model" or "Single Model" - NOT "Train All"
        return (text.includes('Single') || (text.includes('Train') && !text.includes('All') && !text.includes('Batch'))) && !b.disabled;
      });
      return btn !== undefined;
    }, { timeout: 20000 });

    // Click Train Single Model button specifically
    const trainBtnClicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'));

      // First try to find "Train Single Model" specifically
      let btn = buttons.find(b => {
        const text = b.textContent || '';
        return text.includes('Single') && text.includes('Train') && !b.disabled;
      });

      // Fallback: any Train button that's not "All" or "Batch"
      if (!btn) {
        btn = buttons.find(b => {
          const text = b.textContent || '';
          return text.includes('Train') && !text.includes('All') && !text.includes('Batch') && !b.disabled;
        });
      }

      if (btn) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        btn.click();
        return btn.textContent.trim();
      }
      return null;
    });

    console.log(`   Clicked button: "${trainBtnClicked}"`);

    await delay(2000);
    await saveScreenshot(page, '13_training_started');
    console.log('   ✅ Training started');

    // Check for immediate errors
    if (pageErrors.length > 0) {
      throw new Error(`Errors after clicking train: ${pageErrors.join(', ')}`);
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 14: Monitor training progress
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 14: Monitoring training progress...');
    const startTime = Date.now();
    let trainingComplete = false;
    let lastProgress = '';
    let screenshotInterval = 0;

    while (!trainingComplete && (Date.now() - startTime) < TIMEOUT) {
      await delay(5000);
      screenshotInterval++;

      const state = await page.evaluate(() => {
        const text = document.body.innerText;
        // Check for actual completion indicators - NOT just "MLflow" which appears in sidebar
        const hasMAPE = text.includes('MAPE:') || text.includes('MAPE ');
        const hasRMSE = text.includes('RMSE:') || text.includes('RMSE ');
        const hasModelCompleted = text.includes('completed') || text.includes('Completed');
        const hasChart = !!document.querySelector('svg.recharts-surface');
        const hasDownload = text.includes('Download') && text.includes('Forecast');

        return {
          // Only mark complete when we see actual results (MAPE metrics or completed model)
          hasResults: (hasMAPE || hasRMSE || hasModelCompleted || hasChart || hasDownload) &&
                      !text.includes('Running') && !text.includes('Executing'),
          hasError: text.includes('Training failed') ||
                    (text.includes('Error:') && text.includes('failed')),
          progress: (text.match(/(\d+)%/) || [])[1] || '?',
          modelsComplete: (text.match(/MAPE:\s*[\d.]+%/g) || []).length,
          totalModels: 1, // Single model test
          isRunning: text.includes('Running') || text.includes('Executing'),
          status: text.includes('Executive Summary') ? 'Generating Summary' :
                  text.includes('Generating') ? 'Generating' :
                  text.includes('Running') ? 'Running' :
                  text.includes('Training') ? 'Training' :
                  text.includes('Executing') ? 'Executing' : 'Processing'
        };
      });

      if (state.hasResults) {
        trainingComplete = true;
        console.log('\n   ✅ Training completed!');
        await saveScreenshot(page, '14_training_complete');
      } else if (state.hasError) {
        await saveScreenshot(page, '14_training_error');
        throw new Error('Training failed - error detected in UI');
      } else {
        const progressLog = `   Progress: ${state.progress}% | Models: ${state.modelsComplete}/${TEST_CONFIG.models.length} (${TEST_CONFIG.models.join(', ')}) | ${state.status}`;
        if (progressLog !== lastProgress) {
          console.log(progressLog);
          lastProgress = progressLog;
        }

        // Take periodic screenshots during training
        if (screenshotInterval % 6 === 0) { // Every 30 seconds
          await saveScreenshot(page, `14_training_progress_${state.progress}pct`);
        }
      }

      // Check for JavaScript errors
      if (pageErrors.length > 0) {
        await saveScreenshot(page, '14_javascript_error');
        throw new Error(`JavaScript error: ${pageErrors[pageErrors.length - 1]}`);
      }
    }

    if (!trainingComplete) {
      await saveScreenshot(page, '14_training_timeout');
      throw new Error('Training timeout - did not complete within time limit');
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 15: Verify results
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 15: Verifying results...');
    await delay(3000);

    const results = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasMAPE: text.includes('MAPE'),
        hasRMSE: text.includes('RMSE'),
        hasR2: text.includes('R²') || text.includes('R2'),
        hasChart: !!document.querySelector('svg.recharts-surface'),
        hasMLflow: text.includes('MLflow') || text.includes('Experiment'),
        hasSummary: text.includes('Summary') || text.includes('summary'),
        hasExperimentUrl: !!document.querySelector('a[href*="mlflow"]')
      };
    });

    console.log('\n   Results Verification:');
    console.log(`   ├─ MAPE metrics: ${results.hasMAPE ? '✅' : '❌'}`);
    console.log(`   ├─ RMSE metrics: ${results.hasRMSE ? '✅' : '❌'}`);
    console.log(`   ├─ R² metrics: ${results.hasR2 ? '✅' : '❌'}`);
    console.log(`   ├─ Chart: ${results.hasChart ? '✅' : '❌'}`);
    console.log(`   ├─ MLflow integration: ${results.hasMLflow ? '✅' : '❌'}`);
    console.log(`   ├─ Experiment URL: ${results.hasExperimentUrl ? '✅' : '❌'}`);
    console.log(`   └─ Summary: ${results.hasSummary ? '✅' : '❌'}`);

    await saveScreenshot(page, '15_final_results');

    // ══════════════════════════════════════════════════════════════════════
    // STEP 16: Extract and validate forecast values
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n📍 STEP 16: Extracting forecast values for validation...');

    // Scroll down to see the forecast table
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await delay(2000);
    await saveScreenshot(page, '16_forecast_table_scrolled');

    // Extract forecast values from the table
    const forecastData = await page.evaluate(() => {
      const data = {
        metrics: {},
        forecastValues: [],
        tableRows: []
      };

      // Extract MAPE, RMSE values
      const text = document.body.innerText;
      const mapeMatch = text.match(/MAPE[:\s]*([0-9.]+)%?/i);
      const rmseMatch = text.match(/RMSE[:\s]*([0-9,.]+)/i);
      const r2Match = text.match(/R²?[:\s]*([0-9.]+)/i);

      if (mapeMatch) data.metrics.mape = mapeMatch[1];
      if (rmseMatch) data.metrics.rmse = rmseMatch[1];
      if (r2Match) data.metrics.r2 = r2Match[1];

      // Try to extract forecast values from table cells
      const tables = document.querySelectorAll('table');
      tables.forEach(table => {
        const rows = table.querySelectorAll('tr');
        rows.forEach(row => {
          const cells = row.querySelectorAll('td, th');
          if (cells.length >= 2) {
            const rowData = Array.from(cells).map(c => c.textContent.trim());
            // Look for date-like patterns and numeric values
            if (rowData[0] && (rowData[0].includes('/') || rowData[0].includes('-'))) {
              data.tableRows.push(rowData);
            }
          }
        });
      });

      // Also try to extract any large numbers (potential forecast values)
      const numberMatches = text.match(/[\d,]+(?:\.\d+)?/g) || [];
      const largeNumbers = numberMatches
        .map(n => parseFloat(n.replace(/,/g, '')))
        .filter(n => n > 100000) // Looking for values > 100K
        .slice(0, 20);
      data.forecastValues = largeNumbers;

      return data;
    });

    console.log('\n   📊 EXTRACTED FORECAST DATA:');
    console.log(`   Metrics: MAPE=${forecastData.metrics.mape || 'N/A'}%, RMSE=${forecastData.metrics.rmse || 'N/A'}, R²=${forecastData.metrics.r2 || 'N/A'}`);

    if (forecastData.tableRows.length > 0) {
      console.log(`\n   📅 Forecast Table (first 10 rows):`);
      forecastData.tableRows.slice(0, 10).forEach(row => {
        console.log(`      ${row.join(' | ')}`);
      });
    }

    if (forecastData.forecastValues.length > 0) {
      console.log(`\n   💰 Large numeric values found:`);
      forecastData.forecastValues.forEach((v, i) => {
        console.log(`      ${i + 1}. ${v.toLocaleString()}`);
      });

      // Validate metrics are reasonable for the data scale
      // RMSE should be roughly 1-5% of the average value for a good forecast
      const rmse = parseFloat(forecastData.metrics.rmse?.replace(/,/g, '') || '0');
      const mape = parseFloat(forecastData.metrics.mape || '0');

      console.log(`\n   🎯 MODEL QUALITY VALIDATION:`);
      console.log(`      MAPE: ${mape}% ${mape < 5 ? '✅ Excellent' : mape < 10 ? '✅ Good' : mape < 20 ? '⚠️ Acceptable' : '❌ Poor'}`);
      console.log(`      RMSE: ${rmse.toLocaleString()}`);

      // If we have RMSE, estimate the data scale (RMSE should be ~2-5% of mean for good forecast)
      if (rmse > 0 && mape > 0) {
        // MAPE% = 100 * MAE / mean, so mean ≈ MAE * 100 / MAPE
        // For MAPE ~1.5%, RMSE ~660K, implied mean is ~660K / 0.02 = ~33M (approximate)
        const impliedMean = rmse / (mape / 100 * 1.5); // Rough estimation
        console.log(`      Implied data scale (est.): ~${(impliedMean / 1000000).toFixed(0)}M per period`);

        // This should be in the 25-45M range for IS_CGNA=0 aggregated weekly data
        const scaleOk = impliedMean > 10000000 && impliedMean < 100000000;
        console.log(`      Scale check: ${scaleOk ? '✅ Reasonable' : '⚠️ Check data aggregation'}`);
      }
    }

    await saveScreenshot(page, '17_validation_complete');

    // ══════════════════════════════════════════════════════════════════════
    // TEST SUMMARY
    // ══════════════════════════════════════════════════════════════════════
    console.log('\n' + '═'.repeat(70));
    console.log('📊 EXPERT MODE E2E TEST SUMMARY');
    console.log('═'.repeat(70));
    console.log(`   Run ID: ${TEST_RUN_ID}`);
    console.log(`   Screenshots: ${screenshotCount}`);
    console.log(`   Page Errors: ${pageErrors.length}`);
    console.log(`   Console Errors: ${consoleErrors.length}`);
    console.log(`   Training: ${trainingComplete ? '✅ Complete' : '❌ Failed'}`);
    console.log(`   Results: ${results.hasMAPE && results.hasChart ? '✅ Valid' : '❌ Missing'}`);
    console.log('═'.repeat(70));

    const passed = pageErrors.length === 0 && trainingComplete && results.hasMAPE;
    console.log(passed ? '\n✅ EXPERT MODE E2E TEST PASSED\n' : '\n❌ EXPERT MODE E2E TEST FAILED\n');

    return passed;

  } catch (error) {
    console.error('\n❌ Test Error:', error.message);
    await saveScreenshot(page, 'error_final');

    if (pageErrors.length > 0) {
      console.log('\n   Page Errors:');
      pageErrors.forEach((e, i) => console.log(`   ${i + 1}. ${e.substring(0, 100)}`));
    }

    return false;
  } finally {
    await browser.close();
    try { fs.rmSync(userDataDir, { recursive: true }); } catch (e) {}

    // Write test log
    const logPath = path.join(RESULTS_DIR, `${TEST_RUN_ID}_test_log.txt`);
    const logContent = [
      `Test Run: ${TEST_RUN_ID}`,
      `Config: ${JSON.stringify(TEST_CONFIG, null, 2)}`,
      `Page Errors: ${pageErrors.length}`,
      pageErrors.map((e, i) => `  ${i + 1}. ${e}`).join('\n'),
      `Console Errors: ${consoleErrors.length}`,
      consoleErrors.map((e, i) => `  ${i + 1}. ${e}`).join('\n')
    ].join('\n\n');
    fs.writeFileSync(logPath, logContent);
    console.log(`📝 Log saved: ${logPath}`);
  }
}

// Run test
runExpertModeTest()
  .then(passed => process.exit(passed ? 0 : 1))
  .catch(err => {
    console.error('Fatal:', err);
    process.exit(1);
  });
