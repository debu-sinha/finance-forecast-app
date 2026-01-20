/**
 * E2E Test for Finance Forecasting App
 * Tests the full flow: Upload -> Configure -> Train -> Results
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const os = require('os');

const BASE_URL = process.env.TEST_URL || 'http://localhost:3001';
const TIMEOUT = 420000; // 7 minutes for training

// Test configuration - matching manual test settings
const TEST_CONFIG = {
  csvFile: 'datasets/processed/full_merged_data.csv',
  timeColumn: 'WEEK',
  targetColumn: 'NET_ORDERED_PRODUCT_SALES_AMOUNT',
  horizon: 8,
  frequency: 'weekly',
  models: ['prophet', 'sarimax', 'xgboost']
};

// Collect all console errors
const consoleErrors = [];
const pageErrors = [];

async function runE2ETest() {
  console.log('🚀 Starting E2E Test...');
  console.log('📋 Config:', JSON.stringify(TEST_CONFIG, null, 2), '\n');

  // Use home directory for Chrome profile to avoid permission issues
  const userDataDir = path.join(os.homedir(), '.puppeteer-test-profile');
  if (!fs.existsSync(userDataDir)) {
    fs.mkdirSync(userDataDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: false,
    slowMo: 30,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    userDataDir
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  // Helper for delays (waitForTimeout is deprecated)
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

  // Capture console messages
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    if (type === 'error') {
      consoleErrors.push(text);
      console.log(`❌ Console Error: ${text}`);
    } else if (type === 'warning') {
      console.log(`⚠️  Console Warning: ${text}`);
    }
  });

  // Capture page errors (uncaught exceptions)
  page.on('pageerror', error => {
    pageErrors.push(error.message);
    console.log(`❌ Page Error: ${error.message}`);
  });

  try {
    // Step 1: Navigate to app
    console.log('📍 Step 1: Opening app...');
    await page.goto(BASE_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    console.log('✅ App loaded\n');

    // Check for initial render errors
    if (pageErrors.length > 0) {
      throw new Error(`Initial render errors: ${pageErrors.join(', ')}`);
    }

    // Step 2: Click Expert Mode if available
    console.log('📍 Step 2: Selecting Expert Mode...');
    try {
      const expertModeBtn = await page.$('button:has-text("Expert Mode")');
      if (expertModeBtn) {
        await expertModeBtn.click();
        await delay(500);
      }
    } catch (e) {
      console.log('   Expert Mode button not found or already active');
    }

    // Step 3: Upload CSV file
    console.log('📍 Step 3: Uploading CSV file...');
    const csvPath = path.join(__dirname, '../datasets/processed/full_merged_data.csv');

    if (!fs.existsSync(csvPath)) {
      throw new Error(`CSV file not found: ${csvPath}`);
    }

    // Find file input
    const fileInput = await page.$('input[type="file"]');
    if (!fileInput) {
      throw new Error('File input not found');
    }

    await fileInput.uploadFile(csvPath);
    console.log('   Uploaded:', csvPath);

    // Wait for file processing
    await delay(2000);
    console.log('✅ CSV uploaded and processed\n');

    // Step 4: Wait for analysis to complete
    console.log('📍 Step 4: Waiting for data analysis...');
    await page.waitForFunction(() => {
      const summaryEl = document.querySelector('[class*="summary"], [class*="analysis"]');
      return summaryEl && summaryEl.textContent.length > 50;
    }, { timeout: 60000 });
    console.log('✅ Analysis complete\n');

    // Step 5: Configure training (select columns if needed)
    console.log('📍 Step 5: Configuring training...');

    // Check if columns are auto-selected
    await delay(1000);

    // Look for column selectors and ensure they're set
    const timeColSelect = await page.$('select[id*="time"], select[name*="time"]');
    const targetColSelect = await page.$('select[id*="target"], select[name*="target"]');

    if (timeColSelect) {
      console.log('   Found time column selector');
    }
    if (targetColSelect) {
      console.log('   Found target column selector');
    }

    console.log('✅ Configuration ready\n');

    // Step 6: Start training
    console.log('📍 Step 6: Starting training...');

    // Find and click the train/run button
    const trainButton = await page.evaluateHandle(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      return buttons.find(btn =>
        btn.textContent.includes('Train') ||
        btn.textContent.includes('Run') ||
        btn.textContent.includes('Start')
      );
    });

    if (trainButton) {
      await trainButton.click();
      console.log('   Clicked training button');
    } else {
      // Try clicking by PlayCircle icon
      const playBtn = await page.$('button:has(svg)');
      if (playBtn) {
        await playBtn.click();
      }
    }

    // Wait for training to start
    await delay(2000);

    // Check for errors after clicking train
    if (pageErrors.length > 0) {
      console.log('❌ Errors after clicking train:', pageErrors);
    }

    console.log('✅ Training started\n');

    // Step 7: Wait for training to complete
    console.log('📍 Step 7: Waiting for training to complete (this may take several minutes)...');

    const startTime = Date.now();
    let trainingComplete = false;

    while (!trainingComplete && (Date.now() - startTime) < TIMEOUT) {
      await delay(5000);

      // Check for results view or completion indicators
      const resultsVisible = await page.evaluate(() => {
        const text = document.body.innerText;
        return text.includes('Forecast Results') ||
               text.includes('Model Registered') ||
               text.includes('Executive Summary');
      });

      if (resultsVisible) {
        trainingComplete = true;
        console.log('✅ Training completed!\n');
      } else {
        // Check progress
        const progress = await page.evaluate(() => {
          const progressText = document.body.innerText.match(/(\d+)%/);
          return progressText ? progressText[1] : null;
        });
        if (progress) {
          process.stdout.write(`\r   Progress: ${progress}%`);
        }
      }

      // Check for errors during training
      if (pageErrors.length > 0) {
        const newErrors = pageErrors.slice(-3);
        console.log('\n❌ Errors during training:', newErrors);
        break;
      }
    }

    if (!trainingComplete) {
      throw new Error('Training did not complete within timeout');
    }

    // Step 8: Verify results view
    console.log('📍 Step 8: Verifying results view...');
    await delay(2000);

    // Check for key elements in results
    const hasMetrics = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes('MAPE') || text.includes('RMSE') || text.includes('R²');
    });

    const hasChart = await page.$('svg.recharts-surface, canvas');

    if (hasMetrics) {
      console.log('   ✅ Metrics displayed');
    } else {
      console.log('   ⚠️  Metrics not found');
    }

    if (hasChart) {
      console.log('   ✅ Chart rendered');
    } else {
      console.log('   ⚠️  Chart not found');
    }

    // Final error check
    if (pageErrors.length > 0) {
      console.log('\n❌ Total page errors:', pageErrors.length);
      pageErrors.forEach((err, i) => console.log(`   ${i + 1}. ${err}`));
    }

    if (consoleErrors.length > 0) {
      console.log('\n⚠️  Total console errors:', consoleErrors.length);
    }

    // Take screenshot of final state
    const screenshotPath = path.join(__dirname, '../test-results-screenshot.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`\n📸 Screenshot saved: ${screenshotPath}`);

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 E2E TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`   Page Errors: ${pageErrors.length}`);
    console.log(`   Console Errors: ${consoleErrors.length}`);
    console.log(`   Training Completed: ${trainingComplete ? 'Yes' : 'No'}`);
    console.log(`   Results Displayed: ${hasMetrics ? 'Yes' : 'No'}`);
    console.log('='.repeat(60));

    if (pageErrors.length === 0 && trainingComplete && hasMetrics) {
      console.log('\n✅ E2E TEST PASSED\n');
      return true;
    } else {
      console.log('\n❌ E2E TEST FAILED\n');
      return false;
    }

  } catch (error) {
    console.error('\n❌ E2E Test Error:', error.message);

    // Take error screenshot
    const errorScreenshot = path.join(__dirname, '../test-error-screenshot.png');
    await page.screenshot({ path: errorScreenshot, fullPage: true });
    console.log(`📸 Error screenshot saved: ${errorScreenshot}`);

    return false;
  } finally {
    await browser.close();
  }
}

// Run the test
runE2ETest()
  .then(passed => {
    process.exit(passed ? 0 : 1);
  })
  .catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
