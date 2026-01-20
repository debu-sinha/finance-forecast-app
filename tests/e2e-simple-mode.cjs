/**
 * E2E Test for Finance Forecasting App - SIMPLE MODE (Autopilot)
 * Tests the streamlined autopilot flow
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const os = require('os');

const BASE_URL = process.env.TEST_URL || 'http://localhost:3001';
const TIMEOUT = 420000; // 7 minutes for training

const consoleErrors = [];
const pageErrors = [];

async function runSimpleModeTest() {
  console.log('🚀 E2E TEST: SIMPLE MODE (Autopilot)');
  console.log('='.repeat(60));

  const userDataDir = path.join(os.homedir(), '.puppeteer-simple-test');
  if (!fs.existsSync(userDataDir)) {
    fs.mkdirSync(userDataDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: false,
    slowMo: 20,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    userDataDir
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('favicon')) {
      consoleErrors.push(msg.text());
      console.log(`❌ Console: ${msg.text().substring(0, 150)}`);
    }
  });

  page.on('pageerror', error => {
    pageErrors.push(error.message);
    console.log(`❌ Page Error: ${error.message}`);
  });

  try {
    // Step 1: Open app
    console.log('\n📍 Step 1: Opening app...');
    await page.goto(BASE_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    await delay(1000);
    console.log('✅ App loaded');

    // Step 2: Ensure Simple Mode is active
    console.log('\n📍 Step 2: Ensuring Simple Mode...');
    const modeState = await page.evaluate(() => {
      const text = document.body.innerText;
      const isSimple = text.includes('Simple Mode') || text.includes('Autopilot');

      if (!isSimple) {
        const buttons = Array.from(document.querySelectorAll('button, div[role="button"]'));
        const simpleBtn = buttons.find(b =>
          b.textContent.includes('Simple') ||
          b.textContent.includes('Autopilot')
        );
        if (simpleBtn) {
          simpleBtn.click();
          return 'switched';
        }
      }
      return isSimple ? 'already_simple' : 'unknown';
    });

    await delay(1000);
    console.log(`✅ Mode: ${modeState}`);

    // Step 3: Upload CSV
    console.log('\n📍 Step 3: Uploading CSV...');
    const csvPath = path.resolve(__dirname, '../datasets/processed/full_merged_data.csv');

    if (!fs.existsSync(csvPath)) {
      throw new Error(`CSV not found: ${csvPath}`);
    }

    const fileInput = await page.$('input[type="file"]');
    if (fileInput) {
      await fileInput.uploadFile(csvPath);
    }

    await delay(3000);
    console.log('✅ CSV uploaded');

    // Step 4: Wait for autopilot analysis
    console.log('\n📍 Step 4: Waiting for autopilot analysis...');
    await page.waitForFunction(() => {
      const text = document.body.innerText;
      return text.includes('Configuration') ||
             text.includes('Ready') ||
             text.includes('weekly') ||
             text.includes('Prophet') ||
             text.includes('recommended');
    }, { timeout: 90000 });

    await delay(2000);
    console.log('✅ Autopilot analysis complete');

    // Step 5: Check autopilot recommendations
    console.log('\n📍 Step 5: Checking autopilot config...');
    const autopilotConfig = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasFrequency: text.includes('weekly') || text.includes('daily') || text.includes('monthly'),
        hasModels: text.includes('Prophet') || text.includes('XGBoost'),
        hasHorizon: text.match(/horizon[:\s]*(\d+)/i),
        hasRunButton: !!Array.from(document.querySelectorAll('button')).find(b =>
          b.textContent.includes('Run') ||
          b.textContent.includes('Train') ||
          b.textContent.includes('Start')
        )
      };
    });

    console.log(`   Frequency detected: ${autopilotConfig.hasFrequency ? '✅' : '⚠️'}`);
    console.log(`   Models selected: ${autopilotConfig.hasModels ? '✅' : '⚠️'}`);
    console.log(`   Run button: ${autopilotConfig.hasRunButton ? '✅' : '⚠️'}`);

    // Step 6: Start autopilot training
    console.log('\n📍 Step 6: Starting autopilot training...');

    const trainClicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      const trainBtn = buttons.find(btn => {
        const text = btn.textContent || '';
        return (text.includes('Run') || text.includes('Train') || text.includes('Start') || text.includes('Forecast')) &&
               !btn.disabled;
      });

      if (trainBtn) {
        trainBtn.click();
        return true;
      }
      return false;
    });

    await delay(2000);

    if (pageErrors.length > 0) {
      throw new Error(`Errors after train: ${pageErrors.join(', ')}`);
    }

    console.log(trainClicked ? '✅ Training started' : '⚠️ Could not find train button');

    // Step 7: Monitor training
    console.log('\n📍 Step 7: Monitoring training...');
    const startTime = Date.now();
    let trainingComplete = false;
    let lastLog = '';

    while (!trainingComplete && (Date.now() - startTime) < TIMEOUT) {
      await delay(5000);

      const state = await page.evaluate(() => {
        const text = document.body.innerText;
        return {
          hasResults: text.includes('Results') ||
                      text.includes('Forecast Complete') ||
                      text.includes('Download'),
          hasError: text.includes('failed') || text.includes('Error'),
          progress: (text.match(/(\d+)%/) || [])[1],
          status: text.includes('Training') ? 'training' :
                  text.includes('Summary') ? 'summary' : 'processing'
        };
      });

      if (state.hasResults) {
        trainingComplete = true;
        console.log('\n✅ Training completed!');
      } else if (state.hasError) {
        throw new Error('Training failed');
      } else {
        const log = `   Progress: ${state.progress || '?'}% | Status: ${state.status}`;
        if (log !== lastLog) {
          console.log(log);
          lastLog = log;
        }
      }

      if (pageErrors.length > 0) {
        console.log(`\n❌ JS Error: ${pageErrors[pageErrors.length - 1]}`);
        await page.screenshot({
          path: path.join(__dirname, '../simple-error-screenshot.png'),
          fullPage: true
        });
        throw new Error(`JS Error: ${pageErrors[pageErrors.length - 1]}`);
      }
    }

    if (!trainingComplete) {
      throw new Error('Training timeout');
    }

    // Step 8: Verify results
    console.log('\n📍 Step 8: Verifying results...');
    await delay(3000);

    const results = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasMAPE: text.includes('MAPE'),
        hasChart: !!document.querySelector('svg.recharts-surface, canvas'),
        hasDownload: text.includes('Download') || text.includes('Export'),
        hasSummary: text.includes('Summary')
      };
    });

    console.log(`   MAPE: ${results.hasMAPE ? '✅' : '❌'}`);
    console.log(`   Chart: ${results.hasChart ? '✅' : '❌'}`);
    console.log(`   Download: ${results.hasDownload ? '✅' : '❌'}`);
    console.log(`   Summary: ${results.hasSummary ? '✅' : '❌'}`);

    // Take final screenshot
    await page.screenshot({
      path: path.join(__dirname, '../simple-results-screenshot.png'),
      fullPage: true
    });
    console.log('\n📸 Screenshot saved: simple-results-screenshot.png');

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 SIMPLE MODE TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`   Page Errors: ${pageErrors.length}`);
    console.log(`   Console Errors: ${consoleErrors.length}`);
    console.log(`   Training: ${trainingComplete ? '✅ Complete' : '❌ Failed'}`);
    console.log(`   Results: ${results.hasMAPE ? '✅ Displayed' : '❌ Missing'}`);
    console.log('='.repeat(60));

    const passed = pageErrors.length === 0 && trainingComplete && results.hasMAPE;
    console.log(passed ? '\n✅ SIMPLE MODE TEST PASSED\n' : '\n❌ SIMPLE MODE TEST FAILED\n');

    return passed;

  } catch (error) {
    console.error('\n❌ Test Error:', error.message);
    await page.screenshot({
      path: path.join(__dirname, '../simple-error-screenshot.png'),
      fullPage: true
    });

    if (pageErrors.length > 0) {
      console.log('\nAll page errors:');
      pageErrors.forEach((e, i) => console.log(`  ${i + 1}. ${e}`));
    }

    return false;
  } finally {
    await browser.close();
  }
}

runSimpleModeTest()
  .then(passed => process.exit(passed ? 0 : 1))
  .catch(err => {
    console.error('Fatal:', err);
    process.exit(1);
  });
