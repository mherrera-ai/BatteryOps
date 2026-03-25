#!/usr/bin/env node

import { mkdir } from 'node:fs/promises';
import process from 'node:process';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const outputDir = process.env.BATTERYOPS_SCREENSHOT_DIR ?? scriptDir;
const appUrl = process.env.BATTERYOPS_APP_URL ?? 'http://127.0.0.1:8501';
const startupCommand = process.env.BATTERYOPS_APP_COMMAND;
const readyTimeoutMs = Number(process.env.BATTERYOPS_SCREENSHOT_READY_TIMEOUT_MS ?? 20_000);
const settleDelayMs = Number(process.env.BATTERYOPS_SCREENSHOT_SETTLE_MS ?? 800);
const appReadyTimeoutMs = Number(process.env.BATTERYOPS_APP_READY_TIMEOUT_MS ?? 90_000);
const appWaitPollMs = Number(process.env.BATTERYOPS_APP_WAIT_POLL_MS ?? 1_000);
const mainContainerSelector = '[data-testid="stMainBlockContainer"]';

const shots = [
  {
    name: 'overview.png',
    tab: 'Overview',
    ready: 'System Snapshot',
    focusAsset: 'battery36',
  },
  {
    name: 'live-telemetry-replay.png',
    tab: 'Live Telemetry Replay',
    ready: 'Replay cycle cursor',
    focusAsset: 'battery36',
    afterTab: 'jump-latest',
  },
  {
    name: 'anomaly-timeline.png',
    tab: 'Anomaly Timeline',
    ready: 'Flagged cycle queue',
    focusAsset: 'battery36',
  },
  {
    name: 'incident-report.png',
    tab: 'Incident Report',
    ready: 'Evidence',
    focusAsset: 'battery50',
  },
  {
    name: 'similar-cases.png',
    tab: 'Similar Cases',
    ready: 'Retrieved case table',
    focusAsset: 'battery50',
  },
  {
    name: 'evaluation-dashboard.png',
    tab: 'Evaluation Dashboard',
    ready: 'Proxy RUL MAE',
    focusAsset: 'battery50',
  },
];

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function startAppProcess() {
  if (!startupCommand) {
    return null;
  }

  const child = spawn(startupCommand, {
    shell: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
  });
  child.stdout.on('data', (chunk) => {
    process.stdout.write(`\x1b[90m[app]\x1b[0m ${chunk}`);
  });
  child.stderr.on('data', (chunk) => {
    process.stderr.write(`\x1b[90m[app]\x1b[0m ${chunk}`);
  });
  return child;
}

async function waitForAppStartup(url) {
  const deadline = Date.now() + appReadyTimeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { method: 'GET', redirect: 'manual' });
      if (response.ok || response.status === 200 || response.status === 302) {
        return;
      }
    } catch (error) {
      // Keep retrying until timeout while the server starts.
    }
    await sleep(appWaitPollMs);
  }
  throw new Error(`Timed out waiting for ${url} to become available`);
}

async function waitForRenderedApp(page) {
  await page.getByText('BatteryOps', { exact: true }).first().waitFor({
    state: 'visible',
    timeout: readyTimeoutMs,
  });
  await page.evaluate(async () => {
    if (document.fonts?.ready) {
      await document.fonts.ready;
    }
  });
  await page.waitForLoadState('networkidle');
  await page.locator(mainContainerSelector).first().waitFor({
    state: 'visible',
    timeout: readyTimeoutMs,
  });
}

async function stopAppProcess(child) {
  if (child === null || child.exitCode !== null) {
    return;
  }

  if (process.platform !== 'win32') {
    try {
      process.kill(-child.pid, 'SIGTERM');
    } catch (error) {
      if (child.exitCode !== null || error?.code === 'ESRCH') {
        return;
      }
      throw error;
    }
  } else {
    child.kill('SIGTERM');
  }
  await sleep(500);
  if (child.exitCode === null) {
    if (process.platform !== 'win32') {
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch (error) {
        if (error?.code !== 'ESRCH') {
          throw error;
        }
      }
    } else {
      child.kill('SIGKILL');
    }
  }
}

async function resolveFocusAssetSelector(page) {
  const labeled = page.getByRole('combobox', { name: /Focus asset/i });
  if ((await labeled.count()) > 0) {
    return labeled.first();
  }

  const fallback = page.locator('[role="combobox"]');
  if ((await fallback.count()) > 0) {
    return fallback.first();
  }

  return null;
}

async function selectFocusAsset(page, assetId) {
  const combo = await resolveFocusAssetSelector(page);
  if (!combo) {
    return;
  }

  const current = (await combo.textContent())?.trim();
  if (current === assetId) {
    return;
  }

  await combo.click();
  const option = page.getByRole('option', { name: assetId, exact: true });
  await option.first().waitFor({ state: 'visible', timeout: readyTimeoutMs });
  await option.first().click();
  await page.waitForLoadState('networkidle');
  await sleep(settleDelayMs);
}

async function resolveMainContainer(page) {
  const candidates = [
    mainContainerSelector,
    '.block-container',
    '[data-testid="stApp"]',
    'main',
  ];

  for (const selector of candidates) {
    const locator = page.locator(selector).first();
    if ((await locator.count()) > 0) {
      return locator;
    }
  }
  return page.locator('body');
}

async function captureShot(page, shot) {
  await page.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await waitForRenderedApp(page);

  if (shot.focusAsset) {
    await selectFocusAsset(page, shot.focusAsset);
  }

  await page.getByRole('tab', { name: shot.tab, exact: true }).click();
  await page.getByRole('tab', { name: shot.tab, exact: true }).waitFor({
    state: 'visible',
    timeout: readyTimeoutMs,
  });
  await page.waitForLoadState('networkidle');

  if (shot.ready) {
    const readyText = page.getByText(shot.ready, { exact: true }).first();
    try {
      await readyText.waitFor({ state: 'visible', timeout: readyTimeoutMs });
    } catch (error) {
      await page
        .getByRole('tab', { name: shot.tab })
        .first()
        .waitFor({ state: 'visible', timeout: readyTimeoutMs });
    }
  }

  if (shot.afterTab === 'jump-latest') {
    await page.getByRole('button', { name: 'Jump to latest' }).click();
    await sleep(settleDelayMs);
    await page.waitForLoadState('networkidle');
  }

  await sleep(settleDelayMs);
  const container = await resolveMainContainer(page);
  await container.waitFor({ state: 'visible', timeout: readyTimeoutMs });

  const target = path.join(outputDir, shot.name);
  console.log(`capturing ${shot.name}`);
  await container.screenshot({
    path: target,
    animations: 'disabled',
    caret: 'hide',
  });
  console.log(`saved ${target}`);
}

async function main() {
  await mkdir(outputDir, { recursive: true });

  const appProcess = startAppProcess();
  try {
    if (startupCommand) {
      await waitForAppStartup(appUrl);
    }

    const browser = await chromium.launch({ headless: true });
    try {
      const context = await browser.newContext({
        viewport: { width: 1800, height: 1800 },
        deviceScaleFactor: 1.5,
        locale: 'en-US',
        timezoneId: 'America/Los_Angeles',
        colorScheme: 'light',
      });
      const page = await context.newPage();
      await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' });

      for (const shot of shots) {
        await captureShot(page, shot);
      }

      await context.close();
    } finally {
      await browser.close();
    }
  } finally {
    await stopAppProcess(appProcess);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
