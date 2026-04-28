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
const screenshotMinHeightPx = Number(process.env.BATTERYOPS_SCREENSHOT_MIN_HEIGHT_PX ?? 780);
const screenshotMaxHeightPx = Number(process.env.BATTERYOPS_SCREENSHOT_MAX_HEIGHT_PX ?? 1_350);
const screenshotContentPaddingPx = Number(
  process.env.BATTERYOPS_SCREENSHOT_CONTENT_PADDING_PX ?? 32,
);

const shots = [
  {
    name: 'fleet-cockpit.png',
    tab: 'Fleet Cockpit',
    ready: 'Fleet Cockpit',
    focusAsset: 'battery36',
  },
  {
    name: 'asset-replay.png',
    tab: 'Asset Replay',
    ready: 'Replay cycle cursor',
    focusAsset: 'battery36',
    afterTab: 'jump-latest',
  },
  {
    name: 'incident-evidence.png',
    tab: 'Incident Evidence',
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
    name: 'model-evaluation.png',
    tab: 'Model Evaluation',
    ready: 'Proxy RUL MAE',
    focusAsset: 'battery50',
  },
  {
    name: 'data-provenance.png',
    tab: 'Data & Provenance',
    ready: 'Data quality checks',
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

async function warmBelowTheFoldContent(page) {
  await page.evaluate(async () => {
    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const maxScroll = Math.max(
      document.body.scrollHeight,
      document.documentElement.scrollHeight,
    );
    for (let y = 0; y <= maxScroll; y += 650) {
      window.scrollTo(0, y);
      await delay(120);
    }
    window.scrollTo(0, 0);
  });
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

async function resolveScreenshotClip(page, container) {
  const box = await container.boundingBox();
  if (!box) {
    throw new Error('Unable to resolve screenshot container bounds');
  }

  const viewport = page.viewportSize() ?? { width: 1800, height: 1500 };
  const measuredContentHeight = await page.evaluate(
    ({ mainSelector, padding }) => {
      const root =
        document.querySelector(mainSelector) ??
        document.querySelector('.block-container') ??
        document.querySelector('[data-testid="stApp"]') ??
        document.body;
      const rootRect = root.getBoundingClientRect();
      const renderTags = new Set([
        'button',
        'canvas',
        'h1',
        'h2',
        'h3',
        'h4',
        'img',
        'input',
        'li',
        'p',
        'select',
        'svg',
        'table',
        'td',
        'textarea',
        'th',
      ]);
      const renderRoles = new Set(['button', 'img', 'progressbar', 'tab', 'table']);
      let bottom = rootRect.top;

      for (const element of root.querySelectorAll('*')) {
        const style = window.getComputedStyle(element);
        if (
          style.display === 'none' ||
          style.visibility === 'hidden' ||
          Number(style.opacity) === 0
        ) {
          continue;
        }

        const rects = Array.from(element.getClientRects()).filter(
          (rect) => rect.width > 2 && rect.height > 2,
        );
        if (rects.length === 0) {
          continue;
        }

        const tag = element.tagName.toLowerCase();
        const role = element.getAttribute('role') ?? '';
        const hasDirectText = Array.from(element.childNodes).some(
          (node) => node.nodeType === Node.TEXT_NODE && node.textContent?.trim(),
        );
        const hasVisibleBackground =
          !['rgba(0, 0, 0, 0)', 'transparent'].includes(style.backgroundColor) &&
          rects.some((rect) => rect.height < window.innerHeight * 0.9);
        const hasBorder =
          Number.parseFloat(style.borderBottomWidth) > 0 ||
          Number.parseFloat(style.borderLeftWidth) > 0 ||
          Number.parseFloat(style.borderRightWidth) > 0 ||
          Number.parseFloat(style.borderTopWidth) > 0;
        const looksRenderable =
          hasDirectText ||
          renderTags.has(tag) ||
          renderRoles.has(role) ||
          hasVisibleBackground ||
          hasBorder;

        if (!looksRenderable) {
          continue;
        }

        for (const rect of rects) {
          bottom = Math.max(bottom, rect.bottom);
        }
      }

      if (bottom <= rootRect.top) {
        return rootRect.height;
      }
      return Math.ceil(bottom - rootRect.top + padding);
    },
    { mainSelector: mainContainerSelector, padding: screenshotContentPaddingPx },
  );

  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const width = Math.ceil(Math.min(box.width, viewport.width - x));
  const height = Math.ceil(
    Math.min(
      box.height,
      Math.max(screenshotMinHeightPx, Number(measuredContentHeight) || screenshotMinHeightPx),
      screenshotMaxHeightPx,
      Math.max(400, viewport.height - y),
    ),
  );

  return { x, y, width, height };
}

async function captureShot(page, shot) {
  const targetUrl = new URL(appUrl);
  if (shot.focusAsset) {
    targetUrl.searchParams.set('focus_asset', shot.focusAsset);
  }
  await page.goto(targetUrl.toString(), { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await waitForRenderedApp(page);

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
    const jumpLatest = page.getByRole('button', { name: 'Jump to latest' });
    if (await jumpLatest.isEnabled()) {
      await jumpLatest.click();
      await sleep(settleDelayMs);
      await page.waitForLoadState('networkidle');
    }
  }

  await warmBelowTheFoldContent(page);
  await sleep(settleDelayMs);
  const container = await resolveMainContainer(page);
  await container.waitFor({ state: 'visible', timeout: readyTimeoutMs });

  const target = path.join(outputDir, shot.name);
  console.log(`capturing ${shot.name}`);
  const clip = await resolveScreenshotClip(page, container);
  await page.screenshot({
    path: target,
    clip,
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
        viewport: { width: 1800, height: 1500 },
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
