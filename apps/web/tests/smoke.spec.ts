import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const USERNAME = process.env.SCENARIO_STUDIO_USERNAME ?? 'demo';
const PASSWORD = process.env.SCENARIO_STUDIO_PASSWORD ?? 'demo-pass';

async function ensureAuthenticated(page: Page) {
  await page.goto('/login');
  const current = new URL(page.url());

  if (current.pathname !== '/login') {
    await page.goto('/');
    return;
  }

  await page.fill('input[name="username"]', USERNAME);
  await page.fill('input[name="password"]', PASSWORD);

  await Promise.all([
    page.waitForNavigation({ url: (url) => new URL(url).pathname === '/' }),
    page.getByRole('button', { name: 'Sign in' }).click(),
  ]);
  await page.waitForURL((url) => new URL(url).pathname === '/');
}

test.describe('Scenario Studio smoke', () => {
  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page);
  });

  test('renders the primary dashboard layout', async ({ page }) => {
    await expect(page).toHaveTitle(/TradePulse Scenario Studio/i);
    await expect(page.getByRole('heading', { name: 'Scenario Studio' })).toBeVisible();
    await expect(page.getByLabel('Scenario template')).toBeVisible();

    await page.selectOption('#template', 'mean-reversion');
    await expect(page.getByText('Mean Reversion Swing')).toBeVisible();
  });

  test('exports scenario JSON when validation passes and disables actions when invalid', async ({ page }) => {
    await page.addInitScript(() => {
      const writeText = async (value: string) => {
        ;(window as typeof window & { __lastCopied?: string }).__lastCopied = value;
      };
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText },
        configurable: true,
      });
    });

    const previewLocator = page.locator('pre');
    const copyButton = page.getByRole('button', { name: 'Copy to clipboard' });
    const downloadButton = page.getByRole('button', { name: 'Download JSON' });

    await expect(copyButton).toBeEnabled();
    await expect(downloadButton).toBeEnabled();

    const preview = (await previewLocator.textContent()) ?? '';
    await copyButton.click();
    await expect(page.getByRole('status', { name: /Scenario JSON copied/ })).toBeVisible();

    const copied = await page.evaluate(() => (window as typeof window & { __lastCopied?: string }).__lastCopied);
    expect(copied).toBe(preview);

    const [download] = await Promise.all([page.waitForEvent('download'), downloadButton.click()]);
    await expect(download.suggestedFilename()).toMatch(/scenario-.*\.json/);

    await page.fill('input[name="initialBalance"]', '-1');
    await expect(copyButton).toBeDisabled();
    await expect(downloadButton).toBeDisabled();
  });

  test('has no critical accessibility regressions', async ({ page }) => {
    const scanResults = await new AxeBuilder({ page }).analyze();
    const severeViolations = scanResults.violations.filter((violation) =>
      ['critical', 'serious'].includes(violation.impact ?? '')
    );

    expect(severeViolations).toEqual([]);
  });
});
