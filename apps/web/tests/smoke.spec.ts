import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test.describe('Scenario Studio smoke', () => {
  test('renders the primary dashboard layout', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/TradePulse Scenario Studio/i);
    await expect(page.getByRole('heading', { name: 'Scenario Studio' })).toBeVisible();
    await expect(page.getByLabel('Scenario template')).toBeVisible();

    await page.selectOption('#template', 'mean-reversion');
    await expect(page.getByText('Mean Reversion Swing')).toBeVisible();
  });

  test('has no critical accessibility regressions', async ({ page }) => {
    await page.goto('/');
    const scanResults = await new AxeBuilder({ page }).analyze();
    const severeViolations = scanResults.violations.filter((violation) =>
      ['critical', 'serious'].includes(violation.impact ?? '')
    );

    expect(severeViolations).toEqual([]);
  });
});
