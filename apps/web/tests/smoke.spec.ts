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

  test('runs online inference with trace context and ETag reuse', async ({ page }) => {
    const featureResponseTrace = '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01';
    const predictionResponseTrace = '00-cccccccccccccccccccccccccccccccc-dddddddddddddddd-01';

    let featureCallCount = 0;
    let predictionCallCount = 0;

    await page.route('**/features', async (route) => {
      featureCallCount += 1;
      const headers = route.request().headers();
      expect(headers['traceparent']).toBeTruthy();

      if (featureCallCount === 1) {
        expect(headers['if-none-match']).toBeUndefined();
        await route.fulfill({
          status: 200,
          headers: {
            'content-type': 'application/json',
            etag: 'feature-etag',
            traceparent: featureResponseTrace,
          },
          body: JSON.stringify({
            symbol: 'BTC-USD',
            generated_at: '2024-05-01T00:30:00Z',
            features: {
              macd: 1.23,
              rsi: 55.5,
            },
          }),
        });
      } else {
        expect(headers['if-none-match']).toBe('feature-etag');
        expect(headers['traceparent']).toBe(predictionResponseTrace);
        await route.fulfill({
          status: 304,
          headers: {
            etag: 'feature-etag',
            traceparent: featureResponseTrace,
          },
        });
      }
    });

    await page.route('**/predictions', async (route) => {
      predictionCallCount += 1;
      const headers = route.request().headers();
      expect(headers['traceparent']).toBe(featureResponseTrace);

      if (predictionCallCount === 1) {
        expect(headers['if-none-match']).toBeUndefined();
        await route.fulfill({
          status: 200,
          headers: {
            'content-type': 'application/json',
            etag: 'prediction-etag',
            traceparent: predictionResponseTrace,
          },
          body: JSON.stringify({
            symbol: 'BTC-USD',
            generated_at: '2024-05-01T00:30:00Z',
            horizon_seconds: 900,
            score: 0.81234,
            signal: {
              action: 'buy',
              confidence: 0.78,
            },
          }),
        });
      } else {
        expect(headers['if-none-match']).toBe('prediction-etag');
        await route.fulfill({
          status: 200,
          headers: {
            'content-type': 'application/json',
            etag: 'prediction-etag',
            traceparent: predictionResponseTrace,
          },
          body: JSON.stringify({
            symbol: 'BTC-USD',
            generated_at: '2024-05-01T00:45:00Z',
            horizon_seconds: 900,
            score: 0.7421,
            signal: {
              action: 'hold',
              confidence: 0.52,
            },
          }),
        });
      }
    });

    await page.goto('/');

    const runButton = page.getByRole('button', { name: 'Compute features & signal' });
    await runButton.click();

    await expect(page.getByText('Feature vector')).toBeVisible();
    await expect(page.getByText('macd')).toBeVisible();
    await expect(page.getByText('1.2300')).toBeVisible();
    await expect(page.getByText('Prediction')).toBeVisible();
    await expect(page.getByText('0.8123')).toBeVisible();
    await expect(page.getByText('buy')).toBeVisible();
    await expect(page.getByText('feature-etag')).toBeVisible();
    await expect(page.getByText('prediction-etag')).toBeVisible();
    await expect(page.getByText(predictionResponseTrace)).toBeVisible();

    await runButton.click();

    await expect(page.getByText('hold')).toBeVisible();
    await expect(page.getByText('0.7421')).toBeVisible();
    expect(featureCallCount).toBeGreaterThanOrEqual(2);
    expect(predictionCallCount).toBeGreaterThanOrEqual(2);
  });
});
