import { test, expect } from '@playwright/test'

test.describe('Visual Tests', () => {
  test('initial page load screenshot', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    // Wait a bit for WebGL to render
    await page.waitForTimeout(2000)

    await expect(page).toHaveScreenshot('initial-load.png', {
      maxDiffPixelRatio: 0.05,
    })
  })

  test('level loaded screenshot', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Select the first level
    const select = page.locator('.overlay-top .overlay-select').first()
    await select.selectOption({ index: 1 }) // first non-placeholder option

    // Wait for level to render
    await page.waitForTimeout(2000)

    await expect(page).toHaveScreenshot('level-loaded.png', {
      maxDiffPixelRatio: 0.05,
    })
  })
})
