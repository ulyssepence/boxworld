import { test, expect } from '@playwright/test'

test.describe('Smoke Tests', () => {
  test('page loads without JS errors', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (err) => errors.push(err.message))

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Filter out known non-critical errors (WebGL warnings, etc.)
    const criticalErrors = errors.filter((e) => !e.includes('WebGL') && !e.includes('onnxruntime'))
    expect(criticalErrors).toEqual([])
  })

  test('page title is Boxworld', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle('Boxworld')
  })

  test('canvas element is present', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const canvas = page.locator('canvas')
    await expect(canvas).toBeVisible()

    // Verify non-zero dimensions
    const box = await canvas.boundingBox()
    expect(box).toBeTruthy()
    expect(box!.width).toBeGreaterThan(0)
    expect(box!.height).toBeGreaterThan(0)
  })

  test('API /api/checkpoints returns 200', async ({ request }) => {
    const response = await request.get('/api/checkpoints')
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data).toHaveProperty('checkpoints')
    expect(Array.isArray(data.checkpoints)).toBeTruthy()
  })

  test('API /api/levels returns 200 with levels', async ({ request }) => {
    const response = await request.get('/api/levels')
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data).toHaveProperty('levels')
    expect(Array.isArray(data.levels)).toBeTruthy()
    expect(data.levels.length).toBeGreaterThan(0)
  })

  test('API /api/levels/nonexistent returns 404', async ({ request }) => {
    const response = await request.get('/api/levels/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('sidebar is visible with level selector', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Sidebar should be visible
    const sidebar = page.locator('.sidebar')
    await expect(sidebar).toBeVisible()

    // Level selector should have options (loaded from API)
    const select = sidebar.locator('select').first()
    await expect(select).toBeVisible()
  })

  test('level selector loads levels from API', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Wait for the level select to have options
    const select = page.locator('.sidebar select').first()
    // Should have at least one level option + the placeholder
    const options = select.locator('option')
    expect(await options.count()).toBeGreaterThan(1)
  })
})
