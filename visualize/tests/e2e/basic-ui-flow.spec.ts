import { test, expect } from '@playwright/test'

/**
 * Tests the basic user interaction flow:
 * 1. Scrub timeline → step info position changes
 * 2. Select different episode → resets to step 0
 * 3. Episodes start at agentStart
 */

/** Select a level via React 19 onChange hack. */
async function selectLevelAndWait(page: import('@playwright/test').Page, levelId: string) {
  const overlay = page.locator('.overlay-top')
  const levelSelect = overlay.locator('.overlay-select').first()

  await levelSelect.evaluate((el: HTMLSelectElement, val: string) => {
    el.value = val
    const key = Object.keys(el).find((k) => k.startsWith('__reactProps'))!
    const props = (el as any)[key]
    props.onChange({ target: el, currentTarget: el })
  }, levelId)

  await page.waitForResponse((res) => res.url().includes(`/api/levels/${levelId}`))
  await expect(page.getByText('Recordings')).toBeVisible({ timeout: 15000 })
}

test.describe('Basic UI Flow', () => {
  test('Scrub timeline → agent position changes in step info', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    // Wait for step info to appear in the bottom-right overlay
    const posLocator = page
      .locator('.overlay-bottom-right .step-info div')
      .filter({ hasText: /^Position:/ })
    await expect(posLocator).toBeVisible({ timeout: 5000 })

    // Read initial position
    const posTextBefore = await posLocator.textContent()

    // Step forward enough times to move past the initial position
    const fwdBtn = page.locator('.overlay-top button', { hasText: '▶▶' })
    await expect(fwdBtn).toBeEnabled({ timeout: 2000 })
    for (let i = 0; i < 15; i++) {
      if (await fwdBtn.isDisabled()) break
      await fwdBtn.click({ force: true, timeout: 2000 })
      await page.waitForTimeout(100)
    }

    // Read new position
    const posTextAfter = await posLocator.textContent()

    expect(
      posTextAfter,
      `Position should change after stepping. Before: "${posTextBefore}", After: "${posTextAfter}"`,
    ).not.toBe(posTextBefore)
  })

  test('Select different episode → resets to step 0', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    // Step forward a few
    const fwdBtn = page.getByRole('button', { name: '▶▶' })
    for (let i = 0; i < 3; i++) {
      if (await fwdBtn.isDisabled()) break
      await fwdBtn.click()
      await page.waitForTimeout(200)
    }

    const stepBefore = await page
      .locator('.step-info div')
      .filter({ hasText: /^Step:/ })
      .textContent()
    expect(stepBefore).not.toContain('Step: 0')

    // Check if there are multiple episodes — episode select is the second .overlay-select in overlay
    const episodeSelect = page.locator('.overlay-top .overlay-select').nth(1)
    const optionCount = await episodeSelect.locator('option').count()

    if (optionCount < 2) {
      test.skip(true, 'Only one episode available, cannot test episode switching')
      return
    }

    // Select episode 2 (index 1)
    await episodeSelect.selectOption({ index: 1 })
    await page.waitForTimeout(300)

    const stepAfter = await page
      .locator('.step-info div')
      .filter({ hasText: /^Step:/ })
      .textContent()
    expect(stepAfter).toContain('Step: 0')
  })

  test('Episodes start at agentStart, not random positions', async ({ page }) => {
    // Verify that all recorded episodes start at the level's agentStart
    const response = await page.goto('/')
    await page.waitForLoadState('networkidle')

    const data = await page.evaluate(() =>
      fetch('/api/levels/simple_corridor').then((r) => r.json()),
    )

    const agentStart = data.level.agentStart
    const grid = data.level.grid

    for (let i = 0; i < data.episodes.length; i++) {
      const ep = data.episodes[i]
      const startPos = ep.steps[0]?.state?.agentPosition
      expect(
        startPos,
        `Episode ${i + 1} should start at agentStart (${agentStart}), got (${startPos})`,
      ).toEqual(agentStart)

      // Verify agent is not in a wall
      const cellType = grid[startPos[1]][startPos[0]]
      expect(cellType, `Episode ${i + 1} starts in a wall cell`).not.toBe(1)
    }
  })
})
