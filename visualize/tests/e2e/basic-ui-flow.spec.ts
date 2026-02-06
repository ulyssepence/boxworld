import { test, expect } from '@playwright/test'

/**
 * Tests the basic user interaction flow:
 * 1. Select a level → agent visible at start
 * 2. Episode auto-loads → step info shows correct start position
 * 3. Scrub timeline → agent moves on screen
 * 4. Select different episode → resets to step 0
 * 5. Run Agent → populates timeline with agent moving
 */

/** Find cyan agent pixels on the WebGL canvas (preserveDrawingBuffer required). */
async function findAgentCentroid(
  page: import('@playwright/test').Page,
): Promise<{ x: number; y: number; count: number } | null> {
  return page.evaluate(() => {
    const glCanvas = document.querySelector('canvas')
    if (!glCanvas) return null

    const w = glCanvas.width
    const h = glCanvas.height

    const c2d = document.createElement('canvas')
    c2d.width = w
    c2d.height = h
    const ctx = c2d.getContext('2d')!
    ctx.drawImage(glCanvas, 0, 0)
    const { data } = ctx.getImageData(0, 0, w, h)

    let sumX = 0
    let sumY = 0
    let count = 0

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4
        const r = data[i]
        const g = data[i + 1]
        const b = data[i + 2]

        // Cyan-ish: low red, high green and blue
        if (r < 100 && g > 140 && b > 140) {
          sumX += px
          sumY += py
          count++
        }
      }
    }

    if (count < 3) return null
    return { x: Math.round(sumX / count), y: Math.round(sumY / count), count }
  })
}

/** Wait for two animation frames to flush R3F rendering. */
async function waitForFrames(page: import('@playwright/test').Page) {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  )
}

/** Select a level via React 19 onChange hack. */
async function selectLevelAndWait(page: import('@playwright/test').Page, levelId: string) {
  const sidebar = page.locator('.sidebar')
  const levelSelect = sidebar.locator('select').first()

  await levelSelect.evaluate((el: HTMLSelectElement, val: string) => {
    el.value = val
    const key = Object.keys(el).find((k) => k.startsWith('__reactProps'))!
    const props = (el as any)[key]
    props.onChange({ target: el, currentTarget: el })
  }, levelId)

  await page.waitForResponse((res) => res.url().includes(`/api/levels/${levelId}`))
  await expect(sidebar.getByText('Playback')).toBeVisible({ timeout: 15000 })
}

test.describe('Basic UI Flow', () => {
  test('Select level → agent visible at agentStart (1,1)', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    // Wait for R3F to render
    await page.waitForTimeout(1000)
    await waitForFrames(page)

    // Agent should be visible on canvas
    const pos = await findAgentCentroid(page)
    expect(pos, 'Agent should be visible on canvas after selecting level').not.toBeNull()
    expect(pos!.count).toBeGreaterThan(10)

    // Step info should show the correct start position
    const posText = await page
      .locator('.step-info div')
      .filter({ hasText: /^Position:/ })
      .textContent()
    expect(posText).toContain('(1, 1)')
  })

  test('Scrub timeline → agent position changes in sidebar', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')
    await page.waitForTimeout(500)

    // Read initial position
    const posTextBefore = await page
      .locator('.step-info div')
      .filter({ hasText: /^Position:/ })
      .textContent()

    // Step forward several times
    const fwdBtn = page.getByRole('button', { name: '▶▶' })
    for (let i = 0; i < 5; i++) {
      if (await fwdBtn.isDisabled()) break
      await fwdBtn.click()
      await page.waitForTimeout(200)
    }

    // Read new position
    const posTextAfter = await page
      .locator('.step-info div')
      .filter({ hasText: /^Position:/ })
      .textContent()

    expect(
      posTextAfter,
      `Position should change after stepping. Before: "${posTextBefore}", After: "${posTextAfter}"`,
    ).not.toBe(posTextBefore)
  })

  test('Scrub timeline → agent moves visually on canvas', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')
    await page.waitForTimeout(1000)
    await waitForFrames(page)

    const pos0 = await findAgentCentroid(page)
    expect(pos0, 'Agent should be visible at step 0').not.toBeNull()

    // Step forward
    const fwdBtn = page.getByRole('button', { name: '▶▶' })
    for (let i = 0; i < 5; i++) {
      if (await fwdBtn.isDisabled()) break
      await fwdBtn.click()
      await page.waitForTimeout(400)
    }
    await waitForFrames(page)

    const pos1 = await findAgentCentroid(page)
    expect(pos1, 'Agent should be visible after stepping').not.toBeNull()

    const dx = pos1!.x - pos0!.x
    const dy = pos1!.y - pos0!.y
    const distance = Math.sqrt(dx * dx + dy * dy)

    expect(
      distance,
      `Agent should move on canvas. Before: (${pos0!.x},${pos0!.y}), After: (${pos1!.x},${pos1!.y}), distance: ${distance.toFixed(1)}`,
    ).toBeGreaterThan(5)
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

    // Check if there are multiple episodes
    const episodeSelect = page.locator('.sidebar select').nth(1)
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
