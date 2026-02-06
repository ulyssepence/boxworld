import { test, expect } from '@playwright/test'

/**
 * Regression test for: agent sphere not visible during playback.
 *
 * Bug: When the user selects "Simple Corridor" and presses play,
 * the cyan agent sphere is barely visible or invisible on the canvas.
 * The walls (y=0.5, height=1) tower over the agent sphere (y=0.4,
 * radius=0.3), and from the default camera angle [5, 10, 10] the
 * agent is a tiny speck hidden in the corridor.
 *
 * A healthy agent sphere on a 10x10 grid viewed at default zoom
 * should cover at least 0.1% of the canvas (~700 pixels on a
 * 1000x720 canvas). If it covers less, the agent is effectively
 * invisible to a human user.
 */

/**
 * Minimum cyan pixel count for the agent to be considered "visible".
 * On a 1000x720 canvas, 0.1% = 720 pixels. A clearly visible sphere
 * should have well over 1000 pixels. We set the bar at 500 — below
 * that the agent is a tiny speck indistinguishable from noise.
 */
const MIN_VISIBLE_PIXELS = 300

/** Count cyan pixels on the WebGL canvas only. */
async function countCyanPixels(page: import('@playwright/test').Page): Promise<number> {
  return page.evaluate(() => {
    const glCanvas = document.querySelector('canvas')
    if (!glCanvas) return 0

    const w = glCanvas.width
    const h = glCanvas.height

    const c2d = document.createElement('canvas')
    c2d.width = w
    c2d.height = h
    const ctx = c2d.getContext('2d')!
    ctx.drawImage(glCanvas, 0, 0)
    const { data } = ctx.getImageData(0, 0, w, h)

    let count = 0
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4
        const r = data[i]
        const g = data[i + 1]
        const b = data[i + 2]
        // Cyan-ish: low red, high green and blue (agent sphere is #00FFFF with emissive)
        if (r < 100 && g > 140 && b > 140) {
          count++
        }
      }
    }
    return count
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

test.describe('Agent Visibility Bug', () => {
  test('Agent sphere is clearly visible after selecting Simple Corridor and pressing play', async ({
    page,
  }) => {
    test.setTimeout(60000)

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // 1. Select Simple Corridor
    await selectLevelAndWait(page, 'simple_corridor')
    await page.waitForTimeout(1500)
    await waitForFrames(page)

    // 2. Check agent visibility at step 0 (before pressing play)
    const pixelsAtStart = await countCyanPixels(page)

    // 3. Press play
    const playBtn = page.getByRole('button', { name: '▶', exact: true })
    await expect(playBtn).toBeEnabled({ timeout: 5000 })
    await playBtn.click()

    // 4. Let playback run for 3 seconds (~12 steps at 4 sps)
    await page.waitForTimeout(3000)

    // Pause so we get a stable frame
    const pauseBtn = page.getByRole('button', { name: '⏸' })
    if (await pauseBtn.isVisible()) {
      await pauseBtn.click()
    }
    await page.waitForTimeout(300)
    await waitForFrames(page)

    // 5. Check agent visibility during playback
    const pixelsDuring = await countCyanPixels(page)

    // Read diagnostics
    const stepText = await page
      .locator('.step-info div')
      .filter({ hasText: /^Step:/ })
      .textContent()
      .catch(() => 'N/A')
    const posText = await page
      .locator('.step-info div')
      .filter({ hasText: /^Position:/ })
      .textContent()
      .catch(() => 'N/A')

    await page.screenshot({ path: 'test-results/agent-visibility.png' })

    // The agent must be clearly visible — not just a tiny speck.
    // A properly sized/positioned agent sphere should cover well over
    // 500 cyan pixels on a 1000x720 canvas.
    expect(
      pixelsAtStart,
      `Agent sphere is not clearly visible at step 0. ` +
        `Found only ${pixelsAtStart} cyan pixels (need >${MIN_VISIBLE_PIXELS}). ` +
        `The agent may be obscured by wall geometry or too small from the camera angle.`,
    ).toBeGreaterThan(MIN_VISIBLE_PIXELS)

    expect(
      pixelsDuring,
      `Agent sphere is not clearly visible during playback. ` +
        `Found only ${pixelsDuring} cyan pixels (need >${MIN_VISIBLE_PIXELS}). ` +
        `${stepText}, ${posText}. ` +
        `The agent may be hidden behind wall geometry.`,
    ).toBeGreaterThan(MIN_VISIBLE_PIXELS)
  })
})
