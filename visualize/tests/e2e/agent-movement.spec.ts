import { test, expect } from '@playwright/test'

/**
 * Tests that the 3D agent mesh VISUALLY moves on the WebGL canvas.
 *
 * These tests find the cyan agent sphere's pixel centroid at different
 * episode steps and verify it moves to a different screen position.
 * This proves the R3F mesh actually updates in the rendered frame,
 * not just that sidebar text changes.
 *
 * Requires `preserveDrawingBuffer: true` on the R3F Canvas so that
 * drawImage can read the WebGL framebuffer for pixel analysis.
 */

/**
 * Select a level via React's internal onChange (React 19 compatible)
 * and wait for the API response + render.
 */
async function selectLevelAndWait(page: import('@playwright/test').Page, levelId: string) {
  const sidebar = page.locator('.sidebar')
  const levelSelect = sidebar.locator('select').first()

  // Trigger React 19 onChange via the fiber props
  await levelSelect.evaluate((el: HTMLSelectElement, val: string) => {
    el.value = val
    const key = Object.keys(el).find((k) => k.startsWith('__reactProps'))!
    const props = (el as any)[key]
    props.onChange({ target: el, currentTarget: el })
  }, levelId)

  // Wait for the API response (level data + episodes)
  await page.waitForResponse((res) => res.url().includes(`/api/levels/${levelId}`))

  // Wait for React to re-render with the loaded data
  await expect(sidebar.getByText('Playback')).toBeVisible({ timeout: 15000 })
}

/**
 * Find the centroid of cyan-ish pixels by copying the WebGL canvas
 * (with preserveDrawingBuffer) to a 2D canvas and scanning pixels.
 *
 * The agent sphere is #00FFFF with emissive. After 3D lighting, pixels
 * will have low R, high G, high B.
 */
async function findAgentCentroid(
  page: import('@playwright/test').Page,
): Promise<{ x: number; y: number; count: number } | null> {
  return page.evaluate(() => {
    const glCanvas = document.querySelector('canvas')
    if (!glCanvas) return null

    const w = glCanvas.width
    const h = glCanvas.height

    // Copy WebGL canvas to 2D canvas — drawImage composites the current frame
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

/** Click the step-forward button n times with waits for animation. */
async function stepForward(page: import('@playwright/test').Page, n: number) {
  const fwdBtn = page.getByRole('button', { name: '\u25B6\u25B6' })
  for (let i = 0; i < n; i++) {
    if (await fwdBtn.isDisabled()) break
    await fwdBtn.click()
    // Wait for lerp animation (delta*8 = ~125ms) + render
    await page.waitForTimeout(400)
  }
}

/** Wait for two full animation frames to be painted. */
async function waitForFrames(page: import('@playwright/test').Page) {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  )
}

test.describe('Agent Mesh Visual Movement', () => {
  test('Run Agent: cyan sphere moves on canvas between steps', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    // Run live ONNX inference
    const runBtn = page.getByRole('button', { name: 'Run Agent' })
    await expect(runBtn).toBeVisible({ timeout: 10000 })
    await runBtn.click()
    await expect(page.getByText('Running...')).toBeHidden({ timeout: 60000 })

    // Wait for render to settle
    await page.waitForTimeout(1000)
    await waitForFrames(page)

    // Find agent centroid at step 0
    const pos0 = await findAgentCentroid(page)
    expect(pos0, 'Could not find cyan agent sphere on canvas at step 0').not.toBeNull()

    // Check if step-forward button is enabled
    const fwdBtn = page.getByRole('button', { name: '\u25B6\u25B6' })
    const btnDisabled = await fwdBtn.isDisabled()
    const stepText = await page
      .locator('.step-info div')
      .filter({ hasText: /^Step:/ })
      .textContent()

    // Read sidebar position before stepping
    const posTextBefore = await page
      .locator('.step-info div')
      .filter({ hasText: /^Position:/ })
      .textContent()

    // Step forward 5 times (agent moves RIGHT along corridor)
    await stepForward(page, 5)
    await waitForFrames(page)

    // Read sidebar position after stepping
    const posTextAfter = await page
      .locator('.step-info div')
      .filter({ hasText: /^Position:/ })
      .textContent()

    // Find agent centroid after stepping
    const pos1 = await findAgentCentroid(page)
    expect(pos1, 'Could not find cyan agent sphere on canvas after stepping').not.toBeNull()

    const dx = pos1!.x - pos0!.x
    const dy = pos1!.y - pos0!.y
    const distance = Math.sqrt(dx * dx + dy * dy)

    expect(
      distance,
      `Agent mesh did NOT move on canvas! ` +
        `Centroid at step 0: (${pos0!.x}, ${pos0!.y}) [${pos0!.count}px], ` +
        `after 5 steps: (${pos1!.x}, ${pos1!.y}) [${pos1!.count}px], ` +
        `pixel distance: ${distance.toFixed(1)}. ` +
        `Sidebar position before: "${posTextBefore}", after: "${posTextAfter}". ` +
        `Step text: "${stepText}", fwd button disabled: ${btnDisabled}`,
    ).toBeGreaterThan(5)
  })

  test('Recorded episode: cyan sphere moves on canvas during playback', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    // Check if recorded episodes are available
    const episodeSection = page.locator('.sidebar-section').filter({ hasText: /^Episode/ })
    await page.waitForTimeout(1000)
    const episodeVisible = await episodeSection.isVisible().catch(() => false)

    if (!episodeVisible) {
      test.skip(true, 'No recorded episodes available for simple_corridor')
      return
    }

    // Wait for step info to appear + render
    const stepInfo = page.locator('.step-info div').filter({ hasText: /^Step:/ })
    await expect(stepInfo).toBeVisible({ timeout: 5000 })
    await page.waitForTimeout(1000)
    await waitForFrames(page)

    // Find agent centroid at step 0
    const pos0 = await findAgentCentroid(page)
    expect(pos0, 'Could not find cyan agent sphere on canvas at step 0').not.toBeNull()

    // Step forward 5 times
    await stepForward(page, 5)
    await waitForFrames(page)

    // Find agent centroid after stepping
    const pos1 = await findAgentCentroid(page)
    expect(pos1, 'Could not find cyan agent sphere on canvas after stepping').not.toBeNull()

    const dx = pos1!.x - pos0!.x
    const dy = pos1!.y - pos0!.y
    const distance = Math.sqrt(dx * dx + dy * dy)

    expect(
      distance,
      `Agent mesh did NOT move during recorded playback! ` +
        `Centroid at step 0: (${pos0!.x}, ${pos0!.y}) [${pos0!.count}px], ` +
        `after 5 steps: (${pos1!.x}, ${pos1!.y}) [${pos1!.count}px], ` +
        `pixel distance: ${distance.toFixed(1)}`,
    ).toBeGreaterThan(5)
  })
})
