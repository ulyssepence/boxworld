import { test, expect } from '@playwright/test'

/**
 * Bug reproduction tests for agent movement issues.
 *
 * Bug 1: "Run Agent" in browser produces a stationary agent.
 * Bug 2: Recorded episode playback shows a stationary agent.
 */

/** Helper: select a level by value and wait for it to fully load.
 *
 * React's synthetic events require us to set value via the native setter
 * and then dispatch a 'change' event with bubbles:true. This is the same
 * technique used by React Testing Library's fireEvent.change().
 */
async function selectLevelAndWait(page: import('@playwright/test').Page, levelId: string) {
  const sidebar = page.locator('.sidebar')
  const levelSelect = sidebar.locator('select').first()

  await levelSelect.evaluate((el: HTMLSelectElement, val: string) => {
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
      HTMLSelectElement.prototype,
      'value',
    )!.set!
    nativeInputValueSetter.call(el, val)
    el.dispatchEvent(new Event('change', { bubbles: true }))
  }, levelId)

  // Wait for the level to fully load by checking for Playback section
  await expect(sidebar.getByText('Playback')).toBeVisible({ timeout: 15000 })
}

/** Helper: collect agent positions by stepping through an episode.
 * The step-forward button renders as ▶▶ (U+25B6 x2) from &#9654;&#9654; */
async function collectPositions(
  page: import('@playwright/test').Page,
  maxSteps: number,
): Promise<string[]> {
  const positions: string[] = []

  for (let i = 0; i < maxSteps; i++) {
    // Read current position from Step Info section
    const posDiv = page.locator('.step-info div').filter({ hasText: /^Position:/ })
    if ((await posDiv.count()) === 0) break

    const text = await posDiv.textContent()
    if (text) positions.push(text)

    // Try to step forward — button text is ▶▶ (U+25B6 x2)
    const fwdBtn = page.getByRole('button', { name: '\u25B6\u25B6' })
    if (await fwdBtn.isDisabled()) break
    await fwdBtn.click()
    await page.waitForTimeout(100)
  }

  return positions
}

test.describe('Agent Movement Bug Reproduction', () => {
  test('Run Agent creates episode with Q-values', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    const runBtn = page.getByRole('button', { name: 'Run Agent' })
    await expect(runBtn).toBeVisible({ timeout: 10000 })

    await runBtn.click()

    // Wait for inference to complete — "Running..." disappears
    await expect(page.getByText('Running...')).toBeHidden({ timeout: 60000 })

    // Should now have Step Info visible
    const stepInfo = page.locator('.step-info').first()
    await expect(stepInfo).toBeVisible({ timeout: 5000 })

    // Q-Values section should appear
    const qSection = page.locator('.sidebar-section').filter({ hasText: 'Q-Values' })
    await expect(qSection).toBeVisible({ timeout: 5000 })

    // Check all 6 action names are present (use .first() to avoid strict mode on container)
    for (const action of ['Up:', 'Down:', 'Left:', 'Right:', 'Pickup:', 'Toggle:']) {
      await expect(qSection.getByText(action).first()).toBeVisible()
    }
  })

  test('Run Agent on simple_corridor shows agent movement', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    const runBtn = page.getByRole('button', { name: 'Run Agent' })
    await expect(runBtn).toBeVisible({ timeout: 10000 })

    // Select the highest-step checkpoint (last option in the inference select)
    const inferenceSection = page.locator('.sidebar-section').filter({ hasText: 'Agent Inference' })
    const checkpointSelect = inferenceSection.locator('select')
    const options = await checkpointSelect.locator('option').all()
    if (options.length > 0) {
      const lastValue = await options[options.length - 1].getAttribute('value')
      if (lastValue) await checkpointSelect.selectOption(lastValue)
    }

    // Run Agent
    await runBtn.click()
    await expect(page.getByText('Running...')).toBeHidden({ timeout: 60000 })

    // Collect positions by stepping through the episode
    const positions = await collectPositions(page, 20)

    // Bug 1: Agent should have moved — we expect more than 1 unique position
    const unique = new Set(positions)
    expect(
      unique.size,
      `Bug 1: Agent appears stationary on simple_corridor! ` +
        `Only positions seen: ${[...unique].join(', ')}`,
    ).toBeGreaterThan(1)
  })

  test('Recorded episode playback shows agent movement', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'simple_corridor')

    // Check if there are recorded episodes available (Episode section visible)
    const episodeSection = page.locator('.sidebar-section').filter({ hasText: /^Episode/ })
    await page.waitForTimeout(1000)
    const episodeVisible = await episodeSection.isVisible().catch(() => false)

    if (!episodeVisible) {
      test.skip(true, 'No recorded episodes available for simple_corridor')
      return
    }

    // Wait for step info to appear (first step loaded)
    const stepInfo = page.locator('.step-info div').filter({ hasText: /^Step:/ })
    await expect(stepInfo).toBeVisible({ timeout: 5000 })

    // Collect positions by stepping through the episode
    const positions = await collectPositions(page, 20)

    // Bug 2: Agent should have moved during the recorded episode
    const unique = new Set(positions)
    expect(
      unique.size,
      `Bug 2: Recorded episode shows stationary agent! ` +
        `Only positions seen: ${[...unique].join(', ')}`,
    ).toBeGreaterThan(1)
  })
})
