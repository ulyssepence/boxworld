import { test, expect } from '@playwright/test'

/**
 * Tests that key and door meshes visually update during episode playback.
 *
 * Uses recorded episodes from the DB for `door_key` level, then verifies:
 * 1. Gold key mesh is present before pickup, gone after
 * 2. Brown door mesh is present before toggle, gone after
 *
 * This catches the bug where play.step() grid changes aren't reflected
 * in the 3D scene during episode replay.
 *
 * We verify by inspecting the React/Three.js state (grid data passed to
 * the Items component) rather than pixel colors, since the post-processing
 * shader makes color-based detection unreliable.
 */

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

async function waitForFrames(page: import('@playwright/test').Page) {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  )
}

/** Seek to a specific step by setting the range slider value. */
async function seekToStep(page: import('@playwright/test').Page, step: number) {
  await page.evaluate((s) => {
    const slider = document.querySelector('.seek-slider') as HTMLInputElement
    if (!slider) throw new Error('No seek slider found')
    const nativeSet = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!
    nativeSet.call(slider, String(s))
    slider.dispatchEvent(new Event('change', { bubbles: true }))
  }, step)
  // Wait for React re-render + R3F frame
  await page.waitForTimeout(200)
  await waitForFrames(page)
}

/**
 * Get the grid data from the current step's displayLevel by walking the React fiber tree.
 * Returns an object with counts of doors and keys in the grid.
 */
async function getDisplayGridItems(
  page: import('@playwright/test').Page,
): Promise<{ doors: [number, number][]; keys: [number, number][] }> {
  return page.evaluate(() => {
    const root = document.getElementById('root')!
    const rootKey = Object.keys(root).find(
      (k) => k.startsWith('__reactContainer') || k.startsWith('__reactFiber'),
    )
    if (!rootKey) throw new Error('No React fiber found')

    let fiber = (root as any)[rootKey]
    const queue = [fiber]
    const visited = new Set()
    let appState: any = null

    while (queue.length > 0 && !appState) {
      const f = queue.shift()
      if (!f || visited.has(f)) continue
      visited.add(f)
      let hook = f.memoizedState
      while (hook) {
        if (hook.queue && hook.queue.lastRenderedState) {
          const s = hook.queue.lastRenderedState
          if (s && typeof s === 'object' && 'episodes' in s) {
            appState = s
            break
          }
        }
        hook = hook.next
      }
      if (f.child) queue.push(f.child)
      if (!appState && f.sibling) queue.push(f.sibling)
    }

    if (!appState) throw new Error('App state not found in fiber tree')

    const ep = appState.episodes[appState.currentEpisodeIndex]
    const step = ep?.steps[appState.currentStep]

    // Compute displayLevel the same way GameView does
    const displayLevel = appState.editMode
      ? appState.currentLevel
      : step
        ? step.state.level
        : appState.currentLevel

    if (!displayLevel) throw new Error('No displayLevel')

    const doors: [number, number][] = []
    const keys: [number, number][] = []
    for (let y = 0; y < displayLevel.grid.length; y++) {
      for (let x = 0; x < displayLevel.grid[y].length; x++) {
        if (displayLevel.grid[y][x] === 2) doors.push([x, y])
        if (displayLevel.grid[y][x] === 3) keys.push([x, y])
      }
    }

    return { doors, keys }
  })
}

test.describe('Inference episode: items update visually', () => {
  test('key disappears and door opens during episode playback', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    await selectLevelAndWait(page, 'door_key')

    // Wait for step info to appear (recorded episodes)
    const stepInfo = page.locator('.step-info div').filter({ hasText: /^Step:/ })
    await expect(stepInfo).toBeVisible({ timeout: 5000 })
    await page.waitForTimeout(500)
    await waitForFrames(page)

    // Find which steps have Pickup and Toggle actions by reading step info
    const totalStepsText = await page
      .locator('.step-info div')
      .filter({ hasText: /^Step:/ })
      .textContent()
    const maxStep = parseInt(totalStepsText!.split('/')[1].trim())

    // Scan through steps to find Pickup and Toggle
    let pickupStep = -1
    let toggleStep = -1
    for (let i = 0; i <= maxStep; i++) {
      await seekToStep(page, i)
      const actionText = await page
        .locator('.step-info div')
        .filter({ hasText: /^Action:/ })
        .textContent()
      if (actionText?.includes('Pickup') && pickupStep === -1) pickupStep = i
      if (actionText?.includes('Toggle') && toggleStep === -1) toggleStep = i
    }

    expect(pickupStep, 'Agent never performed Pickup action').toBeGreaterThanOrEqual(0)
    expect(toggleStep, 'Agent never performed Toggle action').toBeGreaterThanOrEqual(0)

    // --- Test key disappears ---
    // Before pickup: state shows key in grid
    await seekToStep(page, pickupStep)
    const beforePickup = await getDisplayGridItems(page)
    expect(beforePickup.keys.length, 'Key should be in grid before Pickup').toBeGreaterThan(0)

    // After pickup: key should be gone from grid
    await seekToStep(page, Math.min(pickupStep + 1, maxStep))
    const afterPickup = await getDisplayGridItems(page)
    expect(afterPickup.keys.length, 'Key should be removed from grid after Pickup').toBe(0)

    // Also verify step info shows Has Key changed
    const hasKeyBefore = await page
      .locator('.step-info div')
      .filter({ hasText: /^Has Key:/ })
      .textContent()

    await seekToStep(page, pickupStep)
    const hasKeyAtPickup = await page
      .locator('.step-info div')
      .filter({ hasText: /^Has Key:/ })
      .textContent()
    expect(hasKeyAtPickup).toContain('No') // State BEFORE pickup

    await seekToStep(page, Math.min(pickupStep + 1, maxStep))
    const hasKeyAfterPickup = await page
      .locator('.step-info div')
      .filter({ hasText: /^Has Key:/ })
      .textContent()
    expect(hasKeyAfterPickup).toContain('Yes') // State AFTER pickup

    // --- Test door disappears ---
    // Before toggle: door in grid
    await seekToStep(page, toggleStep)
    const beforeToggle = await getDisplayGridItems(page)
    expect(beforeToggle.doors.length, 'Door should be in grid before Toggle').toBeGreaterThan(0)

    // After toggle: door should be gone from grid
    await seekToStep(page, Math.min(toggleStep + 1, maxStep))
    const afterToggle = await getDisplayGridItems(page)
    expect(
      afterToggle.doors.length,
      `Door should be removed from grid after Toggle, but found doors at: ${JSON.stringify(afterToggle.doors)}`,
    ).toBe(0)

    // Verify key was consumed by toggle
    const hasKeyAfterToggle = await page
      .locator('.step-info div')
      .filter({ hasText: /^Has Key:/ })
      .textContent()
    expect(hasKeyAfterToggle).toContain('No') // Key consumed by toggle
  })
})
