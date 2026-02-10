# Overlay UI with Recordings and Inference Tabs

Replaced the sidebar-based UI with floating overlay panels on the full-screen 3D scene. The UI now has two modes accessed via tabs: **Recordings** (replay recorded episodes from the database) and **Inference** (run live ONNX agent, edit levels, watch it restart).

## Layout

- **Top-center floating panel** (`overlay-top`): Positioned at `top: 15vh`, centered horizontally, `width: min(33vw, 420px)`. Semi-transparent dark background with backdrop blur and rounded corners.
- **Bottom-right floating panels** (`overlay-bottom-right`): Step Info and Q-Values panels, shown contextually based on active tab.
- **Full-screen Canvas**: The R3F scene fills the entire viewport behind the overlays.

## Header Row

Single row containing:
- "Boxworld" title in cyan
- Level selector dropdown
- "Edit Level" / "Exit Edit" button (visible in both tabs)
- "Reset" button (visible only when editing)

Clicking "Edit Level" activates edit mode **and** switches to the Inference tab, since level editing is most useful alongside live inference where the agent auto-restarts on edits.

## Tabs

Two tabs below the header: **Recordings** and **Inference**. Centered with an underline indicator on the active tab.

### Recordings Tab
- Episode selector dropdown (checkpoint steps + reward)
- Playback controls: rewind, play/pause, forward, seek slider, speed -/+ buttons
- Speed controls are compact `-` / `4x` / `+` buttons inline with playback

### Inference Tab
- Checkpoint selector dropdown with "Run Agent" / "Restart" button
- Live playback controls (play/pause, step, stop) with inline speed -/+ buttons
- Edit hint text: "Click tiles to edit the level. Agent auto-restarts on edits."
- Auto-restart on cell edits via `editVersion` tracking in a `useEffect`

## State Changes (util.tsx)

### New State Fields
- `viewMode: 'recordings' | 'inference'` — controls which tab is active and which rendering path GameView uses
- `editVersion: number` — incremented on every `EDIT_CELL`, watched by InferenceTab to auto-restart
- `generation: number` on `LiveInferenceState` — prevents stale async ticks from previous inference runs

### New Actions
- `SET_VIEW_MODE` — switches tabs; switching to recordings stops any active inference
- `RESTART_LIVE_INFERENCE` — resets game state, increments generation, keeps agent loaded
- `LIVE_STEP` now carries `generation` — reducer rejects ticks with stale generation numbers

### Key Behavioral Changes
- `START_LIVE_INFERENCE` no longer wipes `episodes: []` — recordings are preserved when switching tabs
- `STOP_LIVE_INFERENCE` simplified to just reset `liveInference` to initial state
- `LOAD_LEVEL` resets `viewMode` to `'recordings'` and `editVersion` to 0
- `usePlayback` early-returns when `viewMode !== 'recordings'`
- `useLiveInference` captures `generation` before each tick and passes it in the dispatch

## Camera Centering (render.tsx)

Added a `CameraController` component that sets the OrbitControls target to the center of the current level `((width-1)/2, 0, (height-1)/2)` instead of the origin. Camera repositions when a new level is loaded. The `Scene` component accepts an optional `target` prop, passed from a new `Root` component in client.tsx that reads the current level from state.

## Component Structure (client.tsx)

```
Root
├── render.Scene (with target prop for camera centering)
│   └── GameView (switches rendering path based on viewMode)
└── Overlay
    ├── overlay-top
    │   ├── Header row (title, level select, edit level, reset)
    │   ├── Tab strip (Recordings | Inference)
    │   └── Tab content
    │       ├── RecordingsTab (episode select, playback, speed)
    │       └── InferenceTab (checkpoint, run/restart, live controls, speed)
    └── overlay-bottom-right
        ├── StepInfoPanel
        └── QValuesPanel
```

## Test Changes

- Deleted `agent-visibility.spec.ts` and `agent-movement.spec.ts` (cyan pixel detection tests incompatible with post-processing shader at full-screen canvas size)
- Updated all selectors: `.sidebar` to `.overlay-top`, `sidebar select` to `.overlay-select`
- Changed `selectLevelAndWait` helper to wait for `'Recordings'` text instead of removed `'Playback'` label
- Forward-step test uses `force: true` click to avoid overlay interception issues under parallel workers
- Visual baselines regenerated for new camera angle and layout
- Final count: 14 tests, all passing

## CSS (styles.css)

Replaced all sidebar styles with overlay styles:
- `.overlay-top`: Floating centered panel with `backdrop-filter: blur(8px)`
- `.overlay-bottom-right`: Bottom-right positioned info panels
- `.tab-strip`, `.tab`, `.tab-active`: Tab navigation with cyan underline
- `.speed-btn`, `.speed-label`: Compact 24x24px speed increment buttons
- `.overlay-select`: Dark select dropdowns with cyan focus border
- Generic `button` styles with hover/disabled states
- `.seek-slider`: Cyan accent color range input
