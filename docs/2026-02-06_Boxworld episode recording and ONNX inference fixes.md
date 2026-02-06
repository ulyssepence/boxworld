# Boxworld Episode Recording and ONNX Inference Fixes

Three bugs were found and fixed that prevented the Boxworld visualizer from playing back recorded agent episodes and running live ONNX inference.

## Bug 1: Episodes Recorded on Wrong Level Grid

**File:** `training/record.py`

**Symptom:** The agent appeared inside walls or on invalid cells during playback. The sidebar showed positions changing, but the cyan agent sphere was invisible (hidden inside wall geometry).

**Root cause:** `record_episode()` called `env.reset()` at the start of each episode. Since the environment was constructed without a `level_path` (just `width` and `height`), `reset()` triggered `_generate_level(seed=None)`, which generated a random procedural level. This overwrote the grid and agent position that `record_all()` had manually set from the level JSON files.

The result: `state_json` in the database contained `"level": <original JSON level>` but `"agentPosition"` came from a completely different randomly-generated grid. Positions that were valid floor tiles on the random grid appeared as walls on the JSON level grid.

**Fix:** Replaced `obs, info = env.reset()` with `obs, info = env._get_obs(), env._get_info()`, since the caller already configures the env state (grid, agent_pos, has_key) before calling `record_episode()`.

## Bug 2: Immutable Cache on Mutable Episode Data

**File:** `visualize/src/server.ts`

**Symptom:** After fixing Bug 1 and re-recording episodes, the browser still displayed the old broken data. The API returned correct data when tested with `curl`, but the frontend showed stale episode counts and positions.

**Root cause:** The `/api/levels/:levelId` endpoint had `Cache-Control: public, max-age=31536000, immutable` middleware. This endpoint returns episode data from SQLite, which changes whenever episodes are re-recorded. The browser cached the first response and never re-requested it.

**Fix:** Removed the `immutableCache` middleware from the `/api/levels/:levelId` endpoint. The `/api/levels` list endpoint (which only reads static JSON files) retains the immutable cache.

## Bug 3: ONNX Runtime Web Missing .mjs Glue Files

**File:** `visualize/package.json` (build script)

**Symptom:** Clicking "Run Agent" logged: `Inference failed: Error: no available backend found. ERR: [wasm] TypeError: Failed to fetch dynamically imported module: http://localhost:8000/static/ort-wasm-simd-threaded.jsep.mjs`

**Root cause:** The build script copied `ort.min.js` and `*.wasm` files to `static/`, but onnxruntime-web also dynamically imports `.mjs` JavaScript glue modules that bootstrap the WASM backends. These were present in `node_modules/onnxruntime-web/dist/` but not copied to `static/`, causing 404s at runtime.

**Fix:** Added `cp node_modules/onnxruntime-web/dist/*.mjs static/` to the build script.

## Additional Notes

- After fixing Bug 1, the re-recorded episodes show the agent starting at the correct `agentStart` position from each level's JSON file
- The agent was trained on random procedural levels, so it performs poorly on the hand-crafted levels (mostly gets stuck or walks into lava). This is expected behavior, not a bug
- Users who loaded the app before the Bug 2 fix need to hard-refresh (Cmd+Shift+R) to clear the stale cached response from the old immutable header
