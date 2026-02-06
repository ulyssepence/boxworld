import express from 'express'
import path from 'path'
import * as db from './db'

const DATA_DIR = path.resolve(import.meta.dirname, '../../data')
const LEVELS_DIR = path.join(DATA_DIR, 'levels')
const DB_PATH = path.join(DATA_DIR, 'db.sqlite')

const database = new db.DB(DB_PATH)
database.initialize()

const app = express()
app.use(express.json())

// Cache headers for level data (immutable once created)
function immutableCache(_req: express.Request, res: express.Response, next: express.NextFunction) {
  res.set('Cache-Control', 'public, max-age=31536000, immutable')
  next()
}

// API endpoints
app.get('/api/checkpoints', (_req, res) => {
  res.json({ checkpoints: database.getCheckpoints() })
})

app.get('/api/levels', immutableCache, (_req, res) => {
  res.json({ levels: database.getLevels(LEVELS_DIR) })
})

app.get('/api/levels/:levelId', immutableCache, (req, res) => {
  const result = database.getLevelWithEpisodes(req.params.levelId, LEVELS_DIR)
  if (!result) {
    res.status(404).json({ error: 'Level not found' })
    return
  }
  res.json(result)
})

// Static files
app.use('/static', express.static(path.resolve(import.meta.dirname, '../static')))

app.get('/', (_req, res) => {
  res.sendFile(path.resolve(import.meta.dirname, '../static/index.html'))
})

const PORT = 8000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server listening on http://localhost:${PORT}`)
})
