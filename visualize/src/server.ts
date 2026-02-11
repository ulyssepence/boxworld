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

// Disable caching for all responses (dev server)
app.use((_req, res, next) => {
  res.set('Cache-Control', 'no-store')
  next()
})

// API endpoints
app.get('/api/checkpoints', (_req, res) => {
  res.json({ checkpoints: database.getCheckpoints() })
})

app.get('/api/levels', (_req, res) => {
  res.json({ levels: database.getLevels(LEVELS_DIR) })
})

app.get('/api/levels/:levelId', (req, res) => {
  const result = database.getLevelWithEpisodes(req.params.levelId, LEVELS_DIR)
  if (!result) {
    res.status(404).json({ error: 'Level not found' })
    return
  }
  res.json(result)
})

app.get('/api/curated-seeds', (_req, res) => {
  res.json({ seeds: database.getCuratedSeeds() })
})

// Serve ONNX checkpoint files
app.use(
  '/checkpoints',
  express.static(path.join(DATA_DIR, 'checkpoints'), {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.onnx')) {
        res.set('Content-Type', 'application/octet-stream')
      }
    },
  }),
)

// Static files
app.use('/static', express.static(path.resolve(import.meta.dirname, '../static')))

app.get('/', (_req, res) => {
  res.sendFile(path.resolve(import.meta.dirname, '../static/index.html'))
})

const PORT = 8000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server listening on http://localhost:${PORT}`)
})
