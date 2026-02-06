import express from 'express'
import http from 'http'
import path from 'path'
import * as ws from 'ws'
import * as play from './play'
import * as t from './types'

const state = {}

const app = express()
app.use(express.json())

app.get('/', (req, res) => {
  res.sendFile(path.resolve('static/index.html'))
})

app.use(express.static('./'))

const webServer = http.createServer(app)

webServer.listen(8000, '0.0.0.0')
