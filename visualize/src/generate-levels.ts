import * as play from './play'

const start = parseInt(process.argv[2] || '0')
const count = parseInt(process.argv[3] || '10000')

for (let i = 0; i < count; i++) {
  const level = play.generateLevel(start + i)
  if (!level) continue
  console.log(
    JSON.stringify({
      seed: start + i,
      width: level.width,
      height: level.height,
      grid: level.grid,
      agentStart: level.agentStart,
    }),
  )
}
