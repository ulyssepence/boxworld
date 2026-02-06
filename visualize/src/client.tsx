import * as React from 'react'
import * as ReactDOM from "react-dom/client"
import * as audio from "./audio"
import * as play from "./play"
import * as t from "./types"

export type State = {
}

const audioPlayer = new audio.Player('/static')
const websocketProtocol = document.location.protocol == 'http:' ? 'ws' : 'wss'
const mailbox = new mail.Box(new WebSocket(`${websocketProtocol}://${window.location.host}`))

export function initialState(): State {
  return {
  }
}

function onMessage(state: State, message: t.ToClientMessage): State {
  switch (message.type) {
    case 'GAME':
      return { ...state, game: message.game }
  }
}

function View() {
  return (
    <></>
  )
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <View />
  </React.StrictMode>
)
