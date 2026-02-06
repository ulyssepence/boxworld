import * as audio from './audio'

export type Response = { type: 'SUCCESS'; result: any } | { type: 'ERROR'; error: string }
