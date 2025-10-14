import 'server-only'

import fs from 'node:fs/promises'
import path from 'node:path'

import type { ScenarioConfig, ScenarioTemplate } from './scenario-templates-types'

const templatesPath = path.join(process.cwd(), 'configs', 'scenarios', 'templates.json')

function assertConfig(config: any, index: number): ScenarioConfig {
  if (!config || typeof config !== 'object') {
    throw new TypeError(`Scenario template at index ${index} is missing defaults`)
  }
  const { initialBalance, riskPerTrade, maxPositions, timeframe } = config as ScenarioConfig
  if (
    typeof initialBalance !== 'number' ||
    typeof riskPerTrade !== 'number' ||
    typeof maxPositions !== 'number' ||
    typeof timeframe !== 'string'
  ) {
    throw new TypeError(`Scenario template defaults at index ${index} have invalid types`)
  }
  return { initialBalance, riskPerTrade, maxPositions, timeframe }
}

export async function loadScenarioTemplates(): Promise<ScenarioTemplate[]> {
  const raw = await fs.readFile(templatesPath, 'utf-8')
  const parsed = JSON.parse(raw)
  if (!Array.isArray(parsed)) {
    throw new TypeError('Scenario templates JSON must be an array')
  }
  return parsed.map((item, index) => {
    if (!item || typeof item !== 'object') {
      throw new TypeError(`Scenario template at index ${index} must be an object`)
    }
    const { id, label, description, defaults, notes } = item as Record<string, unknown>
    if (
      typeof id !== 'string' ||
      typeof label !== 'string' ||
      typeof description !== 'string' ||
      !Array.isArray(notes) ||
      !notes.every((note) => typeof note === 'string')
    ) {
      throw new TypeError(`Scenario template at index ${index} has invalid metadata`)
    }
    return {
      id,
      label,
      description,
      defaults: assertConfig(defaults, index),
      notes,
    }
  })
}
