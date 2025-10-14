import ScenarioBuilderClient from './ScenarioBuilderClient'
import { loadScenarioTemplates } from '../lib/scenario-templates'

export default async function Page() {
  const templates = await loadScenarioTemplates()
  return <ScenarioBuilderClient templates={templates} />
}
