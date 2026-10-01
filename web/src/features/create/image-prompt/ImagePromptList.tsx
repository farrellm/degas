import { thumbUrl, type Asset, type ImagePromptOptions } from '../api'
import { assetLabel } from '../assets'
import { MAX_PROMPTS, promptSummary, type PromptUnit } from '../imagePrompt'

interface Props {
  units: PromptUnit[]
  /** The family's image prompt models in the Drive index. */
  index: Asset[] | undefined
  steps: number
  options: ImagePromptOptions
  onOpen: (key: string) => void
  onAdd: () => void
}

/** The Image prompts rows in Create: each unit's pictures, model and summary; tap to edit. */
export function PromptList({ units, index, steps, options, onOpen, onAdd }: Props) {
  return (
    <div className="control-group" role="group" aria-labelledby="prompts-label">
      <div className="setting">
        <span className="setting-label" id="prompts-label">
          Image prompts
        </span>
        <button
          type="button"
          className="row-action"
          aria-label="Add image prompt"
          disabled={units.length >= MAX_PROMPTS}
          onClick={onAdd}
        >
          Add
        </button>
      </div>
      {units.map((unit) => {
        const missing = !!unit.model && !!index && !index.some((a) => a.path === unit.model)
        const name = unit.model ? assetLabel(unit.model, index) : 'Choose a model'
        const [first, second] = unit.pictures
        return (
          <div key={unit.key} className={missing ? 'control-unit missing' : 'control-unit'}>
            <button
              type="button"
              className="control-open"
              aria-label={`Edit image prompt: ${name}`}
              onClick={() => {
                onOpen(unit.key)
              }}
            >
              {first ? (
                <span className={second ? 'prompt-thumbs stacked' : 'prompt-thumbs'}>
                  {second && <img className="source-thumb" src={thumbUrl(second.sha)} alt="" />}
                  <img className="source-thumb" src={thumbUrl(first.sha)} alt="" />
                </span>
              ) : (
                <span className="control-thumb empty" aria-hidden />
              )}
              <span className="control-text">
                <span className={unit.model ? 'control-name' : 'control-name none'}>{name}</span>
                <span className="control-meta">{promptSummary(unit, steps, options)}</span>
              </span>
            </button>
            {missing && <p className="row-warning">Not found in Drive. Pick another model.</p>}
          </div>
        )
      })}
    </div>
  )
}
