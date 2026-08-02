import Pitch, { Legend } from './Pitch.jsx'
import { DiagnosticsPanel } from './diagnostics.jsx'
import {
  useDiagnosticsPreference,
  useInsightStream,
  useReplayActions,
  useReplayControl,
  useServiceStatus,
  useStickyReason,
} from './hooks.js'
import { InsightList } from './insights.jsx'
import { useCapabilities } from './jobs.js'
import { Confidence, Header, RuntimeStatus } from './panels.jsx'
import { PipelinePanel } from './pipeline.jsx'
import { Transport } from './transport.jsx'

/**
 * Composes the demo from three independent data sources: service metadata, the
 * SSE stream and the replay controller. Each is a hook, so a stall in one does
 * not block the others — the pitch keeps drawing while `/model` is unreachable.
 *
 * The page serves two readers at once. The main layout is the viewer's: what is
 * being predicted, which way play is running, whether the estimate is worth
 * saying, and what was actually said. Everything an engineer needs instead —
 * raw score, suppression mix, fault counts, schema state — sits in a panel
 * below it, so neither reader pays for the other.
 */
export default function App() {
  const { model, ready, readyReason, runtime } = useServiceStatus()
  const stream = useInsightStream()
  const { replay, matches, control, restart, selectMatch, pending, stale, lastSeen } =
    useReplayControl()
  const capabilities = useCapabilities()
  const stickyReason = useStickyReason(stream.reason, stream.framesSeen)
  const [showDiagnostics, onToggleDiagnostics] = useDiagnosticsPreference()

  const threshold = model?.decision_threshold ?? 0.5
  const switching = pending || Boolean(stream.switchingTo)
  const showPipeline = Boolean(capabilities?.pipeline_controls)
  // Undefined until `/capabilities` answers, and treated as "yes" until then:
  // the transport is the normal case, and hiding it during the first moments of
  // every local page load would be a visible flicker to avoid a wrong guess
  // that only a public deployment can make.
  const showTransport = capabilities?.replay_controls !== false

  const { onRestart, onSelectMatch } = useReplayActions({
    restart,
    selectMatch,
    reopen: stream.reopen,
    status: stream.status,
  })

  return (
    <div className="app">
      <a className="skip-link" href="#insights">
        Skip to insights
      </a>

      <Header
        ready={ready}
        readyReason={readyReason}
        showDiagnostics={showDiagnostics}
        onToggleDiagnostics={onToggleDiagnostics}
        matches={showTransport ? matches : []}
        currentMatch={replay?.match_id ?? null}
        onSelectMatch={onSelectMatch}
        switching={switching}
      />

      <main>
        <section className="pitch-panel">
          <Pitch frame={stream.frame} />
          <Legend />
          {showTransport ? (
            <Transport
              frame={stream.frame}
              replay={replay}
              control={control}
              restart={onRestart}
              pending={pending}
              status={stream.status}
              stale={stale}
              lastSeen={lastSeen}
              switchingTo={stream.switchingTo}
            />
          ) : (
            <p className="transport-readonly">
              Server-controlled replay, looping continuously. Playback controls are disabled on the
              public demo because every viewer shares one replay.
            </p>
          )}
        </section>

        <aside aria-label="Live analysis">
          <Confidence
            probability={stream.probability}
            threshold={threshold}
            history={stream.history}
            isMl={model ? model.is_ml : null}
            reason={stickyReason}
            horizonS={model?.horizon_s}
          />
          <InsightList
            insights={stream.insights}
            status={stream.status}
            endedFrames={stream.endedFrames}
            reason={stickyReason}
            framesSeen={stream.framesSeen}
          />
        </aside>
      </main>

      {showDiagnostics && (
        <DiagnosticsPanel
          model={model}
          ready={ready}
          readyReason={readyReason}
          replay={replay}
          stream={stream}
          stickyReason={stickyReason}
          probability={stream.probability}
          threshold={threshold}
          stale={stale}
          lastSeen={lastSeen}
        />
      )}

      {showPipeline && <PipelinePanel />}

      <footer>
        <RuntimeStatus runtime={runtime} />
        <p>
          Predictions are estimates over a short horizon, not statements of fact. This is a
          demonstration of a tracking-data pipeline: it is not an injury, betting, officiating or
          player-safety system, and must not be used as one. Reported evaluation results were
          measured on Metrica sample matches and do not describe the model serving this page. Not
          affiliated with any league, broadcaster or data provider.
        </p>
      </footer>
    </div>
  )
}
