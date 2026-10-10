import type { Register } from 'claude-code'

import { PIKACHU_SVG } from './pikachu'

const MAX_LABEL = 80

// One line saying what a running tool is doing: `Bash: npm test`, `Read: notes.md`.
const describeCall = (tool: string, input: unknown): string => {
  const fields = (input ?? {}) as Record<string, unknown>
  const detail = [fields.command, fields.description, fields.file_path, fields.path, fields.pattern, fields.url, fields.query]
    .find((value): value is string => typeof value === 'string' && value.length > 0)
  const line = detail ? `${tool}: ${detail.replace(/\s+/g, ' ')}` : tool
  return line.length > MAX_LABEL ? `${line.slice(0, MAX_LABEL - 1)}…` : line
}

export const register: Register = on => {
  // The turn's own "thinking / working" row.
  on('ui.render', { component: 'Spinner' }, ($, e, next) => {
    // Only the desktop draws Svg; every other surface keeps the engine's line.
    if (e.surface !== 'desktop') {
      return next(e)
    }

    const { Box, Svg, Text } = $.ui.resolve(e)
    const { message, word, suffix, mode } = e.props
    const label = message ?? (mode === 'thinking' ? 'Pikachu is thinking' : word)

    return (
      <Box key="pikachu" flexDirection="row" alignItems="center" gap={1}>
        <Svg source={PIKACHU_SVG} alt="Pikachu running" width={60} height={39} />
        <Text dimColor>{`${label}${suffix}`}</Text>
      </Box>
    )
  })

  // A single tool call (a command, an edit, a fetch) while it runs; the
  // engine's own row comes back the moment it finishes.
  on('ui.render', { component: 'ToolUse' }, ($, e, next) => {
    if (e.surface !== 'desktop' || !e.props.isRunning) {
      return next(e)
    }

    const { Box, Svg, Text } = $.ui.resolve(e)

    return (
      <Box key="pikachu-tool" flexDirection="row" alignItems="center" gap={1}>
        <Svg source={PIKACHU_SVG} alt="Pikachu running" width={40} height={26} />
        <Text dimColor>{`${describeCall(e.props.tool, e.props.input)}…`}</Text>
      </Box>
    )
  })

  // A folded run of reads and searches while one of its calls runs.
  on('ui.render', { component: 'ToolGroup' }, ($, e, next) => {
    const running = e.props.calls.find(call => call.isRunning)
    if (e.surface !== 'desktop' || !running || e.props.isExpanded) {
      return next(e)
    }

    const { Box, Svg, Text } = $.ui.resolve(e)
    const done = e.props.calls.filter(call => !call.isRunning).length
    const progress = done > 0 ? ` (${done} done)` : ''

    return (
      <Box key="pikachu-group" flexDirection="row" alignItems="center" gap={1}>
        <Svg source={PIKACHU_SVG} alt="Pikachu running" width={40} height={26} />
        <Text dimColor>{`${describeCall(running.tool, running.input)}…${progress}`}</Text>
      </Box>
    )
  })
}
