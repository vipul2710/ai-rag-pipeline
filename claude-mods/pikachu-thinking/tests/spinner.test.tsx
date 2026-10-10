import { expect, test } from 'claude-code/testing'

const SPINNER = {
  component: 'Spinner',
  props: { word: 'Working', message: null, suffix: '…', mode: 'thinking' },
} as const

test('desktop draws the running Pikachu and its label', async $ => {
  const ui = await $.ui.mount({ plugin: 'pikachu-thinking', surface: 'desktop', ...SPINNER })
  expect(await ui.find({ type: 'Svg' })).toBeDefined()
  expect(await ui.find({ type: 'Text', text: 'Pikachu is thinking…' })).toBeDefined()
  await ui.unmount()
})

test('desktop shows the step word while using tools', async $ => {
  const ui = await $.ui.mount({
    plugin: 'pikachu-thinking',
    surface: 'desktop',
    ...SPINNER,
    props: { ...SPINNER.props, word: 'Creating notes.md', mode: 'tool-use' },
  })
  expect(await ui.find({ type: 'Text', text: 'Creating notes.md…' })).toBeDefined()
  await ui.unmount()
})

test('terminal keeps the engine spinner', async ($, on) => {
  on('ui.render', { component: 'Spinner' }, ($, e) => {
    const { Text } = $.ui.resolve(e)
    return <Text>engine</Text>
  })
  const ui = await $.ui.mount({ plugin: 'pikachu-thinking', surface: 'terminal', ...SPINNER })
  expect(await ui.find({ type: 'Text', text: 'engine' })).toBeDefined()
  await ui.unmount()
})

const BASH = {
  component: 'ToolUse',
  props: {
    tool_use_id: 'toolu_1', tool: 'Bash', input: { command: 'npm   test' },
    isRunning: true, isErrored: false, isInterrupted: false,
  },
} as const

test('a running command shows Pikachu and the command', async $ => {
  const ui = await $.ui.mount({ plugin: 'pikachu-thinking', surface: 'desktop', ...BASH })
  expect(await ui.find({ type: 'Svg' })).toBeDefined()
  expect(await ui.find({ type: 'Text', text: 'Bash: npm test…' })).toBeDefined()
  await ui.unmount()
})

test('a finished command gets the engine row back', async ($, on) => {
  on('ui.render', { component: 'ToolUse' }, ($, e) => {
    const { Text } = $.ui.resolve(e)
    return <Text>engine row</Text>
  })
  const ui = await $.ui.mount({
    plugin: 'pikachu-thinking', surface: 'desktop', ...BASH,
    props: { ...BASH.props, isRunning: false, output: { stdout: 'ok' } },
  })
  expect(await ui.find({ type: 'Svg' })).toBeUndefined()
  expect(await ui.find({ type: 'Text', text: 'engine row' })).toBeDefined()
  await ui.unmount()
})

test('a folded group of reads shows the running one', async $ => {
  const call = { tool: 'Read', isErrored: false, isInterrupted: false }
  const ui = await $.ui.mount({
    plugin: 'pikachu-thinking', surface: 'desktop', component: 'ToolGroup',
    props: {
      isActive: true, isExpanded: false,
      calls: [
        { ...call, input: { file_path: 'a.md' }, isRunning: false, output: 'x' },
        { ...call, input: { file_path: 'b.md' }, isRunning: true },
      ],
    },
  })
  expect(await ui.find({ type: 'Text', text: 'Read: b.md… (1 done)' })).toBeDefined()
  await ui.unmount()
})
