# pikachu-thinking

A Claude Code mod for the desktop app: a running Pikachu-style mouse replaces the
"Thinking…" indicator, and shows on commands and tool calls while they run. The
terminal keeps its normal spinner.

## Install

In a terminal, start `claude` and type:

```
/plugin install pikachu-thinking --marketplace vipul2710/ai-rag-pipeline
```

Answer `y` to add the marketplace, then pick the **user** scope. Then fully quit and
reopen the Claude desktop app.

Mods are early access: if it doesn't show, add this to `~/.claude/settings.json`:

```json
"env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1" }
```

## Remove

```
/plugin uninstall pikachu-thinking
```
