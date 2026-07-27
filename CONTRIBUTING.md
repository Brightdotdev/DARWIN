# Contributing to Darwin

Thanks for your interest. Darwin is a local-first RAG engine — if you're here, you probably care about that too.

## The philosophy

- **Minimal.** Darwin is meant to be slim and fit into any gap, not get in the way. Before adding something, ask: does this earn its place?
- **Local-first.** Every feature should work offline. External APIs are optional enhancements, never requirements.
- **Solid over clever.** Correct code that's slightly boring beats clever code that breaks in edge cases.

## Guidelines

1. **Keep it minimal.** No unnecessary abstractions, no over-engineering. If a feature adds more weight than value, it doesn't belong.
2. **Test your code.** You can use AI to write it — that's fine. But you're responsible for what ships. Run it, break it, fix it.
3. **No spam.** PRs should do one thing well. If you've got five unrelated changes, open five PRs.
4. **Follow existing patterns.** Look at how the codebase is structured before adding new modules.
5. **No secrets.** Never commit API keys, tokens, or credentials.

## How to contribute

1. **Fork** the repo and create a branch from `main`.
2. **Make your changes.** Keep commits focused.
3. **Test.** At minimum, verify imports and basic commands don't break.
4. **Open a PR** with a clear description of what changed and why.

## Reporting issues

Found a bug? Open an issue at [github.com/BrightDotDev/DARWIN/issues](https://github.com/BrightDotDev/DARWIN/issues).

Include:
- What you expected
- What actually happened
- How to reproduce it (steps, commands, etc.)

## License

By contributing, you agree that your contributions will be licensed under the [MIT with Attribution License](LICENSE).
