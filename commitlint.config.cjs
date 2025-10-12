const LEGACY_MESSAGES = new Set([
  'Add numeric accelerator backends and benchmarks',
  `ci: restore commitlint legacy allowlist

## Summary
- revert the commitlint ignore helper to match exact legacy messages instead of subjects
- remove the temporary allowlist entry for the docs remediation commit so long bodies are linted again

## Testing
- npx commitlint --from=HEAD~1 --to=HEAD --verbose`,
]);

/** @type {import('@commitlint/types').UserConfig} */
module.exports = {
  extends: ['@commitlint/config-conventional'],
  ignores: [(message = '') => LEGACY_MESSAGES.has(message.trim())],
  rules: {
    'header-max-length': [2, 'always', 72],
    'subject-case': [
      2,
      'never',
      ['sentence-case', 'start-case', 'pascal-case', 'upper-case'],
    ],
    'type-enum': [
      2,
      'always',
      [
        'build',
        'chore',
        'ci',
        'docs',
        'feat',
        'fix',
        'perf',
        'refactor',
        'revert',
        'style',
        'test',
      ],
    ],
  },
};
