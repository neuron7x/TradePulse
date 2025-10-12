const LEGACY_SUBJECTS = new Set([
  'Add numeric accelerator backends and benchmarks',
  'docs: clarify commitlint remediation guidance',
]);

const getSubject = (message = '') => message.split('\n', 1)[0].trim();

/** @type {import('@commitlint/types').UserConfig} */
module.exports = {
  extends: ['@commitlint/config-conventional'],
  ignores: [(message = '') => LEGACY_SUBJECTS.has(getSubject(message))],
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
