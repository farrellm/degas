import js from '@eslint/js'
import prettier from 'eslint-config-prettier'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import simpleImportSort from 'eslint-plugin-simple-import-sort'
import { defineConfig, globalIgnores } from 'eslint/config'
import globals from 'globals'
import tseslint from 'typescript-eslint'

const PARENT = 'Import from another folder as `@/…`.'

export default defineConfig([
  globalIgnores(['dist', 'coverage']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.strictTypeChecked,
      tseslint.configs.stylisticTypeChecked,
      reactHooks.configs.flat['recommended-latest'],
      reactRefresh.configs.vite,
      prettier,
    ],
    plugins: { 'simple-import-sort': simpleImportSort },
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { ignoreRestSiblings: true }],
      // With `verbatimModuleSyntax`, `import { type A }` would leave a side-effect import behind.
      '@typescript-eslint/no-import-type-side-effects': 'error',
      'no-duplicate-imports': 'error',
      'simple-import-sort/imports': 'error',
      'simple-import-sort/exports': 'error',
      // `./` for this folder and below, `@/` for everything else.
      'no-restricted-imports': ['error', { patterns: [{ group: ['../*'], message: PARENT }] }],
    },
    languageOptions: {
      ecmaVersion: 2023,
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    // Shared code stays below the features: it can be used by any of them, so it knows none.
    files: ['src/{api,lib,hooks,components}/**/*.{ts,tsx}'],
    ignores: ['**/*.test.{ts,tsx}'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            { group: ['../*'], message: PARENT },
            {
              group: ['@/features/*', '@/app/*'],
              message: 'Shared code must not import a feature.',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['public/sw.js'],
    extends: [js.configs.recommended, prettier],
    languageOptions: { globals: globals.serviceworker },
  },
  {
    files: ['eslint.config.js'],
    extends: [js.configs.recommended, prettier],
    languageOptions: { globals: globals.node },
  },
])
