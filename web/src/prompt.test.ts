import { describe, expect, it } from 'vitest'
import { insertWord } from './prompt'

describe('insertWord', () => {
  it.each([
    ['', 0, 'fg', 'fg'],
    ['a cat', 5, 'fg', 'a cat, fg'],
    ['a cat', 0, 'fg', 'fg, a cat'],
    ['a cat, ', 7, 'fg', 'a cat, fg'],
    ['a cat, on a mat', 6, 'fg', 'a cat, fg, on a mat'],
    ['a cat, on a mat', 5, 'fg', 'a cat, fg, on a mat'],
  ])('%j at %i', (text, at, word, expected) => {
    expect(insertWord(text, at, word)).toBe(expected)
  })
})
