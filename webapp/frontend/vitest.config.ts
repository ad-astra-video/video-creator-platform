import { defineConfig } from 'vitest/config'

// Unit tests for the pure, dependency-free lib modules (media decoding etc.).
// These must run in a plain Node environment with no DOM/browser mocks.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['frontend/**/*.test.ts'],
  },
})
