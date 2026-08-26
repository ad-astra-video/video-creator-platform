import { describe, expect, it } from 'vitest'
import { encryptPromptsKeyed, decryptPromptsKeyed, newSalt } from './prompt-crypto'

// node >=20 exposes global crypto.subtle + btoa/atob, so these run as-is.

describe('prompt-crypto envelope encryption', () => {
  it('round-trips ciphertext back to the plaintext', async () => {
    const salt = newSalt()
    const plain = { enhancerT2V: 'secret custom prompt', gapFillRubric: 'another one' }
    const c = await encryptPromptsKeyed('hunter2', salt, JSON.stringify(plain))
    const out = await decryptPromptsKeyed('hunter2', c)
    expect(JSON.parse(out)).toEqual(plain)
  })

  it('the wrong passphrase rejects', async () => {
    const salt = newSalt()
    const c = await encryptPromptsKeyed('right-key', salt, 'secret')
    await expect(decryptPromptsKeyed('wrong-key', c)).rejects.toThrow()
  })

  it('produces a unique IV (ciphertext differs) across two encryptions of the same text', async () => {
    const salt = newSalt()
    const a = await encryptPromptsKeyed('pw', salt, 'same text')
    const b = await encryptPromptsKeyed('pw', salt, 'same text')
    expect(a.enc).not.toBe(b.enc)
  })

  it('ciphertext never contains the plaintext substring', async () => {
    const salt = newSalt()
    const secret = 'ultra-secret-custom-prompt-xyz'
    const c = await encryptPromptsKeyed('pw', salt, secret)
    expect(c.enc).not.toContain(secret)
    expect(c.keyEnc).not.toContain(secret)
  })
})
