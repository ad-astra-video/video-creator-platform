// Opt-in envelope encryption for custom prompt storage.
//
// Two modes (user decision, surfaced in the Prompt Builder settings tab):
//  - NO encryption key  -> prompts stored as PLAINTEXT in D1 (server-readable).
//  - encryption key set -> prompts stored as AES-256-GCM ciphertext; the data key (DEK)
//    is wrapped under a PBKDF2(passphrase)-derived KEK. The passphrase never leaves the
//    browser and is never sent to the server, so only the client can decrypt.
//
// Salt: generated per-user via newSalt() and persisted alongside (it need not be secret;
// it is not used as the passphrase). PBKDF2 at OWASP-2023 recommended ~310k iterations.

export const PBKDF2_ITERATIONS = 310_000

export interface PromptCrypto {
  enc: string
  keyEnc: string
  kdfSalt: string
}

const te = new TextEncoder()
const td = new TextDecoder()
const b64 = (u: Uint8Array) => btoa(Array.from(u, (b) => String.fromCharCode(b)).join(''))
const unb64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0))
const iv = () => crypto.getRandomValues(new Uint8Array(12))

async function kek(password: string, salt: Uint8Array): Promise<CryptoKey> {
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: salt as unknown as BufferSource, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
    await crypto.subtle.importKey('raw', te.encode(password) as unknown as BufferSource, 'PBKDF2', false, ['deriveKey']),
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  )
}

const gcm = (key: CryptoKey, data: Uint8Array, ivv: Uint8Array, mode: 'encrypt' | 'decrypt') =>
  crypto.subtle[mode](
    { name: 'AES-GCM', iv: ivv as unknown as BufferSource },
    key,
    data as unknown as BufferSource,
  ).then((r) => new Uint8Array(r))

export const newSalt = () => b64(crypto.getRandomValues(new Uint8Array(16)))

/** Encrypt prompt JSON under a random DEK; wrap the DEK under a PBKDF2(passphrase) KEK. */
export async function encryptPromptsKeyed(
  passphrase: string,
  salt: string,
  plaintext: string,
): Promise<PromptCrypto> {
  const dek = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt'])
  const i1 = iv()
  const ct = await gcm(dek, te.encode(plaintext), i1, 'encrypt')
  const blob = new Uint8Array(12 + ct.length)
  blob.set(i1)
  blob.set(ct, 12)
  const i2 = iv()
  const dekRaw = new Uint8Array(await crypto.subtle.exportKey('raw', dek))
  const wrapped = await gcm(await kek(passphrase, unb64(salt)), dekRaw, i2, 'encrypt')
  const blob2 = new Uint8Array(12 + wrapped.length)
  blob2.set(i2)
  blob2.set(wrapped, 12)
  return { enc: b64(blob), keyEnc: b64(blob2), kdfSalt: salt }
}

/** Decrypt with the passphrase (throws on the wrong passphrase / a tampered blob). */
export async function decryptPromptsKeyed(passphrase: string, c: PromptCrypto): Promise<string> {
  const k = await kek(passphrase, unb64(c.kdfSalt))
  const d2 = unb64(c.keyEnc)
  const i2 = d2.slice(0, 12)
  const dekRaw = await gcm(k, d2.slice(12), i2, 'decrypt')
  const dek = await crypto.subtle.importKey('raw', dekRaw as unknown as BufferSource, { name: 'AES-GCM' }, true, ['decrypt'])
  const d1 = unb64(c.enc)
  const i1 = d1.slice(0, 12)
  return td.decode(await gcm(dek, d1.slice(12), i1, 'decrypt'))
}
