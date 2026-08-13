import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle2, ClipboardCopy, FileText, FolderOpen, KeyRound, Loader2, Mail, Server, ShieldCheck, X } from 'lucide-react'
import { ApiClient } from '../lib/api-client'
import { resetBackendCredentials } from '../lib/backend'
import { supportsFileSystemAccess, getProjectAssetsName, pickProjectAssetsFolder } from '../lib/runtime/fs-access'
import { Button } from './ui/button'

type Mode = 'signup' | 'login'
type LoginMethod = 'email' | 'backup'
// Sign-up onboarding wizard pages (shown inside the single card).
type Page = 'signup' | 'code' | 'folder' | 'license'

const WIZARD_STEPS: { id: Page; label: string }[] = [
  { id: 'signup', label: 'Sign up' },
  { id: 'code', label: 'Recovery code' },
  { id: 'folder', label: 'Folder' },
  { id: 'license', label: 'Licenses' },
]

/**
 * Web-only first-run gate. Shown on first launch of the static web app instead of the
 * desktop setup flow / the stuck "Loading settings..." spinner.
 *
 * From the account side it mirrors the desktop flow's Sign Up / Login split:
 *  - Sign Up: a single-card, paged wizard: (1) create the account (recovery email is
 *    optional) -> (2) show the one-time BACKUP recovery code to save (shown once, never
 *    emailed) -> (3) choose where project assets are saved -> (4) review the license /
 *    open-source model notices and acknowledge. Finishing is gated on that acknowledgement.
 *  - Login: recover an existing account by EMAIL (request code -> confirm) OR by the
 *    one-time BACKUP code the user saved at sign-up. Both rotate the API key back onto the
 *    same externalUserId so the stored settings come back.
 *
 * It does NOT need settings to be loaded first — that is the whole point: it is reachable
 * from the state where settings can't load (no key yet).
 */
export function WebFirstRun({ onComplete }: { onComplete?: () => void }) {
  const [mode, setMode] = useState<Mode>('signup')
  const [loginMethod, setLoginMethod] = useState<LoginMethod>('email')
  const [loginAwaitingCode, setLoginAwaitingCode] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Sign-up wizard position.
  const [page, setPage] = useState<Page>('signup')
  // True when the backupCode currently shown came from a Login-by-backup rotation (so its
  // "continue" completes instead of advancing through the sign-up wizard).
  const [fromBackupLogin, setFromBackupLogin] = useState(false)

  // Sign Up fields
  const [folderName, setFolderName] = useState<string | null>(null)
  const [email, setEmail] = useState('')
  // The platform (Worker / API base) URL this app talks to. Prefilled with the effective
  // value from config, editable here so a user can point at a different Worker without a rebuild.
  const [platformUrl, setPlatformUrl] = useState('')
  // The one-time backup code returned by provision — shown once, only in memory.
  const [backupCode, setBackupCode] = useState<string | null>(null)

  // License / notices viewer (fetches LICENSE.txt / NOTICES.md).
  const [doc, setDoc] = useState<{ title: string; body: string } | null>(null)
  const [docLoading, setDocLoading] = useState(false)

  // Login (recovery) fields
  const [recoveryEmail, setRecoveryEmail] = useState('')
  const [recoveryCode, setRecoveryCode] = useState('')
  const [backupInput, setBackupInput] = useState('')

  useEffect(() => {
    void getProjectAssetsName().then((n) => n && setFolderName(n)).catch(() => {})
    void window.electronAPI.getBackend().then((b) => b.url && setPlatformUrl(b.url)).catch(() => {})
  }, [])

  const supportedFolder = supportsFileSystemAccess()

  const chooseFolder = async () => {
    setError(null)
    try {
      await pickProjectAssetsFolder()
      const n = await getProjectAssetsName()
      setFolderName(n || 'selected folder')
    } catch (e) {
      // User cancelled or no FS Access — do not block, they can skip.
      void getProjectAssetsName().then((n) => n && setFolderName(n)).catch(() => {})
    }
  }

  // Persist an API key so the rest of the app (backendFetch getKey) can load settings.
  const persistKey = (apiKey: string) => {
    try {
      localStorage.setItem('vcp_key', apiKey)
    } catch {
      /* ignore */
    }
  }

  const markLicenseAck = () => {
    try {
      localStorage.setItem('vcp_license', '1')
    } catch {
      /* ignore */
    }
  }

  const persistPlatformUrl = (url: string) => {
    const trimmed = url.trim().replace(/\/+$/, '')
    try {
      if (trimmed) {
        localStorage.setItem('vcp_platform_url', trimmed)
      } else {
        localStorage.removeItem('vcp_platform_url')
      }
    } catch {
      /* ignore */
    }
  }

  const installationId = (() => {
    try {
      return localStorage.getItem('vcp_installation_id') ?? crypto.randomUUID()
    } catch {
      return 'web-install'
    }
  })()

  // ---- Sign Up step 1: provision an account (mints key + one-time backup code) ----
  const createAccount = async () => {
    setError(null)
    setMessage(null)
    setBusy(true)
    // Persist the chosen platform URL before provisioning so the request goes to the
    // intended Worker. Blanks clear any previous override (falls back to config.js).
    persistPlatformUrl(platformUrl)
    resetBackendCredentials()
    const provision = await ApiClient.provisionWorker(installationId)
    if (provision.ok && provision.data.apiKey) {
      persistKey(provision.data.apiKey)
      if (email.trim()) {
        await ApiClient.linkEmail(email.trim()).catch(() => null)
      }
      setBusy(false)
      if (provision.data.backupCode) {
        setBackupCode(provision.data.backupCode)
        setFromBackupLogin(false)
        setPage('code')
      } else {
        // No backup code returned — go straight to folder selection.
        setPage('folder')
      }
      return
    }
    if (!provision.ok && provision.status !== 409) {
      const e = ((provision as { ok: false; error: any }).error ?? 'unknown')
      setError("Could not create the account: " + (e?.error ?? e?.message ?? e))
      setBusy(false)
      return
    }
    // 409 — an account already exists on this browser; send them to Login to recover it.
    setBusy(false)
    setMessage("An account already exists on this browser. Sign in below to recover it and its settings.")
    setMode('login')
  }

  // ---- Login: recover by email ----
  const sendRecoveryCode = async () => {
    setError(null)
    setMessage(null)
    if (!recoveryEmail.trim()) {
      setError('Enter the email you linked to your account.')
      return
    }
    setBusy(true)
    const result = await ApiClient.requestRecovery(recoveryEmail.trim())
    setBusy(false)
    setLoginAwaitingCode(true)
    if (result.ok) {
      setMessage('If an account uses that email, a recovery code is on its way. Enter it below to sign back in.')
    } else {
      setError((result.error as any)?.error ?? result.error?.message ?? 'Could not send a recovery code.')
    }
  }

  const confirmRecovery = async () => {
    setError(null)
    setMessage(null)
    if (!recoveryEmail.trim() || !recoveryCode.trim()) {
      setError('Enter both the email and the recovery code.')
      return
    }
    setBusy(true)
    const result = await ApiClient.confirmRecovery(recoveryEmail.trim(), recoveryCode.trim())
    setBusy(false)
    if (result.ok) {
      if (result.data.apiKey) persistKey(result.data.apiKey)
      setMessage('Welcome back. Your key was rotated — your account and settings are restored.')
      onComplete?.()
    } else {
      setError((result.error as any)?.error ?? result.error?.message ?? 'That code did not work.')
    }
  }

  // ---- Login: recover by one-time backup code (no email) ----
  const recoverByBackup = async () => {
    setError(null)
    setMessage(null)
    if (!backupInput.trim()) {
      setError('Enter the backup code you saved at sign-up.')
      return
    }
    setBusy(true)
    const result = await ApiClient.recoverByBackupCode(backupInput.trim().toUpperCase())
    setBusy(false)
    if (result.ok) {
      if (result.data.apiKey) persistKey(result.data.apiKey)
      // A fresh backup code was minted (the presented one is consumed) — show it once.
      if (result.data.backupCode) {
        setBackupCode(result.data.backupCode)
        setFromBackupLogin(true)
        setBackupInput('')
        return // stay on the screen so the user can save the NEW code
      }
      setMessage('Welcome back. Your key was rotated — your account and settings are restored.')
      onComplete?.()
    } else {
      setError((result.error as any)?.error ?? result.error?.message ?? 'That code did not work.')
    }
  }

  const copyBackup = async () => {
    if (!backupCode) return
    try {
      await navigator.clipboard.writeText(backupCode)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* ignore */
    }
  }

  // Advance past the "save your recovery code" screen.
  const handleCodeContinue = () => {
    if (fromBackupLogin) {
      onComplete?.()
      return
    }
    // Clear the backup code so the wizard advances to the folder page (the code is
    // intentionally shown only once, and only kept in memory).
    setBackupCode(null)
    setPage('folder')
  }

  // Open the license / notices doc in the in-card viewer.
  const openDoc = async (kind: 'license' | 'notices') => {
    setDocLoading(true)
    setError(null)
    try {
      const text =
        kind === 'license'
          ? await window.electronAPI.fetchLicenseText()
          : await window.electronAPI.getNoticesText()
      setDoc({
        title: kind === 'license' ? 'Software & Model License' : 'Open-Source Model Notices & Attribution',
        body: text,
      })
    } catch (e) {
      setDoc({ title: 'License & Notices', body: 'Could not load the document: ' + (e instanceof Error ? e.message : String(e)) })
    } finally {
      setDocLoading(false)
    }
  }

  const finish = () => {
    markLicenseAck()
    onComplete?.()
  }

  // Show tab switcher only on the first sign-up page (or on login); hide during the wizard.
  const showTabs = !backupCode && (mode === 'login' || page === 'signup')

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-2xl border border-zinc-800 bg-zinc-950 text-zinc-100 shadow-2xl overflow-hidden">
        {/* Header + mode tabs */}
        <div className="px-6 pt-6 pb-4 border-b border-zinc-800">
          <div className="flex items-center gap-2 mb-4">
            <ShieldCheck className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-semibold">Welcome to Video Creator</h1>
          </div>
          <p className="text-sm text-zinc-400 mb-5">
            Your account lives in the cloud — sign up to create one, or sign in to recover a
            previous account and its settings from any browser.
          </p>

          {showTabs && (
            <div className="grid grid-cols-2 gap-1 rounded-lg bg-zinc-900 p-1">
              <button
                onClick={() => { setMode('signup'); setError(null); setMessage(null); setLoginAwaitingCode(false) }}
                className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${mode === 'signup' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
              >
                Sign Up
              </button>
              <button
                onClick={() => { setMode('login'); setError(null); setMessage(null); setLoginAwaitingCode(false) }}
                className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${mode === 'login' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
              >
                Login
              </button>
            </div>
          )}
        </div>

        <div className="px-6 py-5 space-y-5">
          {/* Wizard step indicator (sign-up flow) */}
          {mode === 'signup' && !backupCode && page !== 'signup' && (
            <div className="flex items-center gap-1.5">
              {WIZARD_STEPS.map((s, i) => {
                const active = page === s.id
                const passed = WIZARD_STEPS.findIndex((x) => x.id === page) > i
                return (
                  <div key={s.id} className="flex items-center gap-1.5 flex-1">
                    <div
                      className={`flex items-center justify-center h-5 w-5 rounded-full text-[10px] font-bold ${
                        active ? 'bg-primary text-black' : passed ? 'bg-emerald-600/70 text-white' : 'bg-zinc-800 text-zinc-500'
                      }`}
                    >
                      {passed ? <CheckCircle2 className="h-3 w-3" /> : i + 1}
                    </div>
                    <span className={`text-[11px] ${active ? 'text-white' : 'text-zinc-500'}`}>{s.label}</span>
                    {i < WIZARD_STEPS.length - 1 && <div className="flex-1 h-px bg-zinc-800" />}
                  </div>
                )
              })}
            </div>
          )}

          {/* One-time backup code (shown once after sign-up or backup recovery) */}
          {backupCode ? (
            <div className="rounded-lg border border-amber-700/50 bg-amber-950/40 p-4 space-y-3">
              <div className="flex items-center gap-2">
                <KeyRound className="h-4 w-4 text-amber-400" />
                <span className="text-sm font-semibold text-amber-200">Save your recovery code</span>
              </div>
              <p className="text-xs text-amber-100/80 leading-relaxed">
                This code <b>never expires and is shown only once</b> — it is never emailed or sent
                again. Save it somewhere safe (password manager / paper). Use it on the <b>Login</b>{" "}
                tab to recover this account if you lose this browser — no email needed.
              </p>
              <div className="flex items-center gap-2 rounded-md bg-zinc-950 px-3 py-2">
                <code className="flex-1 text-sm font-mono tracking-wider text-amber-100 break-all">{backupCode}</code>
                <Button variant="outline" size="sm" onClick={copyBackup}>
                  <ClipboardCopy className="h-3.5 w-3.5 mr-1" />
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              </div>
              <Button onClick={handleCodeContinue} className="w-full">
                <CheckCircle2 className="h-4 w-4 mr-1" />
                I've saved it — continue
              </Button>
            </div>
          ) : mode === 'signup' && page === 'signup' ? (
            <>
              {/* Platform URL: the server this app talks to (account, credits, jobs). */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Server className="h-4 w-4 text-emerald-400" />
                  <span className="text-sm font-medium text-white">Platform URL</span>
                </div>
                <p className="text-xs text-zinc-500 mb-3">
                  Where your Video Creator account, credits and jobs live — the server this app talks to.
                </p>
                <input
                  type="text"
                  inputMode="url"
                  value={platformUrl}
                  onChange={(e) => setPlatformUrl(e.target.value)}
                  placeholder="https://video-creator.example.workers.dev"
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </div>

              {/* Sign Up: email only (optional). Folder selection lives on its own page. */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Mail className="h-4 w-4 text-blue-400" />
                  <span className="text-sm font-medium text-white">Recovery email <span className="text-zinc-500 font-normal">(optional)</span></span>
                </div>
                <p className="text-xs text-zinc-500 mb-3">
                  Optional. Add an email to also recover by email. You'll get a one-time backup
                  code either way, so email is not required.
                </p>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </>
          ) : mode === 'signup' && page === 'folder' ? (
            <>
              {/* Folder (its own page, kept separate from signup) */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <FolderOpen className="h-4 w-4 text-amber-400" />
                  <span className="text-sm font-medium text-white">Where should project assets be saved?</span>
                </div>
                <p className="text-xs text-zinc-500 mb-3">
                  Video clips, images and project files stay in a folder you choose on your computer. Nothing is uploaded.
                </p>
                {!supportedFolder && (
                  <p className="mb-3 rounded-md border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-xs text-amber-200">
                    Your browser doesn't support choosing a folder directly. Use <b>Chrome, Edge or Opera</b> to pick a folder, or skip and set it later in Settings.
                  </p>
                )}
                {folderName ? (
                  <div className="flex items-center gap-2 rounded-md border border-emerald-700/50 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-200">
                    <CheckCircle2 className="h-4 w-4" /> Saved to <b className="ml-1">{folderName}</b>
                  </div>
                ) : (
                  <div className="rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-500">
                    No folder selected yet. You can skip and choose one later in Settings.
                  </div>
                )}
                <div className="mt-2">
                  <Button variant="outline" size="sm" onClick={chooseFolder} disabled={!supportedFolder}>
                    {folderName ? 'Change Folder' : 'Choose Folder'}
                  </Button>
                </div>
              </div>
            </>
          ) : mode === 'signup' && page === 'license' ? (
            <>
              {/* Licenses + open-source reminder + acknowledgement */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <FileText className="h-4 w-4 text-emerald-400" />
                  <span className="text-sm font-medium text-white">Licenses & acknowledgement</span>
                </div>
                <div className="rounded-md border border-zinc-800 bg-zinc-900/60 px-4 py-3 text-sm text-zinc-300 leading-relaxed">
                  <p>
                    Video Creator uses a mix of open-source and third-party components and AI models
                    (including LTX). Some models are under permissive licenses; others carry their own
                    terms.
                  </p>
                  <p className="mt-2">
                    <b className="text-zinc-100">By proceeding, you agree to use any open-source model in
                    accordance with its respective license</b>, and to comply with all relevant license
                    terms for the models you use.
                  </p>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  <Button variant="outline" size="sm" onClick={() => openDoc('license')} disabled={docLoading}>
                    {docLoading ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <FileText className="h-3.5 w-3.5 mr-1" />}
                    View License
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => openDoc('notices')} disabled={docLoading}>
                    {docLoading ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <FileText className="h-3.5 w-3.5 mr-1" />}
                    View Model Notices
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <KeyRound className="h-4 w-4 text-blue-400" />
                  <span className="text-sm font-medium text-white">Sign back in</span>
                </div>
                {/* method toggle */}
                <div className="grid grid-cols-2 gap-1 rounded-lg bg-zinc-900 p-1 mb-3">
                  <button
                    onClick={() => { setLoginMethod('email'); setError(null); setMessage(null); }}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${loginMethod === 'email' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
                  >
                    By email
                  </button>
                  <button
                    onClick={() => { setLoginMethod('backup'); setError(null); setMessage(null); }}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${loginMethod === 'backup' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
                  >
                    Backup code
                  </button>
                </div>

                {loginMethod === 'email' ? (
                  <div className="space-y-2">
                    <p className="text-xs text-zinc-500 mb-2">
                      We'll send a code to your email to verify it's you and restore your account.
                    </p>
                    <input
                      type="email"
                      value={recoveryEmail}
                      onChange={(e) => setRecoveryEmail(e.target.value)}
                      placeholder="you@example.com"
                      className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    {loginAwaitingCode && (
                      <input
                        type="text"
                        value={recoveryCode}
                        onChange={(e) => setRecoveryCode(e.target.value)}
                        placeholder="Recovery code"
                        className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    )}
                  </div>
                ) : (
                  <div className="space-y-2">
                    <p className="text-xs text-zinc-500 mb-2">
                      Enter the one-time backup code you saved at sign-up. No email needed.
                    </p>
                    <input
                      type="text"
                      value={backupInput}
                      onChange={(e) => setBackupInput(e.target.value.toUpperCase())}
                      placeholder="XXXX-XXXX-XXXX-XXXX"
                      className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm font-mono text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                    />
                  </div>
                )}
              </div>
            </>
          )}

          {error && (
            <div className="flex items-center gap-2 rounded-md bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">
              <AlertCircle className="h-4 w-4 flex-shrink-0" /> {error}
            </div>
          )}
          {message && !backupCode && (
            <div className="flex items-center gap-2 rounded-md bg-green-500/10 border border-green-500/30 px-3 py-2 text-sm text-green-300">
              <CheckCircle2 className="h-4 w-4 flex-shrink-0" /> {message}
            </div>
          )}
        </div>

        {/* Footer actions */}
        {!backupCode && (
          <div className="flex justify-between items-center gap-2 border-t border-zinc-800 px-6 py-4">
            <div className="flex gap-2">
              {mode === 'signup' && page === 'folder' && (
                <Button variant="ghost" onClick={() => { setError(null); setPage('signup') }}>
                  Back
                </Button>
              )}
              {mode === 'signup' && page === 'license' && (
                <Button variant="ghost" onClick={() => { setError(null); setPage('folder') }}>
                  Back
                </Button>
              )}
            </div>
            <div className="flex gap-2">
              {mode === 'signup' && page === 'signup' && (
                <Button onClick={createAccount} disabled={busy}>
                  {busy ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                  Create account
                </Button>
              )}
              {mode === 'signup' && page === 'folder' && (
                <Button onClick={() => { setError(null); setPage('license') }}>
                  Next
                </Button>
              )}
              {mode === 'signup' && page === 'license' && (
                <Button onClick={finish}>
                  <CheckCircle2 className="h-4 w-4 mr-1" />
                  I understand & agree — finish
                </Button>
              )}
              {mode === 'login' && (
                loginMethod === 'email' ? (
                  loginAwaitingCode ? (
                    <Button onClick={confirmRecovery} disabled={busy || !recoveryEmail.trim() || !recoveryCode.trim()}>
                      {busy ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                      Confirm & Sign In
                    </Button>
                  ) : (
                    <Button onClick={sendRecoveryCode} disabled={busy || !recoveryEmail.trim()}>
                      {busy ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                      Send Code
                    </Button>
                  )
                ) : (
                  <Button onClick={recoverByBackup} disabled={busy || !backupInput.trim()}>
                    {busy ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                    Recover with code
                  </Button>
                )
              )}
            </div>
          </div>
        )}
      </div>

      {/* In-card license / notices viewer */}
      {doc && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4" onClick={() => setDoc(null)}>
          <div
            className="w-full max-w-2xl h-[70vh] rounded-xl border border-zinc-700 bg-zinc-900 text-zinc-100 shadow-2xl flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
              <h3 className="text-sm font-semibold">{doc.title}</h3>
              <button onClick={() => setDoc(null)} className="rounded-md p-1 text-zinc-400 hover:text-white hover:bg-zinc-800" aria-label="Close">
                <X className="h-4 w-4" />
              </button>
            </div>
            <pre className="flex-1 overflow-auto px-5 py-4 text-xs text-zinc-300 whitespace-pre-wrap font-mono">{doc.body}</pre>
            <div className="flex justify-end border-t border-zinc-800 px-5 py-3">
              <Button onClick={() => setDoc(null)}>Close</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
